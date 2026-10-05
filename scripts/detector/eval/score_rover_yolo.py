"""Stock yolo26n (COCO, the rover's deployed detector) on the held-out hospital val split.

Reproduces the rover's path: the YOLO26 end-to-end ONNX export (one-to-one head, output
[1,300,6] = x0,y0,x1,y1,conf,cls in 640-letterbox pixels), 640x640 letterbox, and the rover's
decode_head 6-column branch: a confidence filter only, no NMS. Writes harness JSONL with every
person (class 0) box at conf >= 0.005, so it can be scored at matched FA and at the rover's own
operating point (conf 0.25)."""
import json, sys, os, hashlib
import numpy as np, cv2, onnxruntime as ort, torch, torchvision
from ultralytics import YOLO
D = "/storage/hackathon_teams/omc-team15/codefest/rover_yolo"
VAL = "/storage/hackathon_teams/omc-team15/codefest/ubr-det/data/coco_rebal"
pt = "/storage/hackathon_teams/omc-team15/codefest/ubr-det/weights/yolo26n.pt"
onnx_path = f"{D}/yolo26n_e2e.onnx"
if not os.path.exists(onnx_path):
    p = YOLO(pt).export(format="onnx", imgsz=640, opset=12, nms=False)   # nms=False selects the end2end (one-to-one) head; opset as the rover file
    os.replace(p, onnx_path)
print("onnx sha256", hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()[:16], flush=True)
s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp = s.get_inputs()[0].name
print("io", [(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()], flush=True)
names = [im["file_name"] for im in json.load(open(f"{VAL}/val.json"))["images"]]
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
if nsh == 0: sys.exit(0)  # export-only call
names = names[shard::nsh]
with open(f"{D}/preds.s{shard}.jsonl", "w") as f:
    for k, n in enumerate(names):
        im = cv2.imread(f"{VAL}/val/{n}"); h, w = im.shape[:2]
        r = min(640 / h, 640 / w); nh, nw = round(h * r), round(w * r)
        top, left = (640 - nh) // 2, (640 - nw) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        o = s.run(None, {inp: x})[0][0]                      # [300, 6]: x0,y0,x1,y1,conf,cls
        assert o.shape[-1] == 6, o.shape
        out = []
        for x0, y0, x1, y1, cf, c in o:
            if int(c) != 0 or cf < 0.005: continue
            x0, x1 = np.clip([(x0 - left) / r, (x1 - left) / r], 0, w); y0, y1 = np.clip([(y0 - top) / r, (y1 - top) / r], 0, h)
            out.append({"conf": round(float(cf), 4), "cls": 0, "bbox": [round(float(x0), 1), round(float(y0), 1), round(float(x1 - x0), 1), round(float(y1 - y0), 1)]})
        f.write(json.dumps({"file_name": n, "boxes": out}) + "\n")
        if k % 2000 == 0: print(shard, k, len(names), flush=True)
print("done", shard, flush=True)
