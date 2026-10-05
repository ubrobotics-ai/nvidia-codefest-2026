"""Harness JSONL from an RF-DETR checkpoint: one line per val frame, boxes [x,y,w,h] in original
pixels, conf floor 0.005, no extra NMS. Usage: infer_rfdetr.py CKPT OUT.jsonl SHARD NSHARDS"""
import json, sys
from PIL import Image
from rfdetr import RFDETR

ckpt, out, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SZ = int(sys.argv[5]) if len(sys.argv) > 5 else None  # square inference size; None = trained 1280
VAL = "/storage/hackathon_teams/omc-team15/codefest/ubr-det/data/coco_rebal"
names = [im["file_name"] for im in json.load(open(f"{VAL}/val.json"))["images"]][shard::nsh]
try:
    m = RFDETR.from_checkpoint(ckpt)
except Exception as e:  # trainer .ckpt without model metadata: load weights into the trained variant
    print("from_checkpoint failed, falling back:", e, flush=True)
    from rfdetr import RFDETRMedium
    m = RFDETRMedium(pretrain_weights=ckpt, resolution=1280)
with open(out, "w") as f:
    for i in range(0, len(names), 16):
        chunk = names[i:i + 16]
        imgs = [Image.open(f"{VAL}/val/{n}").convert("RGB") for n in chunk]
        dets = m.predict(imgs, threshold=0.005, **({"shape": (SZ, SZ)} if SZ else {}))
        dets = dets if isinstance(dets, list) else [dets]
        for n, d in zip(chunk, dets):
            boxes = [{"conf": round(float(c), 4), "cls": int(k),
                      "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)]}
                     for (x1, y1, x2, y2), c, k in zip(d.xyxy, d.confidence, d.class_id)]
            f.write(json.dumps({"file_name": n, "boxes": boxes}) + "\n")
        if i % 1600 == 0:
            print(shard, i, len(names), flush=True)
print("done", shard, flush=True)
