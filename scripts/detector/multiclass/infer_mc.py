"""10-class RF-DETR-S@640 inference over the multiclass valid split.

Writes, per shard:
  preds/mc_<tag>.s<k>.jsonl  harness JSONL for every image (boxes [x,y,w,h] in original pixels,
                             conf >= 0.005, cls = model channel index), used for the Isaac hospital
                             frames by the harness replica
  preds/mc_<tag>.s<k>.coco.json  COCO results (category_id = channel + 1) for COCOeval
Usage: infer_mc.py CKPT TAG SHARD NSHARDS
"""
import json, sys
from PIL import Image
from rfdetr import RFDETR

ckpt, tag, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
d = json.load(open(f"{M}/data/valid/_annotations.coco.json"))
ims = d["images"][shard::nsh]
m = RFDETR.from_checkpoint(ckpt)
res = []
with open(f"{M}/preds/mc_{tag}.s{shard}.jsonl", "w") as f:
    for i in range(0, len(ims), 16):
        chunk = ims[i:i + 16]
        pil = [Image.open(f"{M}/data/valid/{im['file_name']}").convert("RGB") for im in chunk]
        dets = m.predict(pil, threshold=0.005)
        dets = dets if isinstance(dets, list) else [dets]
        for im, dt in zip(chunk, dets):
            boxes = []
            for (x1, y1, x2, y2), c, k in zip(dt.xyxy, dt.confidence, dt.class_id):
                b = [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)]
                boxes.append({"conf": round(float(c), 4), "cls": int(k), "bbox": b})
                res.append({"image_id": im["id"], "category_id": int(k) + 1, "bbox": b, "score": round(float(c), 4)})
            name = im["file_name"][len("isaac_"):] if im["file_name"].startswith("isaac_") else im["file_name"]
            f.write(json.dumps({"file_name": name, "boxes": boxes}) + "\n")
        if i % 3200 == 0:
            print(shard, i, len(ims), flush=True)
json.dump(res, open(f"{M}/preds/mc_{tag}.s{shard}.coco.json", "w"))
print("done", shard, flush=True)
