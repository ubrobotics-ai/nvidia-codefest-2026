"""10-class RF-DETR-S@640 inference over a multiclass valid split, CPU or GPU.

Usage: infer_mc_any.py CKPT TAG DATA_DIR SHARD NSHARDS [FILE_PREFIX]
Writes preds/mc_<TAG>.s<k>.jsonl (harness JSONL, original pixels, conf >= 0.005, cls = channel) and
preds/mc_<TAG>.s<k>.coco.json (COCO results, category_id = channel + 1). FILE_PREFIX limits the images
(e.g. "wbox_" for the warehouse box renders only).
"""
import json, os, sys
import torch
from PIL import Image
from rfdetr import RFDETR

ckpt, tag, data, shard, nsh = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
prefix = sys.argv[6] if len(sys.argv) > 6 else ""
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
d = json.load(open(f"{data}/valid/_annotations.coco.json"))
ims = [i for i in d["images"] if i["file_name"].startswith(prefix)][shard::nsh]
m = RFDETR.from_checkpoint(ckpt)
res = []
with open(f"{M}/preds/mc_{tag}.s{shard}.jsonl", "w") as f:
    for i in range(0, len(ims), 8):
        chunk = ims[i:i + 8]
        pil = [Image.open(f"{data}/valid/{im['file_name']}").convert("RGB") for im in chunk]
        with torch.inference_mode():
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
        if i % 800 == 0:
            print(shard, i, len(ims), flush=True)
json.dump(res, open(f"{M}/preds/mc_{tag}.s{shard}.coco.json", "w"))
print("done", shard, flush=True)
