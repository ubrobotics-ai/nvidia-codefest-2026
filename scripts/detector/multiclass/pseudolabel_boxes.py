"""v2b step 1: label boxes in the Isaac rebal warehouse train frames (dropped from v2 because their boxes were
unlabelled) with detector v2. Keep box (channel 8) at score >= 0.5 (80.7% precision on the unseen warehouse
renders) and short side >= 12 px. Usage: pseudolabel_boxes.py SHARD NSHARDS"""
import json, sys
import torch
from PIL import Image
from rfdetr import RFDETR
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
T = "/storage/hackathon_teams/omc-team15/codefest"
REBAL = f"{T}/ubr-det/data/coco_rebal"
names = [i["file_name"].split("/")[-1] for i in json.load(open(f"{REBAL}/train.json"))["images"]]
names = sorted(n for n in names if n.split("_")[1] == "warehouse")[shard::nsh]
m = RFDETR.from_checkpoint(f"{T}/multiclass/runs/mc_v2/last_ema.pth")
with open(f"{T}/multiclass/pseudo/boxes.s{shard}.jsonl", "w") as f:
    for i in range(0, len(names), 16):
        chunk = names[i:i + 16]
        with torch.inference_mode():
            dets = m.predict([Image.open(f"{REBAL}/train/{n}").convert("RGB") for n in chunk], threshold=0.5)
        dets = dets if isinstance(dets, list) else [dets]
        for n, dt in zip(chunk, dets):
            bx = [[round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1), round(float(c), 3)]
                  for (x1, y1, x2, y2), c, k in zip(dt.xyxy, dt.confidence, dt.class_id) if int(k) == 8 and min(x2 - x1, y2 - y1) >= 12]
            f.write(json.dumps({"file_name": n, "boxes": bx}) + "\n")
        if i % 3200 == 0: print(shard, i, len(names), flush=True)
print("done", shard, flush=True)
