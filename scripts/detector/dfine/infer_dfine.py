"""Harness JSONL from a D-FINE checkpoint (EMA weights): boxes [x,y,w,h] in original pixels,
conf floor 0.005, no extra NMS. Usage: infer_dfine.py CFG CKPT OUT.jsonl SHARD NSHARDS"""
import json, sys
sys.path.insert(0, "/storage/hackathon_teams/omc-team15/codefest/permissive/D-FINE")
import torch, torchvision.transforms as T
from PIL import Image
from src.core import YAMLConfig

cfg_p, ckpt, out, shard, nsh = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
VAL = "/storage/hackathon_teams/omc-team15/codefest/ubr-det/data/coco_rebal"
names = [im["file_name"] for im in json.load(open(f"{VAL}/val.json"))["images"]][shard::nsh]
cfg = YAMLConfig(cfg_p, resume=ckpt)
c = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg.model.load_state_dict(c["ema"]["module"] if "ema" in c else c["model"])
model, post = cfg.model.deploy().cuda().eval(), cfg.postprocessor.deploy()
tf = T.Compose([T.Resize((704, 1280)), T.ToTensor()])  # [H, W], as trained
with open(out, "w") as f, torch.no_grad():
    for i in range(0, len(names), 32):
        chunk = names[i:i + 32]
        ims = [Image.open(f"{VAL}/val/{n}").convert("RGB") for n in chunk]
        x = torch.stack([tf(im) for im in ims]).cuda()
        sizes = torch.tensor([[im.width, im.height] for im in ims]).cuda()  # D-FINE takes (w, h)
        labels, boxes, scores = post(model(x), sizes)
        for n, l, b, s in zip(chunk, labels.cpu(), boxes.cpu(), scores.cpu()):
            keep = s >= 0.005
            f.write(json.dumps({"file_name": n, "boxes": [
                {"conf": round(float(sc), 4), "cls": int(lb),
                 "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)]}
                for lb, (x1, y1, x2, y2), sc in zip(l[keep], b[keep], s[keep])]}) + "\n")
        if i % 3200 == 0:
            print(shard, i, len(names), flush=True)
print("done", shard, flush=True)
