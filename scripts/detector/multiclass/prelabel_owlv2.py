"""Fill the label gaps of the 9-class train split with OWLv2 (google/owlv2-base-patch16-ensemble,
Apache-2.0), an open-vocabulary detector driven by text queries.

Which classes are queried depends on the source, because each source already labels some classes
exhaustively:
  isaac_* frames: backpack, handbag, suitcase, dog, car, truck, box (people are rendered ground truth)
  coco_*  images: box only (LVIS box labels are federated, i.e. missing on most COCO images;
                  the other 7 COCO classes are already labelled by COCO)

A pseudo-box is kept when: score >= THR, it does not duplicate an existing label of the same class
(IoU >= 0.5), and, for "box" on Isaac frames, it does not sit on an Isaac distractor primitive
(IoU >= 0.5 with any distractor box). Output: one JSONL row per image with the added boxes.

Usage: prelabel_owlv2.py SHARD NSHARDS
"""
import json, os, sys, collections
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

T = "/storage/hackathon_teams/omc-team15/codefest"
DATA = f"{T}/multiclass/data/train"
ISA = f"{T}/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd/annotations"
OUT = f"{T}/multiclass/prelabel"
THR = 0.30
QUERIES = {"backpack": ["a backpack"], "handbag": ["a handbag"], "suitcase": ["a suitcase"],
           "dog": ["a dog"], "car": ["a car"], "truck": ["a truck"],
           "box": ["a cardboard box", "a wooden crate"]}
ISAAC_Q = list(QUERIES)
COCO_Q = ["box"]

shard, nsh = int(sys.argv[1]), int(sys.argv[2])
CALIB = len(sys.argv) > 3 and sys.argv[3] == "calib"   # raw boxes at a low threshold on valid, for per-class thresholds
if CALIB:
    DATA = f"{T}/multiclass/data/valid"; THR = 0.05
os.makedirs(OUT, exist_ok=True)
d = json.load(open(f"{DATA}/_annotations.coco.json"))
cname = {c["id"]: c["name"] for c in d["categories"]}
labels = collections.defaultdict(list)
for a in d["annotations"]:
    labels[a["image_id"]].append((cname[a["category_id"]], a["bbox"]))
images = d["images"][shard::nsh]
if CALIB:   # COCO images only: those carry COCO (and, for LVIS-val images, box) ground truth
    images = [i for i in images if i["file_name"].startswith("coco_")]
    ISAAC_Q = list(QUERIES); COCO_Q = list(QUERIES)

# Isaac distractor boxes, by file name
distr = collections.defaultdict(list)
for sp in ("train", "validation"):
    j = json.load(open(f"{ISA}/coco_{sp}.json"))
    did = [c["id"] for c in j["categories"] if c["name"] == "distractor"][0]
    fn = {i["id"]: "isaac_" + i["file_name"].split("/")[-1] for i in j["images"]}
    for a in j["annotations"]:
        if a["category_id"] == did:
            distr[fn[a["image_id"]]].append(a["bbox"])
    del j


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0., min(ax2, bx2) - max(a[0], b[0])); ih = max(0., min(ay2, by2) - max(a[1], b[1]))
    i = iw * ih; u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.


proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval().half()

stats = collections.Counter()
with open(f"{OUT}/{'calib' if CALIB else 'prelabel'}.s{shard}.jsonl", "w") as f:
    for k in range(0, len(images), 16):
        batch = images[k:k + 16]
        for group in ("isaac", "coco"):
            sub = [im for im in batch if im["file_name"].startswith(group + "_")]
            if not sub:
                continue
            classes = ISAAC_Q if group == "isaac" else COCO_Q
            texts = [q for c in classes for q in QUERIES[c]]
            owner = [c for c in classes for _ in QUERIES[c]]
            pil = [Image.open(f"{DATA}/{im['file_name']}").convert("RGB") for im in sub]
            inp = proc(text=[texts] * len(pil), images=pil, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                out = model(**inp)
            # OWLv2 pads to a square, so boxes are rescaled with the padded size max(w, h)
            sizes = torch.tensor([[max(im["width"], im["height"])] * 2 for im in sub], device="cuda")
            res = proc.post_process_grounded_object_detection(out, threshold=THR, target_sizes=sizes)
            for im, r in zip(sub, res):
                added = []
                for s, l, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                    c = owner[l]
                    x1, y1, x2, y2 = b
                    x1, y1 = max(0., x1), max(0., y1)
                    x2, y2 = min(im["width"], x2), min(im["height"], y2)
                    bb = [x1, y1, x2 - x1, y2 - y1]
                    if bb[2] < 4 or bb[3] < 4:
                        continue
                    if not CALIB and any(cc == c and iou(bb, eb) >= 0.5 for cc, eb in labels[im["id"]] + [(a, b2) for a, _, b2 in added]):
                        stats["dup"] += 1; continue
                    if c == "box" and group == "isaac" and any(iou(bb, db) >= 0.5 for db in distr.get(im["file_name"], [])):
                        stats["on_distractor"] += 1; continue
                    added.append((c, round(s, 3), [round(v, 1) for v in bb]))
                    stats[f"{group}:{c}"] += 1
                f.write(json.dumps({"file_name": im["file_name"], "added": added}) + "\n")
        if k % 3200 == 0:
            print(shard, k, len(images), dict(stats), flush=True)
print("done", shard, dict(stats), flush=True)
