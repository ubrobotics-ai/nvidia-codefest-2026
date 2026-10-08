"""Build the 10-class detection set (v2b: v2 + the Isaac warehouse train frames back, boxes pseudo-labelled by v2) (RF-DETR / Roboflow COCO layout) from COCO 2017, LVIS v1 and
the Isaac rescue renders.

Classes (category ids 1..10):
  1 person          COCO person; Isaac: every person class, any pose
  2 person_lying    Isaac lying classes (a second box on the same person as class 1)
  3 backpack  4 handbag  5 suitcase  6 dog  7 car  8 truck    COCO
  9 box             LVIS box + crate + carton (generic containers, not only shipping cartons)
 10 object          catch-all: every other COCO class, and Isaac distractors (random primitive
                    shapes). The detector boxes these; Cosmos3-Edge names them on the rover.

Splits:
  train: every COCO train2017 image with at least one non-crowd annotation, plus the Isaac rebal
         train split (91,647 frames, same seed-1234 split as every earlier arm).
  valid: every annotated COCO val2017 image, plus the held-out Isaac hospital split
         (22,060 frames), plus every COCO train2017 image that LVIS v1 val labels with a box/crate/
         carton (LVIS val is drawn mostly from COCO train2017; without this the box class would
         have only ~100 held-out instances). Hospital stays out of train.

Known label gaps, not fixed here (a pre-labelling pass is the planned fix):
  - LVIS is federated: most COCO images have boxes nobody labelled.
  - Isaac frames have unlabelled boxes, bags and vehicles (the scenes' own props).
  - COCO labels lying people as person only.
COCO crowd annotations (iscrowd=1) are dropped.
"""
import json, os, collections

T = "/storage/hackathon_teams/omc-team15/codefest"
COCO = f"{T}/data_coco"
ISA = f"{T}/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd/annotations"
REBAL = f"{T}/ubr-det/data/coco_rebal"
OUT = f"{T}/multiclass/data_v2b"
PSEUDO = f"{T}/multiclass/pseudo/boxes.jsonl"   # pseudolabel_boxes.py: v2, box score >= 0.5, short side >= 12 px
BOXES = f"{T}/data_boxes"   # ubr-physical-ai/isaac-sdg-warehouse-boxes @ 3988dba6
MIN_BOX_SIDE = 12          # px at 1280x720; about 6 px at the 640 input, the smallest worth learning

CATS = ["person", "person_lying", "backpack", "handbag", "suitcase", "dog", "car", "truck", "box", "object"]
CID = {n: i + 1 for i, n in enumerate(CATS)}
COCO_MAP = {"person": "person", "backpack": "backpack", "handbag": "handbag", "suitcase": "suitcase",
            "dog": "dog", "car": "car", "truck": "truck"}
LVIS_BOX = {"box", "crate", "carton"}


class Split:
    def __init__(self, name):
        self.name, self.images, self.anns, self.n = name, [], [], collections.Counter()
        os.makedirs(f"{OUT}/{name}", exist_ok=True)

    def add(self, src_path, file_name, w, h, boxes):
        iid = len(self.images) + 1
        link = f"{OUT}/{self.name}/{file_name}"
        if not os.path.islink(link):
            os.symlink(src_path, link)
        elif self.name == "valid":
            other = f"{OUT}/train/{file_name}"
            if os.path.islink(other): os.remove(other)
        self.images.append({"id": iid, "file_name": file_name, "width": w, "height": h})
        for cat, (x, y, bw, bh) in boxes:
            if bw <= 1 or bh <= 1:
                continue
            self.anns.append({"id": len(self.anns) + 1, "image_id": iid, "category_id": CID[cat],
                              "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                              "area": round(bw * bh, 2), "iscrowd": 0})
            self.n[cat] += 1

    def save(self):
        json.dump({"images": self.images, "annotations": self.anns,
                   "categories": [{"id": CID[c], "name": c, "supercategory": "object"} for c in CATS]},
                  open(f"{OUT}/{self.name}/_annotations.coco.json", "w"))
        print(f"{self.name}: {len(self.images)} images, boxes {dict(self.n)}", flush=True)


# LVIS box labels, keyed by COCO image id (LVIS ids are COCO ids)
lvis_boxes = collections.defaultdict(list)
lvis_val_box_imgs = set()
for sp in ("train", "val"):
    d = json.load(open(f"{COCO}/lvis_v1_{sp}.json"))
    keep = {c["id"] for c in d["categories"] if c["name"] in LVIS_BOX}
    for a in d["annotations"]:
        if a["category_id"] in keep:
            lvis_boxes[a["image_id"]].append(("box", a["bbox"]))
            if sp == "val":
                lvis_val_box_imgs.add(a["image_id"])
    del d
print("LVIS images with box/crate/carton:", len(lvis_boxes), flush=True)

train, valid = Split("train"), Split("valid")

# COCO 2017
for sp, split in (("train2017", train), ("val2017", valid)):
    d = json.load(open(f"{COCO}/annotations/instances_{sp}.json"))
    cname = {c["id"]: c["name"] for c in d["categories"]}
    per = collections.defaultdict(list)
    for a in d["annotations"]:
        n = cname[a["category_id"]]
        if not a.get("iscrowd", 0):
            per[a["image_id"]].append((COCO_MAP.get(n, "object"), a["bbox"]))
    for im in d["images"]:
        b = per.get(im["id"], []) + lvis_boxes.get(im["id"], [])
        if b:
            dst = valid if im["id"] in lvis_val_box_imgs else split
            dst.add(f"{COCO}/{sp}/{im['file_name']}", f"coco_{im['file_name']}", im["width"], im["height"], b)
    print(f"COCO {sp} done", flush=True)
    del d

# Isaac: full annotations, restricted to the rebal split's frames
full = {}
for sp in ("train", "validation"):
    d = json.load(open(f"{ISA}/coco_{sp}.json"))
    cname = {c["id"]: c["name"] for c in d["categories"]}
    byid = {i["id"]: i for i in d["images"]}
    for i in d["images"]:
        full[i["file_name"].split("/")[-1]] = (i["width"], i["height"], [])
    for a in d["annotations"]:
        n = cname[a["category_id"]]
        fn = byid[a["image_id"]]["file_name"].split("/")[-1]
        if n == "distractor":
            full[fn][2].append(("object", a["bbox"]))
            continue
        full[fn][2].append(("person", a["bbox"]))
        if "lying" in n:
            full[fn][2].append(("person_lying", a["bbox"]))
    del d
pseudo = {d["file_name"]: d["boxes"] for d in map(json.loads, open(PSEUDO))}
for sp, split in (("train", train), ("val", valid)):
    r = json.load(open(f"{REBAL}/{sp}.json"))
    for im in r["images"]:
        fn = im["file_name"].split("/")[-1]
        w, h, b = full[fn]
        if sp == "train" and fn.split("_")[1] == "warehouse":
            b = b + [("box", p[:4]) for p in pseudo[fn]]   # v2b: the scenes' unlabelled boxes, labelled by v2
        split.add(f"{REBAL}/{sp}/{fn}", f"isaac_{fn}", w, h, b)
    print(f"Isaac {sp} done", flush=True)

# v2: warehouse box renders. full_warehouse -> train; warehouse_multiple_shelves -> valid (a stage never seen in training).
for sp in ("train", "validation"):
    d = json.load(open(f"{BOXES}/annotations/coco_{sp}.json"))
    cname = {c["id"]: c["name"] for c in d["categories"]}
    per = collections.defaultdict(list)
    for a in d["annotations"]:
        n = cname[a["category_id"]]; x, y, w, h = a["bbox"]
        if n == "box":
            if min(w, h) >= MIN_BOX_SIDE: per[a["image_id"]].append(("box", a["bbox"]))
        elif n == "person":
            per[a["image_id"]].append(("person", a["bbox"]))
        elif n == "person_lying":
            per[a["image_id"]].append(("person", a["bbox"])); per[a["image_id"]].append(("person_lying", a["bbox"]))
    for im in d["images"]:
        fn = im["file_name"]
        dst = valid if "multiple_shelves" in fn else train
        dst.add(f"{BOXES}/{sp}/{fn}", f"wbox_{fn}", im["width"], im["height"], per.get(im["id"], []))
    print(f"box renders {sp} done", flush=True)
train.save(); valid.save()
