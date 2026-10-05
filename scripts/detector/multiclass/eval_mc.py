"""Per-class COCO AP for the 10-class model on the valid split.

  COCO classes + object: every COCO image in valid (val2017, plus the train2017 images moved there
                         for LVIS box; both carry exhaustive COCO labels)
  box:                   images that carry LVIS box labels (LVIS is federated, so box AP is computed
                         only where boxes are known to be labelled; optimistic, as with any LVIS
                         federated evaluation)
Isaac hospital frames are scored separately with the harness replica (lying-casualty recall).
Usage: eval_mc.py TAG
"""
import json, sys, glob, io, contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

tag = sys.argv[1]
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
gt = COCO(f"{M}/data/valid/_annotations.coco.json")
res = []
for p in sorted(glob.glob(f"{M}/preds/mc_{tag}.s*.coco.json")):
    res += json.load(open(p))
cats = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
coco_imgs = [i for i, im in gt.imgs.items() if im["file_name"].startswith("coco_")]
box_cat = [k for k, v in cats.items() if v == "box"][0]
box_imgs = sorted({a["image_id"] for a in gt.anns.values() if a["category_id"] == box_cat})
coco_ids = set(coco_imgs)
dt = gt.loadRes([r for r in res if r["image_id"] in coco_ids])

print(f"{'class':<14}{'AP50:95':>9}{'AP50':>8}{'AR100':>8}   images")
for cid, name in cats.items():
    if name == "person_lying":
        continue  # Isaac-only class: scored by the harness
    imgs = box_imgs if name == "box" else coco_imgs
    E = COCOeval(gt, dt, "bbox")
    E.params.imgIds = imgs; E.params.catIds = [cid]
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate(); E.accumulate(); E.summarize()
    print(f"{name:<14}{E.stats[0]:>9.3f}{E.stats[1]:>8.3f}{E.stats[8]:>8.3f}   {len(imgs)}")
