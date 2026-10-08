"""Detector v2 vs v1 on the v2 valid split. COCO AP per class:
  COCO classes + object on the COCO images; box on (a) the LVIS-labelled COCO images and (b) the held-out
  warehouse_multiple_shelves renders (a stage never seen in training); person / person_lying on the renders too.
Usage: eval_mc2.py   (reads preds/mc_v2.s*.coco.json and preds/mc_v1wbox.s*.coco.json)"""
import json, glob, io, contextlib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
gt = COCO(f"{M}/data_v2/valid/_annotations.coco.json")
cats = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}; cid = {v: k for k, v in cats.items()}
fn = {i: im["file_name"] for i, im in gt.imgs.items()}
coco_imgs = [i for i, f in fn.items() if f.startswith("coco_")]
wbox_imgs = [i for i, f in fn.items() if f.startswith("wbox_")]
lvis_box_imgs = sorted({a["image_id"] for a in gt.anns.values() if a["category_id"] == cid["box"] and fn[a["image_id"]].startswith("coco_")})
def load(tag):
    r = []
    for p in sorted(glob.glob(f"{M}/preds/mc_{tag}.s*.coco.json")): r += json.load(open(p))
    return r
def ap(res, imgs, cat):
    ids = set(imgs); dt = gt.loadRes([x for x in res if x["image_id"] in ids and x["category_id"] == cat] or [{"image_id": imgs[0], "category_id": cat, "bbox": [0, 0, 1, 1], "score": 0.001}])
    E = COCOeval(gt, dt, "bbox"); E.params.imgIds = imgs; E.params.catIds = [cat]; E.params.maxDets = [1, 10, 300]
    with contextlib.redirect_stdout(io.StringIO()): E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0], E.stats[1], E.stats[8]
v2, v1 = load("v2"), load("v1wbox")
print(f"images: COCO {len(coco_imgs)}, LVIS-box {len(lvis_box_imgs)}, held-out warehouse renders {len(wbox_imgs)}")
print(f"{'class / set':<34}{'AP':>7}{'AP50':>7}{'AR':>7}")
for name in ["person", "backpack", "handbag", "suitcase", "dog", "car", "truck", "object"]:
    a = ap(v2, coco_imgs, cid[name]); print(f"v2 {name:<31}{a[0]:7.3f}{a[1]:7.3f}{a[2]:7.3f}")
a = ap(v2, lvis_box_imgs, cid["box"]); print(f"v2 {'box on LVIS images':<31}{a[0]:7.3f}{a[1]:7.3f}{a[2]:7.3f}")
for tag, res in (("v1", v1), ("v2", v2)):
    for name in ["box", "person", "person_lying"]:
        a = ap(res, wbox_imgs, cid[name]); print(f"{tag} {name + ' on unseen warehouse':<31}{a[0]:7.3f}{a[1]:7.3f}{a[2]:7.3f}")
