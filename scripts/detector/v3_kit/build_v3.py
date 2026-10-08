"""Rebuild the v2b dataset by symlinks and add the re-textured renders -> data_v3 (Roboflow COCO layout for RF-DETR).
Usage: build_v3.py OUT COCO_DIR RESCUE_DIR WBOX_DIR RETEX_DIR
  COCO_DIR   has train2017/ and val2017/ (images.cocodataset.org zips)
  RESCUE_DIR has train/ and validation/ (isaac-sdg-rescue-target @ 92bb894 tar shards, extracted)
  WBOX_DIR   has train/ and validation/ (isaac-sdg-warehouse-boxes @ 3988dba)
  RETEX_DIR  has <render basename>.jpg for each re-textured render (same size as the source, 1280x720); pass "none"
             to rebuild v2b exactly.
Each re-textured render enters TRAIN as a new image 'wbt_<basename>' with the source render's labels copied unchanged
(the source stays in too). Valid is v2b's valid unchanged, so v3 scores compare directly with v1 / v2 / v2b."""
import json, os, sys
out, roots = sys.argv[1], {"coco": sys.argv[2], "rescue": sys.argv[3], "wbox": sys.argv[4]}
retex = sys.argv[5]
for split in ("train", "valid"):
    os.makedirs(f"{out}/{split}", exist_ok=True)
missing = 0
for m in map(json.loads, open("manifest.jsonl")):
    src = os.path.join(roots[m["src"]], m["rel"]); dst = f"{out}/{m['split']}/{m['name']}"
    if not os.path.exists(src): missing += 1; continue
    if not os.path.lexists(dst): os.symlink(src, dst)
assert missing == 0, f"{missing} source images missing"
tr = json.load(open("train_annotations_v2b.coco.json"))
if retex != "none":
    by_name = {im["file_name"]: im for im in tr["images"]}
    anns = {}
    for a in tr["annotations"]: anns.setdefault(a["image_id"], []).append(a)
    nid, aid, added = max(by_name[n]["id"] for n in by_name) + 1, max(a["id"] for a in tr["annotations"]) + 1, 0
    for f in sorted(os.listdir(retex)):
        srcname = "wbox_" + f
        if srcname not in by_name: continue
        im = dict(by_name[srcname]); im["id"] = nid; im["file_name"] = "wbt_" + f
        os.symlink(os.path.abspath(f"{retex}/{f}"), f"{out}/train/wbt_{f}") if not os.path.lexists(f"{out}/train/wbt_{f}") else None
        tr["images"].append(im)
        for a in anns.get(by_name[srcname]["id"], []):
            b = dict(a); b["id"] = aid; b["image_id"] = nid; tr["annotations"].append(b); aid += 1
        nid += 1; added += 1
    print("re-textured renders added to train:", added)
json.dump(tr, open(f"{out}/train/_annotations.coco.json", "w"))
os.system(f"cp valid_annotations_v2b.coco.json {out}/valid/_annotations.coco.json")
print("train images", len(tr["images"]), "boxes", len(tr["annotations"]))
