"""v3 kit: map every image of the v2b dataset (data_v2b/{train,valid}) to its source, so the dataset can be rebuilt
by symlinks on another cluster from public / HF downloads, without re-running the builders.
Writes manifest.jsonl: {"split", "name", "src", "rel"} with src one of coco (images.cocodataset.org 2017 zips),
rescue (ubr-physical-ai/isaac-sdg-rescue-target @ 92bb894, tar shards), wbox (ubr-physical-ai/isaac-sdg-warehouse-boxes
@ 3988dba) and rel the path inside that source."""
import json, os
T = "/storage/hackathon_teams/omc-team15/codefest"
D = f"{T}/multiclass/data_v2b"
ISA = f"{T}/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd/annotations"
rescue = {}
for sp in ("train", "validation"):
    for i in json.load(open(f"{ISA}/coco_{sp}.json"))["images"]:
        rescue[os.path.basename(i["file_name"])] = i["file_name"]          # e.g. train/xxx.jpg inside the tars
out = open(f"{T}/multiclass/v3_kit/manifest.jsonl", "w"); n = {}
for split in ("train", "valid"):
    for im in json.load(open(f"{D}/{split}/_annotations.coco.json"))["images"]:
        name = im["file_name"]; tgt = os.readlink(f"{D}/{split}/{name}")
        if name.startswith("coco_"):
            src, rel = "coco", tgt.split("/data_coco/")[1]                    # train2017/xxx.jpg or val2017/xxx.jpg
        elif name.startswith("isaac_"):
            src, rel = "rescue", rescue[os.path.basename(tgt)]
        elif name.startswith("wbox_"):
            src, rel = "wbox", tgt.split("/data_boxes/")[1]                   # train/xxx.jpg or validation/xxx.jpg
        else:
            raise SystemExit(f"unknown prefix {name}")
        out.write(json.dumps({"split": split, "name": name, "src": src, "rel": rel}) + "\n"); n[(split, src)] = n.get((split, src), 0) + 1
print(n)
