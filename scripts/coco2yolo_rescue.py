#!/usr/bin/env python3
"""Convert the isaac-sdg-rescue-target COCO annotations to YOLO labels and write Ultralytics data YAMLs.

  python scripts/coco2yolo_rescue.py --src $TEAM/data/isaac-sdg-rescue-target --dst $TEAM/data/rescue_yolo

Writes <dst>/{train,validation}/{images -> symlinks, labels}/, and YAMLs:
  rescue_2cls.yaml   classes person_lying_vest, distractor
  rescue_1cls.yaml   person_lying_vest only (distractor boxes dropped)
  rescue_1cls_val_<view>.yaml / rescue_2cls_val_<view>.yaml   validation sliced by camera view (orbit / roverview / roverview_far)
  rescue_*_val_env_<environment>.yaml                          validation sliced by environment
Slices come from metadata.jsonl (view / environment per image) and are what the sweep is scored on.
"""
import argparse, json, os, collections
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    classes = {int(k): v for k, v in json.load(open(src / "classes.json")).items()}
    names = [classes[i] for i in sorted(classes)]
    meta = {}
    for split in ("train", "validation"):
        for line in open(src / split / "metadata.jsonl"):
            r = json.loads(line); meta[r["file_name"]] = r
    slices = collections.defaultdict(list)
    for split in ("train", "validation"):
        coco = json.load(open(src / "annotations" / f"coco_{split}.json"))
        anns = collections.defaultdict(list)
        for an in coco["annotations"]:
            anns[an["image_id"]].append(an)
        for variant in ("2cls", "1cls"):
            img_dir = dst / variant / split / "images"; lab_dir = dst / variant / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
            n_img = n_box = 0
            for im in coco["images"]:
                fn = Path(im["file_name"]).name
                link = img_dir / fn
                if not link.exists():
                    os.symlink((src / split / fn).resolve(), link)
                W, H = im["width"], im["height"]
                lines = []
                for an in anns[im["id"]]:
                    c = an["category_id"]
                    if variant == "1cls" and c != 0:
                        continue
                    x, y, w, h = an["bbox"]
                    lines.append(f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                (lab_dir / (Path(fn).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1; n_box += len(lines)
                if split == "validation":
                    m = meta[fn]
                    slices[(variant, "view", m["view"])].append(str(link))
                    slices[(variant, "env", m["environment"])].append(str(link))
            print(f"{variant} {split}: {n_img} images, {n_box} boxes")
    for variant in ("2cls", "1cls"):
        vnames = names if variant == "2cls" else names[:1]
        base = {"path": str(dst / variant), "train": "train/images", "val": "validation/images", "names": dict(enumerate(vnames))}
        (dst / f"rescue_{variant}.yaml").write_text(yaml_dump(base))
        for (v, kind, key), files in sorted(slices.items()):
            if v != variant: continue
            lst = dst / variant / f"val_{kind}_{key}.txt"; lst.write_text("\n".join(sorted(files)) + "\n")
            y = dict(base); y["val"] = str(lst)
            (dst / f"rescue_{variant}_val_{kind}_{key}.yaml").write_text(yaml_dump(y))
            print(f"slice {variant} {kind}={key}: {len(files)} val images")
    print("wrote", sorted(p.name for p in dst.glob("*.yaml")))

def yaml_dump(d):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{k}:"); out += [f"  {i}: {n}" for i, n in v.items()]
        else:
            out.append(f"{k}: {v}")
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    main()
