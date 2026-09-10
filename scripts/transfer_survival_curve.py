#!/usr/bin/env python3
"""Label survival as a function of frame index — the mechanism, not just the verdict.

  python scripts/transfer_survival_curve.py --clips <src-root> --runs <dir> --arm transfer --out curve.csv \
      [--indices 0 23 46 69 92] [--strengths 0.4 0.7 1.0]

Prediction on record before running (stated 2026-09-10 03:40, from the project owner):
    a mask-conditioned Transfer holds the person on every frame  -> survival > 0.9, FLAT in frame index
    an image-conditioned I2V is anchored only at frame 0         -> survival < 0.5 by frame 92, DECAYING
A flat-but-low Transfer curve falsifies the first without supporting the second: it would mean the mask is being
ignored from the start rather than drifting away.

The judge is Grounding-DINO with the prompt "a person lying on the ground". The generic prompt "person" scores a
prone figure ~0.27 against ~0.76 for the specific one on the same frame, so the generic prompt would report a
collapse that is the detector's, not the generator's.
"""
import argparse, csv, glob, json, os
from pathlib import Path

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    if x2 <= x1 or y2 <= y1: return 0.0
    i = (x2-x1)*(y2-y1)
    return i/(aw*ah + bw*bh - i)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--arm", default="transfer")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 23, 46, 69, 92])
    ap.add_argument("--strengths", nargs="+", default=["0.4", "0.7", "1.0"])
    ap.add_argument("--prompt", default="a person lying on the ground")
    ap.add_argument("--box-threshold", type=float, default=0.15); ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    import numpy as np, torch
    from PIL import Image
    import imageio.v3 as iio
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda").eval()

    def detect(img):
        inp = proc(images=[img], text=[[a.prompt]], return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inp)
        r = proc.post_process_grounded_object_detection(o, inp["input_ids"], threshold=a.box_threshold,
                                                        text_threshold=0.20, target_sizes=[img.size[::-1]])[0]
        return [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in r["boxes"].tolist()]

    rows = []
    for s in a.strengths:
        d = os.path.join(a.runs, f"s{s}") if os.path.isdir(os.path.join(a.runs, f"s{s}")) else a.runs
        for vp in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            clip = Path(vp).stem
            cj = os.path.join(a.clips, clip, "coco.json")
            if not os.path.exists(cj): continue
            coco = json.load(open(cj))
            n2i = {os.path.basename(im["file_name"]): im["id"] for im in coco["images"]}
            dims = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
            anns = {}
            for an in coco["annotations"]: anns.setdefault(an["image_id"], []).append(an)
            vid = iio.imread(vp, plugin="pyav"); T, H, W = vid.shape[:3]
            for i in a.indices:
                if i >= T: continue
                iid = n2i.get(f"frame_{i:04d}.jpg")
                if iid is None: continue
                ga = anns.get(iid, [])
                if not ga: continue
                W0, H0 = dims[iid]; sx, sy = W/W0, H/H0
                det = detect(Image.fromarray(vid[i]).convert("RGB"))
                for an in ga:
                    x, y, w, h = an["bbox"]
                    g = [x*sx, y*sy, w*sx, h*sy]
                    best = max((iou(g, dd) for dd in det), default=0.0)
                    rows.append({"arm": a.arm, "strength": s, "clip": clip, "frame": i,
                                 "best_iou": round(best, 3), "kept": int(best >= a.iou)})
        print(f"  s={s} done ({sum(1 for r in rows if r['strength']==s)} observations)", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import statistics as st
    print(f"\n== survival by frame index (IoU>={a.iou}) ==")
    hdr = "  " + "strength".ljust(10) + "".join(f"f{i:<8d}" for i in a.indices) + "  slope"
    print(hdr)
    for s in a.strengths:
        cells, vals = [], []
        for i in a.indices:
            r = [x for x in rows if x["strength"] == s and x["frame"] == i]
            v = (sum(x["kept"] for x in r)/len(r)) if r else float("nan")
            vals.append(v); cells.append(f"{v:.2f}({len(r)})".ljust(9))
        slope = (vals[-1]-vals[0]) if vals and vals[0] == vals[0] else float("nan")
        print("  " + s.ljust(10) + "".join(cells) + f"  {slope:+.2f}")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
