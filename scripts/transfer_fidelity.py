#!/usr/bin/env python3
"""Task B analysis: does the label track survive the translation, and does the target's defining colour?

  python scripts/transfer_fidelity.py --clips <src-clip-root> --runs <transfer-out> --out B_fidelity.csv \
      [--strengths 0.4 0.7 1.0] [--every 12]

For every sampled frame of every transferred clip:
  box kept   Grounding-DINO ("person") is run on the TRANSFERRED frame and matched against the source COCO box.
             IoU >= 0.5 means the annotation that shipped with the render still describes the translated frame.
             This is the box-preservation question, answered by a detector rather than by eye.
  vest kept  the source class is `person_lying_vest` -- defined by a hi-vis garment. We measure the fraction of
             pixels inside the box that are hi-vis (high saturation, hue in the yellow-orange band) in the source
             and in the transfer. The mask encodes silhouette, not colour, and the prompt rule forbids naming the
             subject, so nothing instructs Cosmos to keep the garment; this quantifies whether it does anyway.
"""
import argparse, csv, glob, json, os, sys
from pathlib import Path

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    if x2 <= x1 or y2 <= y1: return 0.0
    i = (x2-x1)*(y2-y1)
    return i / (aw*ah + bw*bh - i)

def hivis_fraction(rgb):
    """fraction of pixels that read as hi-vis yellow/orange: strong saturation, hue 35-75 deg, not dark"""
    import numpy as np
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(2); mn = a.min(2); v = mx; s = np.where(mx > 0, (mx-mn)/np.maximum(mx, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.zeros_like(mx)
    d = np.maximum(mx-mn, 1e-6)
    m = (mx == r); h[m] = ((g-b)[m]/d[m]) % 6
    m = (mx == g); h[m] = ((b-r)[m]/d[m]) + 2
    m = (mx == b); h[m] = ((r-g)[m]/d[m]) + 4
    h *= 60
    hi = (s > 0.45) & (v > 0.45) & (h >= 35) & (h <= 75)
    return float(hi.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--strengths", nargs="+", default=["0.4","0.7","1.0"])
    ap.add_argument("--every", type=int, default=12); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--teacher", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-threshold", type=float, default=0.15); ap.add_argument("--text-threshold", type=float, default=0.20)
    # A generic "person" prompt scores a PRONE figure at ~0.27 while "a person lying on the ground" scores it ~0.76
    # on the same frame. The class here is person_lying_vest, so the generic prompt systematically misses it.
    ap.add_argument("--prompt", default="a person lying on the ground")
    a = ap.parse_args()

    import numpy as np, torch
    from PIL import Image
    import imageio.v3 as iio
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(a.teacher)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.teacher).to("cuda").eval()

    def detect(img):
        inp = proc(images=[img], text=[[a.prompt]], return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inp)
        r = proc.post_process_grounded_object_detection(o, inp["input_ids"], threshold=a.box_threshold,
                                                        text_threshold=a.text_threshold,
                                                        target_sizes=[img.size[::-1]])[0]
        return [([x1, y1, x2-x1, y2-y1], float(s)) for s, (x1, y1, x2, y2)
                in zip(r["scores"].tolist(), r["boxes"].tolist())]

    rows = []
    for s in a.strengths:
        vids = sorted(glob.glob(os.path.join(a.runs, f"s{s}", "*.mp4")))
        for vp in vids:
            clip = Path(vp).stem
            src = os.path.join(a.clips, clip)
            cj = os.path.join(src, "coco.json")
            if not os.path.exists(cj): continue
            coco = json.load(open(cj))
            n2i = {os.path.basename(im["file_name"]): im["id"] for im in coco["images"]}
            dims = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
            anns = {}
            for an in coco["annotations"]: anns.setdefault(an["image_id"], []).append(an)
            vid = iio.imread(vp, plugin="pyav")
            T, H, W = vid.shape[:3]
            kept = tot = 0; hv_src = []; hv_tr = []; ious = []
            for i in range(0, min(T, 93), a.every):
                iid = n2i.get(f"frame_{i:04d}.jpg")
                if iid is None: continue
                ga = anns.get(iid, [])
                if not ga: continue                      # occluded frame, no annotation to preserve
                W0, H0 = dims[iid]; sx, sy = W / W0, H / H0
                gt = [[b*sx if k % 2 == 0 else b*sy for k, b in enumerate(an["bbox"])] for an in ga]
                fr = Image.fromarray(vid[i]).convert("RGB")
                det = detect(fr)
                for g in gt:
                    tot += 1
                    best = max((iou(g, d) for d, _ in det), default=0.0)
                    ious.append(best)
                    if best >= a.iou: kept += 1
                # vest colour, inside the box
                srcf = os.path.join(src, f"frame_{i:04d}.jpg")
                if os.path.exists(srcf):
                    x, y, w, h = ga[0]["bbox"]
                    si = np.array(Image.open(srcf).convert("RGB"))[int(y):int(y+h), int(x):int(x+w)]
                    ti = np.array(fr)[int(y*sy):int((y+h)*sy), int(x*sx):int((x+w)*sx)]
                    if si.size and ti.size:
                        hv_src.append(hivis_fraction(si)); hv_tr.append(hivis_fraction(ti))
            rows.append({"strength": s, "clip": clip, "boxes_checked": tot, "boxes_kept": kept,
                         "box_kept_pct": round(100*kept/max(tot,1), 1),
                         "median_best_iou": round(float(np.median(ious)) if ious else 0.0, 3),
                         "hivis_src": round(float(np.mean(hv_src)) if hv_src else 0, 4),
                         "hivis_transfer": round(float(np.mean(hv_tr)) if hv_tr else 0, 4)})
            print(f"  s={s} {clip}: {kept}/{tot} boxes kept, hi-vis {rows[-1]['hivis_src']:.3f} -> {rows[-1]['hivis_transfer']:.3f}", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n== per strength ==")
    for s in a.strengths:
        r = [x for x in rows if x["strength"] == s]
        if not r: continue
        k = sum(x["boxes_kept"] for x in r); t = sum(x["boxes_checked"] for x in r)
        import statistics as st
        print(f"  s={s}: boxes kept {k}/{t} = {100*k/max(t,1):.1f}%  |  median best IoU "
              f"{st.median(x['median_best_iou'] for x in r):.3f}  |  hi-vis {st.mean(x['hivis_src'] for x in r):.3f} "
              f"-> {st.mean(x['hivis_transfer'] for x in r):.3f}")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
