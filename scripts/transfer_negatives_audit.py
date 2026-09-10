#!/usr/bin/env python3
"""Can the transferred clips ship as hard negatives?

  python scripts/transfer_negatives_audit.py --clips <src-root> --runs <transfer-out> \
      --out T_negatives.csv [--strengths 0.4 0.7 1.0] [--every 6]

The transferred clips carry their source COCO annotation, and that annotation is wrong most of the time (measured:
0 / 13.6 / 35 % box survival at strength 0.4 / 0.7 / 1.0). So they cannot ship as positives. They may still be worth
keeping as NEGATIVES -- photoreal industrial interiors with no person -- which is the distribution where v0 currently
false-alarms on 3.2 % of real frames.

That only holds if the person is genuinely ABSENT, not merely undetected in the labelled box. Each sampled frame is
therefore sorted into one of three states, not two:

  kept       a person is detected and overlaps the source box (IoU >= 0.5) -> the label survived, usable as a positive
  displaced  a person is detected SOMEWHERE ELSE in the frame              -> NOT a clean negative; the generator moved
                                                                              the subject, and shipping it as a negative
                                                                              would teach the detector to ignore a real
                                                                              person
  absent     no person anywhere in the frame                               -> a clean negative, safe to ship once the
                                                                              COCO annotation is stripped

Two prompts are used because they disagree: a generic "person" scores a prone figure ~0.27 while "a person lying on
the ground" scores it ~0.76. A frame counts as `absent` only if BOTH find nothing, which is the conservative call --
we would rather discard a usable negative than ship a frame with a person in it labelled as empty.
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--strengths", nargs="+", default=["0.4", "0.7", "1.0"])
    ap.add_argument("--every", type=int, default=6); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--box-threshold", type=float, default=0.15)
    ap.add_argument("--prompts", nargs="+", default=["person", "a person lying on the ground"])
    a = ap.parse_args()

    import torch
    from PIL import Image
    import imageio.v3 as iio
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda").eval()

    def detect(img, prompt):
        inp = proc(images=[img], text=[[prompt]], return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inp)
        r = proc.post_process_grounded_object_detection(o, inp["input_ids"], threshold=a.box_threshold,
                                                        text_threshold=0.20, target_sizes=[img.size[::-1]])[0]
        return [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in r["boxes"].tolist()]

    rows = []
    for s in a.strengths:
        d = os.path.join(a.runs, f"s{s}")
        if not os.path.isdir(d): continue
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
            kept = disp = absent = nolabel = 0
            for i in range(0, min(T, 93), a.every):
                iid = n2i.get(f"frame_{i:04d}.jpg")
                fr = Image.fromarray(vid[i]).convert("RGB")
                det = []
                for p in a.prompts: det += detect(fr, p)
                ga = anns.get(iid, []) if iid else []
                if not ga:
                    # source frame was occluded: no annotation to preserve. Still useful as a negative if empty.
                    nolabel += 1
                    if not det: absent += 1
                    else: disp += 1
                    continue
                W0, H0 = dims[iid]; sx, sy = W/W0, H/H0
                gt = [[an["bbox"][0]*sx, an["bbox"][1]*sy, an["bbox"][2]*sx, an["bbox"][3]*sy] for an in ga]
                if not det: absent += 1
                elif any(iou(g, dd) >= a.iou for g in gt for dd in det): kept += 1
                else: disp += 1
            n = kept + disp + absent
            rows.append({"strength": s, "clip": clip, "frames": n, "kept": kept, "displaced": disp,
                         "absent": absent, "src_unlabelled": nolabel,
                         "absent_pct": round(100*absent/max(n, 1), 1)})
            print(f"  s={s} {clip}: kept {kept} displaced {disp} absent {absent} of {n}", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n== can these ship as negatives? ==")
    for s in a.strengths:
        r = [x for x in rows if x["strength"] == s]
        if not r: continue
        k = sum(x["kept"] for x in r); d = sum(x["displaced"] for x in r); ab = sum(x["absent"] for x in r)
        n = k + d + ab
        print(f"  s={s}: {n} frames -> kept {k} ({100*k/n:.1f}%)  displaced {d} ({100*d/n:.1f}%)  "
              f"ABSENT {ab} ({100*ab/n:.1f}%)")
        print(f"        clean negatives available: {ab}  |  frames that must NOT ship as negatives: {d}")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
