#!/usr/bin/env python3
"""Task A check: are the source COCO boxes still valid on the transferred video?

  python scripts/overlay_boxes.py --video <transferred.mp4> --coco <coco.json> --frames <src-frame-dir> \
      --out A_overlay.jpg [--n 5]

Draws each source frame's boxes onto the corresponding transferred frame. Transfer is geometry-preserving in
principle (the control signal fixes the layout), so if the boxes still land on the objects, the annotations carry
over and the transferred clips are usable as labelled training data without re-annotation.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--coco", required=True)
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    import imageio.v3 as iio
    from PIL import Image, ImageDraw

    vid = iio.imread(a.video, plugin="pyav")           # (T,H,W,3)
    T, H, W = vid.shape[:3]
    srcs = sorted(glob.glob(os.path.join(a.frames, "*.png")) + glob.glob(os.path.join(a.frames, "*.jpg")))
    coco = json.load(open(a.coco))
    by_name = {os.path.basename(im["file_name"]): im for im in coco["images"]}
    anns = {}
    for an in coco["annotations"]: anns.setdefault(an["image_id"], []).append(an)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    COL = {0: (255, 60, 60), 1: (60, 160, 255)}

    idxs = np.linspace(0, min(T, len(srcs)) - 1, a.n).astype(int)
    tiles = []
    for i in idxs:
        frame = Image.fromarray(vid[i]).convert("RGB")
        src = os.path.basename(srcs[i])
        # the smoke sequence renamed frames 0000.png..; map back through the source list order
        im_rec = by_name.get(src)
        if im_rec is None:
            # fall back: positional match into the coco image list
            im_rec = coco["images"][i] if i < len(coco["images"]) else None
        d = ImageDraw.Draw(frame)
        drawn = 0
        if im_rec:
            sx, sy = W / im_rec["width"], H / im_rec["height"]
            for an in anns.get(im_rec["id"], []):
                x, y, w, h = an["bbox"]
                d.rectangle([x*sx, y*sy, (x+w)*sx, (y+h)*sy], outline=COL.get(an["category_id"], (0,255,0)), width=4)
                d.text((x*sx+4, y*sy+2), cats.get(an["category_id"], "?")[:14], fill=COL.get(an["category_id"], (0,255,0)))
                drawn += 1
        d.text((6, 6), f"frame {i}  boxes {drawn}", fill=(255, 255, 0))
        tiles.append(frame)
    tw = 620
    tiles = [t.resize((tw, int(t.height * tw / t.width))) for t in tiles]
    cols = min(len(tiles), 5); rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * tiles[0].height), (18, 18, 18))
    for k, t in enumerate(tiles): sheet.paste(t, ((k % cols) * tw, (k // cols) * t.height))
    sheet.save(a.out, quality=90)
    print(f"wrote {a.out} from {a.video} ({T} frames) — {len(tiles)} panels")

if __name__ == "__main__":
    main()
