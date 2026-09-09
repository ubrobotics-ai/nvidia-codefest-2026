#!/usr/bin/env python3
"""Contact sheet: one frame (default #90) from every .mp4 in a directory, tiled with the clip name.
  python scripts/contact_sheet.py --clips $TEAM/data/synthetic_i2v --out A_contact.jpg [--frame 90] [--cols 5]
Uses imageio (ffmpeg plugin bundled with imageio-ffmpeg) so it runs inside the container without a system ffmpeg.
"""
import argparse, glob, os
from PIL import Image, ImageDraw
import imageio.v3 as iio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=90); ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-w", type=int, default=480)
    a = ap.parse_args()
    clips = sorted(glob.glob(os.path.join(a.clips, "*.mp4")))
    tiles = []
    for c in clips:
        try:
            fr = iio.imread(c, index=a.frame, plugin="pyav")
        except Exception:
            fr = iio.imread(c, index=-1, plugin="pyav") if False else iio.imread(c, index=0, plugin="pyav")
        im = Image.fromarray(fr); im = im.resize((a.tile_w, int(im.height * a.tile_w / im.width)))
        ImageDraw.Draw(im).text((6, 6), os.path.basename(c)[:60], fill=(255, 255, 0))
        tiles.append(im)
    if not tiles:
        raise SystemExit("no clips")
    tw, th = tiles[0].size; rows = (len(tiles) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * tw, rows * th), (20, 20, 20))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % a.cols) * tw, (i // a.cols) * th))
    sheet.save(a.out, quality=88); print("wrote", a.out, len(tiles), "clips")

if __name__ == "__main__":
    main()
