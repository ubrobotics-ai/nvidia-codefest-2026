#!/usr/bin/env python3
"""Tasks A/B: Cosmos Transfer 2.5 (seg control) over rendered clips, at one or more control strengths.

  python scripts/cosmos_transfer_batch.py --clips <dir-of-clip-dirs> --out <dir> \
      [--strengths 0.4 0.7 1.0] [--frames 93] [--height 704] [--steps 36] [--limit N] [--seed 0]

Each clip dir must hold `frames/*.png|jpg` and `masks/*.png` (segmentation control), or a flat sequence plus
`--masks-from <dir>`. Writes <out>/s<strength>/<clip>.mp4 and B_timing.csv.

PROMPT RULE (decided 2026-09-09, do not undo): the text prompt names the environment and lighting ONLY -- never the
rover, the camera platform, "robot", "tracked" or "vehicle". In the earlier image-to-video run those words put the
vehicle in frame and in some clips rendered it as a wheelchair. Geometry comes from the segmentation masks, not the text.
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
from pathlib import Path

NEGATIVE = ("The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
            "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
            "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
            "Overall, the video is of poor quality.")

BANNED = ("robot", "rover", "tracked", "vehicle", "camera platform", "drone", "ugv", "wheelchair")

def check_prompt(p):
    low = p.lower()
    hit = [w for w in BANNED if w in low]
    if hit:
        sys.exit(f"prompt names the camera platform {hit} -- forbidden, see PROMPT RULE in this file's docstring:\n  {p}")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.7])
    ap.add_argument("--prompt", default="warehouse interior, low camera height, overcast daylight, dust in the air")
    ap.add_argument("--model", default="nvidia/Cosmos-Transfer2.5-2B")
    ap.add_argument("--control", default="seg", choices=["seg", "depth", "edge", "blur"])
    ap.add_argument("--frames", type=int, default=93); ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--height", type=int, default=704); ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--guidance", type=float, default=3.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    check_prompt(a.prompt)

    import torch
    from diffusers import Cosmos2_5_TransferPipeline, AutoModel
    from diffusers.utils import export_to_video
    from PIL import Image

    clips = sorted(d for d in glob.glob(os.path.join(a.clips, "*")) if os.path.isdir(d))
    if a.limit: clips = clips[: a.limit]
    if not clips: sys.exit(f"no clip directories in {a.clips}")
    print(f"{len(clips)} clips x {len(a.strengths)} strengths", flush=True)

    t0 = time.time()
    controlnet = AutoModel.from_pretrained(a.model, revision=f"diffusers/controlnet/general/{a.control}",
                                           torch_dtype=torch.bfloat16)
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(a.model, controlnet=controlnet,
                                                      revision="diffusers/general", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    load_s = time.time() - t0
    print(f"pipeline loaded in {load_s:.0f}s", flush=True)

    # Replicator's instance-segmentation annotator paints the raw instance id (1, 2, ...). Written straight to an
    # 8-bit PNG that is pixel value 1 out of 255 -- visually black, and the ControlNet sees no constraint at all.
    # So: detect low-valued id masks and map each distinct id to a saturated colour before conditioning.
    PALETTE = [(220, 20, 60), (0, 130, 200), (60, 220, 60), (255, 200, 0), (200, 60, 255), (0, 220, 220),
               (255, 130, 0), (150, 150, 150)]

    def colourise_ids(im):
        arr = np.array(im.convert("L") if im.mode not in ("L", "I", "I;16", "P") else im)
        ids = [int(v) for v in np.unique(arr) if v != 0]
        if not ids: return Image.new("RGB", im.size, (0, 0, 0)), 0     # empty mask = occluded frame, no constraint
        out = np.zeros(arr.shape + (3,), dtype=np.uint8)
        for k, v in enumerate(sorted(ids)):
            out[arr == v] = PALETTE[k % len(PALETTE)]
        return Image.fromarray(out), len(ids)

    def load_seq(d, n, is_mask=False):
        # Two layouts: <clip>/{frames,masks}/*, or the v2 flat layout <clip>/frame_NNNN.jpg + mask_NNNN.png.
        # Globbing both extensions in a flat dir would interleave frames with masks, so match by prefix first.
        pat = "mask_*.png" if is_mask else "frame_*.jpg"
        fs = sorted(glob.glob(os.path.join(d, pat)))
        if not fs:
            fs = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
        if not fs: return None, 0
        fs = (fs * ((n // len(fs)) + 1))[:n]
        imgs, empties, maxid = [], 0, 0
        for f in fs:
            im = Image.open(f)
            if is_mask:
                arr = np.array(im.convert("L"))
                hi = int(arr.max())
                if hi <= 16:                       # raw instance ids, not a colourised mask
                    im, nid = colourise_ids(im)
                    maxid = max(maxid, nid)
                    if nid == 0: empties += 1
                else:
                    im = im.convert("RGB")
                    if np.array(im).max() == 0: empties += 1
            else:
                im = im.convert("RGB")
            imgs.append(im.resize((a.width, a.height), Image.NEAREST if is_mask else Image.BICUBIC))
        return imgs, empties

    rows = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in a.strengths:
        sdir = out / f"s{s}"; sdir.mkdir(exist_ok=True)
        for i, c in enumerate(clips, 1):
            name = Path(c).name
            mdir = os.path.join(c, "masks") if os.path.isdir(os.path.join(c, "masks")) else c
            fdir = os.path.join(c, "frames") if os.path.isdir(os.path.join(c, "frames")) else c
            controls, n_empty = load_seq(mdir, a.frames, is_mask=True)
            if controls is None:
                print(f"[warn] {name}: no control frames in {mdir}, skipping", flush=True); continue
            torch.cuda.reset_peak_memory_stats(); t = time.time()
            res = pipe(controls=controls, controls_conditioning_scale=s, prompt=a.prompt, negative_prompt=NEGATIVE,
                       height=a.height, width=a.width, num_frames=a.frames, num_inference_steps=a.steps,
                       guidance_scale=a.guidance,
                       generator=torch.Generator("cuda").manual_seed(a.seed))
            dt = time.time() - t
            vid = res.frames[0] if hasattr(res, "frames") else res.videos[0]
            export_to_video(vid, str(sdir / f"{name}.mp4"), fps=a.fps)
            peak = torch.cuda.max_memory_allocated() / 1e9
            rows.append([f"{s}", name, f"{dt:.1f}", f"{peak:.1f}", a.frames, a.steps, f"{a.width}x{a.height}", n_empty])
            print(f"[s={s}] [{i}/{len(clips)}] {name}: {dt:.1f}s, peak {peak:.1f} GB, empty-mask frames {n_empty}/{a.frames}", flush=True)

    if rows:
        with open(out / "B_timing.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["strength", "clip", "seconds", "peak_gb", "frames", "steps", "resolution", "empty_mask_frames"]); w.writerows(rows)
            mean = sum(float(r[2]) for r in rows) / len(rows)
            w.writerow(["mean_s_per_clip", "", f"{mean:.1f}", "", "", "", ""])
            w.writerow(["pipeline_load_s", "", f"{load_s:.0f}", "", "", "", ""])
            w.writerow(["prompt", a.prompt, "", "", "", "", ""])
        print(f"\nmean_s_per_clip = {mean:.1f} over {len(rows)} clip-runs", flush=True)

if __name__ == "__main__":
    main()
