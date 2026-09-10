#!/usr/bin/env python3
"""Third/fourth arms for the survival comparison: image-conditioned generation from a clip's first frame.

  python scripts/i2v_arm.py --clips <src-root> --out <dir> --model nvidia/Cosmos3-Nano --frames 93 [--limit 13]

Conditions on frame_0000.jpg only — no per-frame control — so the anchor is temporal, not spatial. That is the
contrast with Transfer, whose mask is re-applied every frame. Same prompt rule: environment and lighting only.

--model nvidia/Cosmos3-Super runs the 64B arm. If it OOMs the run aborts for that clip, records the peak memory it
reached, and moves on -- it does not retry, since a retry of an OOM is just a slower OOM.
"""
import argparse, csv, glob, json, os, sys, time
from pathlib import Path

NEG = ("The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
       "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas.")
BANNED = ("robot", "rover", "tracked", "vehicle", "camera platform", "drone", "ugv", "wheelchair")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/Cosmos3-Nano")
    ap.add_argument("--prompt", default="industrial hall interior, low camera height, overcast daylight, dust in the air")
    ap.add_argument("--frames", type=int, default=93); ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--height", type=int, default=704); ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=13); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--static", action="store_true",
                    help="control arm: hold frame 0 for the whole clip (no camera motion in the source)")
    a = ap.parse_args()
    low = a.prompt.lower()
    if any(w in low for w in BANNED): sys.exit(f"prompt names the camera platform: {a.prompt}")

    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.utils import export_to_video, load_image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = sorted(d for d in glob.glob(os.path.join(a.clips, "*")) if os.path.isdir(d))[: a.limit]
    print(f"{len(clips)} clips, model {a.model}", flush=True)

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    print(f"pipeline loaded in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for i, c in enumerate(clips, 1):
        name = Path(c).name
        f0 = os.path.join(c, "frame_0000.jpg")
        if not os.path.exists(f0):
            print(f"[warn] {name}: no frame_0000.jpg", flush=True); continue
        img = load_image(f0)
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        try:
            res = pipe(prompt=a.prompt, negative_prompt=NEG, image=img, num_frames=a.frames,
                       height=a.height, width=a.width, num_inference_steps=a.steps, guidance_scale=6.0, fps=24.0,
                       generator=torch.Generator("cuda").manual_seed(a.seed))
        except torch.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated()/1e9
            print(f"[OOM] {name}: aborted at peak {peak:.1f} GB — not retrying", flush=True)
            rows.append([name, "OOM", f"{peak:.1f}", a.frames, a.steps]); torch.cuda.empty_cache(); continue
        dt = time.time()-t
        vid = getattr(res, "video", None) or (res.frames[0] if hasattr(res, "frames") else res.videos[0])
        if isinstance(vid, list) and vid and isinstance(vid[0], list): vid = vid[0]
        export_to_video(vid, str(out / f"{name}.mp4"), fps=24)
        peak = torch.cuda.max_memory_allocated()/1e9
        rows.append([name, f"{dt:.1f}", f"{peak:.1f}", a.frames, a.steps])
        print(f"[{i}/{len(clips)}] {name}: {dt:.1f}s, peak {peak:.1f} GB", flush=True)

    with open(out / "timing.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "seconds", "peak_gb", "frames", "steps"]); w.writerows(rows)
        w.writerow(["model", a.model, "", "", ""]); w.writerow(["prompt", a.prompt, "", "", ""])
    ok = [r for r in rows if r[1] != "OOM"]
    if ok:
        print(f"\nmean {sum(float(r[1]) for r in ok)/len(ok):.1f} s/clip over {len(ok)} clips "
              f"({len(rows)-len(ok)} OOM)", flush=True)

if __name__ == "__main__":
    main()
