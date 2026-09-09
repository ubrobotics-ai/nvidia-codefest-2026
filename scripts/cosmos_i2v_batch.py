#!/usr/bin/env python3
"""Generate N image->video clips with Cosmos 3 from real UGV frames and time them.

Usage:
  python scripts/cosmos_i2v_batch.py --frames data/real_frames --out data/synthetic --n 20 \
      [--model nvidia/Cosmos3-Nano] [--frames-per-clip 189] [--steps 35] [--nemotron-url http://localhost:8000/v1]

Writes <out>/<frame>_<variant>.mp4, <out>/<frame>_<variant>.json (prompt + ground-truth tags) and <out>/timing.csv.
"""
import argparse, csv, json, os, sys, time, glob, random
from pathlib import Path

# Condition variants. Kept environment-neutral so they read correctly for indoor (building) and outdoor frames alike.
VARIANTS = [
    dict(tag="day_clear",   scene="daylight, clear visibility, calm"),
    dict(tag="dusk_smoke",  scene="low light, thin grey smoke drifting in from the right"),
    dict(tag="night_thermal", scene="dark, scene lit only by the robot's headlights, dust in the air"),
    dict(tag="dense_smoke", scene="dense white smoke reducing visibility to about ten metres, embers"),
]

def has_person(frame_path):
    """Ground truth comes from the source frame's name: <idx>_<date>_<res>_person.jpg or ..._empty.jpg."""
    return not Path(frame_path).stem.endswith("_empty")

def upsample_prompt(base_desc, variant, url, person=True):
    """Ask the Nemotron NIM to write the structured JSON prompt Cosmos 3 expects. Falls back to a minimal dict."""
    fallback = {"scene": f"{base_desc}. {variant['scene']}",
                "camera": "fixed forward camera on a small tracked ground robot, roughly thirty centimetres above the floor, slight vibration",
                "motion": "robot drives slowly forward for eight seconds, keeping the layout of the source image",
                "subjects": ("one or more people standing or walking a few metres ahead, partially occluded by furniture"
                             if person else "no people anywhere in view, only floor, furniture and equipment"),
                "style": "photorealistic, handheld-quality sensor footage, no text, no UI"}
    if not url:
        return fallback
    try:
        import urllib.request
        models = json.load(urllib.request.urlopen(f"{url}/models"))["data"][0]["id"]
        body = {"model": models, "temperature": 0.3, "max_tokens": 800,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": "You write structured JSON prompts for a physical-AI video generator. Output only a JSON object with keys scene, camera, motion, subjects, style; every value a single plain string of at most 60 words."},
                             {"role": "user", "content": json.dumps(fallback)}]}
        req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
        s, e = txt.find("{"), txt.rfind("}")
        return json.loads(txt[s:e+1])
    except Exception as ex:
        print(f"[warn] prompt upsampling failed ({ex}); using fallback prompt", file=sys.stderr)
        return fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--model", default="nvidia/Cosmos3-Nano")
    ap.add_argument("--frames-per-clip", type=int, default=189); ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--height", type=int, default=720); ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--nemotron-url", default=os.environ.get("NEMOTRON_URL", ""))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.utils import export_to_video, load_image

    frames = sorted(glob.glob(os.path.join(a.frames, "*.jpg")) + glob.glob(os.path.join(a.frames, "*.png")))
    if not frames:
        sys.exit(f"no frames in {a.frames}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    try:
        pipe.enable_safety_checker()
    except Exception:
        pass
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s; GPU mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    negative = {"avoid": "text, watermark, UI overlays, cartoon, blurry, distorted people, extra limbs"}
    rows = []; blocked = []; upsample_fail = 0
    def generate(prompt, image, seed):
        return pipe(prompt=json.dumps(prompt), negative_prompt=json.dumps(negative), image=image,
                    num_frames=a.frames_per_clip, height=a.height, width=a.width,
                    num_inference_steps=a.steps, guidance_scale=6.0, fps=24.0,
                    generator=torch.Generator("cuda").manual_seed(seed))
    jobs = [(f, VARIANTS[i % len(VARIANTS)]) for i, f in enumerate((frames * 100)[: a.n])]
    for i, (f, v) in enumerate(jobs, 1):
        stem = f"{Path(f).stem}_{v['tag']}"
        person = has_person(f)
        base = "Forward camera view from a small tracked search-and-rescue robot driving through an indoor office and lab building"
        prompt = upsample_prompt(base, v, a.nemotron_url, person)
        fallback = upsample_prompt(base, v, "", person)
        if a.nemotron_url and prompt == fallback:
            upsample_fail += 1
        image = load_image(f)
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        try:
            res = generate(prompt, image, a.seed + i)
        except ValueError as ex:  # cosmos_guardrail rejects the prompt text with a ValueError
            if "Guardrail" not in str(ex) or prompt == fallback:
                print(f"[warn] {stem}: guardrail blocked prompt, skipping clip: {str(ex)[:160]}", file=sys.stderr)
                blocked.append(stem); continue
            print(f"[warn] {stem}: guardrail blocked upsampled prompt, retrying with fallback", file=sys.stderr)
            prompt = fallback; blocked.append(stem + " (upsampled)")
            try:
                res = generate(prompt, image, a.seed + i)
            except ValueError as ex2:
                print(f"[warn] {stem}: guardrail blocked fallback too, skipping clip: {str(ex2)[:160]}", file=sys.stderr)
                continue
        dt = time.time() - t
        export_to_video(res.video, str(out / f"{stem}.mp4"), fps=24, macro_block_size=1)
        (out / f"{stem}.json").write_text(json.dumps({"source_frame": f, "variant": v["tag"], "prompt": prompt,
                                                       "ground_truth": {"person_present": person,
                                                                        "person_state": "standing_or_walking" if person else None,
                                                                        "visibility": v["tag"]}}, indent=2))
        peak = torch.cuda.max_memory_allocated() / 1e9
        rows.append([i, stem, f"{dt:.1f}", f"{peak:.1f}"])
        print(f"[{i}/{a.n}] {stem}: {dt:.1f}s, peak {peak:.1f} GB")

    if not rows:
        sys.exit(f"no clips generated ({len(blocked)} blocked by guardrail)")
    mean = sum(float(r[2]) for r in rows) / len(rows)
    with open(out / "timing.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["idx", "clip", "seconds", "peak_gb"]); w.writerows(rows)
        w.writerow(["mean_s_per_clip", "", f"{mean:.1f}", ""]); w.writerow(["model_load_s", "", f"{load_s:.1f}", ""])
        w.writerow(["n_clips", "", len(rows), ""]); w.writerow(["guardrail_blocked", ";".join(blocked), len(blocked), ""])
        w.writerow(["upsample_failures", "", upsample_fail, ""])
        w.writerow(["settings", f"frames={a.frames_per_clip} steps={a.steps} {a.width}x{a.height} model={a.model}", "", ""])
    print(f"\nmean_s_per_clip = {mean:.1f} (n={len(rows)}, blocked={len(blocked)}, upsample_failures={upsample_fail})"
          f"  ->  600 clips ≈ {600*mean/3600:.1f} GPU-hours on this setup")

if __name__ == "__main__":
    main()
