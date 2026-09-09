#!/usr/bin/env python3
"""Task B: v0 baseline + imgsz/distractor sweep for the rescue-target detector, all on ONE GPU inside one process.

  python scripts/yolo_sweep.py --data-root $TEAM/data/rescue_yolo --runs $WORK/runs [--epochs 60] [--seeds 0 1 2]
        [--parallel 4] [--only v0] [--model yolo26n.pt]

Runs (each in its own subprocess so failures are isolated; `--parallel` of them at a time share the GPU):
  v0                       2cls, imgsz 640, seed 0          (frozen baseline)
  B_<imgsz>_<cls>_s<seed>  imgsz in {640,1280} x cls in {2cls,1cls} x seeds       (4 arms x 3 seeds = 12)
Each run dir gets args.yaml / results.csv / weights/best.pt from Ultralytics plus run.json (cmd, start/end, exit).
Recipe when the team's is unknown: epochs 60, batch 64, default augmentation, patience off, seed as given.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--model", default="yolo26n.pt"); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--parallel", type=int, default=4); ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    root, runs = Path(a.data_root), Path(a.runs); runs.mkdir(parents=True, exist_ok=True)

    jobs = [("v0", 640, "2cls", 0)]
    for imgsz in (640, 1280):
        for cls in ("2cls", "1cls"):
            for s in a.seeds:
                jobs.append((f"B_{imgsz}_{cls}_s{s}", imgsz, cls, s))
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only.split(",")]
    # biggest first so the 1280 runs do not end up alone at the tail
    jobs.sort(key=lambda j: (-j[1], j[0]))

    def launch(name, imgsz, cls, seed):
        d = runs / name
        # Ultralytics writes best.pt/results.csv from epoch 1, so only our own run.json (exit 0) proves completion.
        if d.exists():
            try:
                done = json.loads((d / "run.json").read_text()).get("exit") == 0
            except Exception:
                done = False
            if done:
                print(f"[skip] {name}: already trained"); return None
            aside = runs / f"_partial_{name}_{time.strftime('%Y%m%d-%H%M%S')}"
            d.rename(aside); print(f"[partial] {name}: moved incomplete run to {aside.name}")
        cmd = [sys.executable, "-c", f"""
from ultralytics import YOLO
m = YOLO({a.model!r})
m.train(data={str(root / f'rescue_{cls}.yaml')!r}, imgsz={imgsz}, epochs={a.epochs}, batch={a.batch}, seed={seed},
        deterministic=False, workers={a.workers}, project={str(runs)!r}, name={name!r}, exist_ok=True,
        patience=0, plots=False, verbose=False, device=0)
"""]
        d.mkdir(parents=True, exist_ok=True)
        meta = {"name": name, "imgsz": imgsz, "classes": cls, "seed": seed, "epochs": a.epochs, "batch": a.batch,
                "model": a.model, "cmd": cmd[-1], "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (d / "run.json").write_text(json.dumps(meta, indent=2))
        log = open(d / "train.log", "a")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"[start] {name} pid {p.pid} imgsz={imgsz} {cls} seed={seed}", flush=True)
        return (p, d, meta, log)

    pending = list(jobs); active = []
    while pending or active:
        while pending and len(active) < a.parallel:
            r = launch(*pending.pop(0))
            if r: active.append(r)
        time.sleep(20)
        for r in list(active):
            p, d, meta, log = r
            if p.poll() is None: continue
            meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S"); meta["exit"] = p.returncode
            (d / "run.json").write_text(json.dumps(meta, indent=2)); log.close()
            print(f"[done] {meta['name']} exit {p.returncode} ({meta['start']} -> {meta['end']})", flush=True)
            active.remove(r)
    print("sweep finished")

if __name__ == "__main__":
    main()
