"""Fixed operating point: a frame fires if any box of class CLS has conf >= THR (and w/h >= RULE).
Reports recall on positives, FA on hard negatives and FA on empty frames (hospital val split)."""
import json, os, sys, collections
exec(open(os.path.join(os.path.dirname(__file__), "harness_replica.py")).read().split("import sys")[0])  # builds `kind`
path, cls, thr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
best = {}
for l in open(path):
    r = json.loads(l)
    best[r["file_name"]] = [(b["conf"], b["bbox"][2] / max(b["bbox"][3], 1e-6)) for b in r["boxes"] if b["cls"] == cls]
for rule in (1.3, 0.0):
    fire = {f: any(c >= thr and a >= rule for c, a in v) for f, v in best.items()}
    out = {k: sum(fire.get(f, False) for f, kk in kind.items() if kk == k) / sum(1 for kk in kind.values() if kk == k) for k in ("pos", "hard", "empty")}
    print(f"conf>={thr} rule w/h>={rule}: recall {out['pos']:.3f}  FA hard {out['hard']:.3f}  FA empty {out['empty']:.3f}")
