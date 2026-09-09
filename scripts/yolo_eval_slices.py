#!/usr/bin/env python3
"""Task B.5: score every run in --runs on the validation split sliced by camera view (distance bucket) and environment,
count false alarms on the empty real frames, and write B_summary.csv. Also exports ONNX for --export runs.

  python scripts/yolo_eval_slices.py --data-root $TEAM/data/rescue_yolo --runs $WORK/runs \
      --empty-frames data/real_frames --out $WORK/logs/night-20260909/B_summary.csv [--export v0,B_1280_2cls_s0]

Metrics are for class 0 (person_lying_vest) only, so 1cls and 2cls arms are comparable. Recall is at the conf/IoU
Ultralytics reports for the mAP curve (conf 0.001, IoU 0.7 NMS); mAP50 / mAP50-95 are standard.
"""
import argparse, csv, json, glob, os, time
from pathlib import Path

SLICES = [("view", "orbit"), ("view", "roverview"), ("view", "roverview_far"),
          ("env", "hospital"), ("env", "office"), ("env", "warehouse")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--empty-frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--export", default="v0", help="comma-separated run names to export to ONNX")
    ap.add_argument("--export-best", action="store_true", help="also export the best B_ run by mAP50-95")
    ap.add_argument("--conf-fa", type=float, default=0.25)
    a = ap.parse_args()
    from ultralytics import YOLO
    root, runs = Path(a.data_root), Path(a.runs)
    empty = sorted(f for f in glob.glob(os.path.join(a.empty_frames, "*")) if Path(f).stem.endswith("_empty"))
    rows = []
    for d in sorted(runs.iterdir()):
        w = d / "weights" / "best.pt"
        if not w.exists(): continue
        meta = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        cls, imgsz = meta.get("classes", "2cls"), meta.get("imgsz", 640)
        m = YOLO(str(w))
        row = {"run": d.name, "arm": f"{imgsz}_{cls}", "seed": meta.get("seed"), "imgsz": imgsz, "classes": cls,
               "epochs": meta.get("epochs"), "n_val": 440}
        r = m.val(data=str(root / f"rescue_{cls}.yaml"), imgsz=imgsz, batch=32, plots=False, verbose=False, device=0)
        row["mAP50"] = round(float(r.box.ap50[0]), 4); row["mAP50-95"] = round(float(r.box.ap[0]), 4)
        row["recall_all"] = round(float(r.box.r[0]), 4); row["precision_all"] = round(float(r.box.p[0]), 4)
        for kind, key in SLICES:
            y = root / f"rescue_{cls}_val_{kind}_{key}.yaml"
            n = sum(1 for _ in open(root / cls / f"val_{kind}_{key}.txt"))
            rs = m.val(data=str(y), imgsz=imgsz, batch=32, plots=False, verbose=False, device=0)
            row[f"recall_{key}"] = round(float(rs.box.r[0]), 4); row[f"mAP50_{key}"] = round(float(rs.box.ap50[0]), 4)
            row[f"n_{key}"] = n
        fa = 0
        if empty:
            for res in m.predict(empty, imgsz=imgsz, conf=a.conf_fa, verbose=False, device=0):
                fa += int((res.boxes.cls == 0).sum()) if res.boxes is not None else 0
        row["FA_empty_real"] = fa; row["n_empty_real"] = len(empty)
        rows.append(row); print(row, flush=True)
    to_export = set(x for x in a.export.split(",") if x)
    if a.export_best:
        best = max((r for r in rows if r["run"].startswith("B_")), key=lambda r: r["mAP50-95"], default=None)
        if best: to_export.add(best["run"]); print("[best arm by mAP50-95]", best["run"], best["mAP50-95"])
    for name in sorted(to_export):
        w = runs / name / "weights" / "best.pt"
        if not w.exists(): print(f"[onnx] {name}: no weights"); continue
        imgsz = next((r["imgsz"] for r in rows if r["run"] == name), 640)
        t = time.time(); p = YOLO(str(w)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
        print(f"[onnx] {name} imgsz={imgsz} -> {p} in {time.time()-t:.1f}s, {os.path.getsize(p)/1e6:.1f} MB", flush=True)
        for r in rows:
            if r["run"] == name: r["onnx"] = p
    keys = sorted(set(k for r in rows for k in r), key=lambda k: (k != "run", k)) if rows else ["run"]
    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader(); wtr.writerows(rows)
    print("wrote", a.out, len(rows), "runs")

if __name__ == "__main__":
    main()
