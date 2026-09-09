#!/usr/bin/env python3
"""Assemble REPORT.md for the overnight run from B_summary.csv, timing.csv and the run dirs."""
import argparse, csv, json, glob, os, statistics as st
from pathlib import Path

def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--summary", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.summary)))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
    v0 = next((r for r in rows if r["run"] == "v0"), None)
    arms = {}
    for r in rows:
        if not r["run"].startswith("B_"): continue
        arms.setdefault(r["arm"], []).append(r)
    L = []
    L.append("### v0 baseline (YOLO26n, imgsz 640, 2 classes, seed 0, 60 epochs)\n")
    if v0:
        L.append(f"| metric | value |\n|---|---|")
        L.append(f"| mAP@0.5 (person_lying_vest) | **{f(v0,'mAP50'):.4f}** |")
        L.append(f"| mAP@0.5:0.95 | **{f(v0,'mAP50-95'):.4f}** |")
        for k, lab in (("recall_orbit","recall orbit"), ("recall_roverview","recall roverview 4–8.5 m"),
                       ("recall_roverview_far","recall roverview_far 10–17.5 m")):
            L.append(f"| {lab} | **{f(v0,k):.4f}** (n={v0.get('n_'+k.split('recall_')[1],'?')}) |")
        L.append(f"| false alarms on {v0['n_empty_real']} empty real frames (conf 0.25) | **{v0['FA_empty_real']}** |")
    L.append("\n### Sweep — median over 3 seeds, class `person_lying_vest` only\n")
    hdr = "| arm | mAP@0.5 | mAP@0.5:0.95 | R orbit | R roverview | R roverview_far | FA empty |"
    L.append(hdr); L.append("|" + "---|" * 7)
    order = ["640_2cls", "640_1cls", "1280_2cls", "1280_1cls"]
    for arm in order:
        rs = arms.get(arm, [])
        if not rs: continue
        L.append(f"| `{arm}` | {med(f(r,'mAP50') for r in rs):.4f} | {med(f(r,'mAP50-95') for r in rs):.4f} | "
                 f"{med(f(r,'recall_orbit') for r in rs):.4f} | {med(f(r,'recall_roverview') for r in rs):.4f} | "
                 f"{med(f(r,'recall_roverview_far') for r in rs):.4f} | {med(f(r,'FA_empty_real') for r in rs):.0f} |")
    def d(a1, a2, key):
        if a1 not in arms or a2 not in arms: return None
        return med(f(r,key) for r in arms[a2]) - med(f(r,key) for r in arms[a1])
    L.append("\n**Does 1280 move roverview_far recall?**")
    for cls in ("2cls", "1cls"):
        dd = d(f"640_{cls}", f"1280_{cls}", "recall_roverview_far")
        dn = d(f"640_{cls}", f"1280_{cls}", "recall_roverview")
        if dd is not None:
            L.append(f"- `{cls}`: roverview_far recall {med(f(r,'recall_roverview_far') for r in arms[f'640_{cls}']):.4f} → "
                     f"{med(f(r,'recall_roverview_far') for r in arms[f'1280_{cls}']):.4f} (**{dd:+.4f}**); "
                     f"mid-range roverview {dn:+.4f}.")
    print("\n".join(L))
    Path(a.out).write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
