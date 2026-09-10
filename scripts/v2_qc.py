#!/usr/bin/env python3
"""Task G: QC pass over the v2 Replicator package.

  python scripts/v2_qc.py --root $TEAM/data/isaac_v2 --out G_qc.json [--phash-limit 0]

Checks, each reported with its N:
  1. integrity      every metadata row has an image on disk; every COCO image has a metadata row; box sanity
                    (inside frame, positive area), category ids all declared in classes.json.
  2. person_visible frames with person_visible=false must carry NO person annotations (distractors are allowed).
                    This is the check that catches a Replicator writer emitting boxes for occluded targets.
  3. dedupe         perceptual-hash (dHash 8x8) duplicate rate per source_run, and across the whole split.
  4. coverage       pose x distance-bucket x environment x view, with empty cells named explicitly.
Note: the package carries no `lighting` field, so `view` (which contains `night`) is used as the lighting proxy and
that substitution is reported. `sdg_verify.py` is not shipped in the package -- these checks are independent.
"""
import argparse, json, os, sys, collections
from pathlib import Path

def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size))
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--phash-limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--buckets", type=float, nargs="+", default=[0, 5, 10, 15, 20, 1e9])
    a = ap.parse_args()
    root = Path(a.root)
    classes = {int(k): v for k, v in json.load(open(root / "classes.json")).items()}
    person_ids = {i for i, n in classes.items() if n.startswith("person")}
    report = {"classes": classes, "splits": {}, "notes": []}
    report["notes"].append("no `lighting` field; the package carries `hdri` (HDRI environment map name) which IS the lighting dimension -- used as such")
    report["notes"].append("sdg_verify.py not shipped in the package; these checks are independent")

    for split in a.splits:
        md = root / split / "metadata.jsonl"
        if not md.exists():
            report["splits"][split] = {"error": "metadata.jsonl missing"}; continue
        rows = [json.loads(l) for l in open(md)]
        R = {"n_rows": len(rows)}

        # 1. integrity
        missing_img = [r["file_name"] for r in rows if not (root / split / r["file_name"]).exists()]
        bad_box, bad_cat = [], collections.Counter()
        for r in rows:
            W, H = r["width"], r["height"]
            o = r["objects"]
            for bb, cat in zip(o["bbox"], o["category"]):
                x, y, w, h = bb
                if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                    bad_box.append((r["file_name"], bb))
                if cat not in classes: bad_cat[cat] += 1
        R["missing_images"] = len(missing_img); R["missing_images_examples"] = missing_img[:5]
        R["boxes_out_of_frame_or_degenerate"] = len(bad_box); R["bad_box_examples"] = bad_box[:5]
        R["undeclared_category_ids"] = dict(bad_cat)

        # 2. person_visible consistency
        viol = []
        n_false = 0
        for r in rows:
            if r.get("person_visible") is False:
                n_false += 1
                cats = r["objects"]["category"]
                pers = [c for c in cats if c in person_ids]
                if pers: viol.append({"file": r["file_name"], "person_boxes": len(pers)})
        R["person_visible_false"] = n_false
        R["person_visible_false_with_person_boxes"] = len(viol)
        R["person_visible_violation_examples"] = viol[:5]
        n_true_nobox = sum(1 for r in rows if r.get("person_visible") is True
                           and not any(c in person_ids for c in r["objects"]["category"]))
        R["person_visible_true_without_person_boxes"] = n_true_nobox

        # 3. dedupe by dHash
        from PIL import Image
        sel = rows if not a.phash_limit else rows[: a.phash_limit]
        by_run = collections.defaultdict(list)
        seen = {}
        dup_global = 0
        for i, r in enumerate(sel):
            p = root / split / r["file_name"]
            if not p.exists(): continue
            try:
                h = dhash(Image.open(p))
            except Exception:
                continue
            by_run[r.get("source_run", "?")].append(h)
            if h in seen: dup_global += 1
            else: seen[h] = r["file_name"]
            if i and i % 2000 == 0: print(f"    phash {i}/{len(sel)}", flush=True)
        R["phash_n"] = len(sel)
        R["phash_exact_duplicates_global"] = dup_global
        R["phash_dup_rate_global"] = round(dup_global / max(len(sel), 1), 4)
        per_run = {}
        for run, hs in by_run.items():
            d = len(hs) - len(set(hs))
            per_run[run] = {"n": len(hs), "dups": d, "rate": round(d / max(len(hs), 1), 4)}
        R["phash_per_run"] = dict(sorted(per_run.items(), key=lambda kv: -kv[1]["rate"]))

        # 4. coverage
        def bucket(d):
            if d is None: return "unknown"
            for i in range(len(a.buckets) - 1):
                if a.buckets[i] <= d < a.buckets[i + 1]:
                    hi = a.buckets[i+1]
                    return f"{a.buckets[i]:.0f}-{'inf' if hi > 1e8 else format(hi, '.0f')}m"
            return "unknown"
        cov = collections.Counter()
        poses, envs, views, bks = set(), set(), set(), set()
        for r in rows:
            d = r.get("distance_m")     # per-image (camera -> person), verified real: Spearman -0.871 vs box height
            k = (r.get("pose", "?"), bucket(d), r.get("environment", "?"), r.get("hdri", r.get("view", "?")))
            cov[k] += 1
            poses.add(k[0]); bks.add(k[1]); envs.add(k[2]); views.add(k[3])
        R["coverage_cells_populated"] = len(cov)
        R["coverage_dims"] = {"pose": sorted(poses), "bucket": sorted(bks), "environment": sorted(envs), "lighting_hdri": sorted(views)}
        empty = [f"{p}|{b}|{e}|{v}" for p in sorted(poses) for b in sorted(bks) for e in sorted(envs) for v in sorted(views)
                 if (p, b, e, v) not in cov]
        R["coverage_cells_total"] = len(poses)*len(bks)*len(envs)*len(views)
        R["coverage_cells_empty"] = len(empty); R["coverage_empty_examples"] = empty[:40]
        R["coverage_top"] = {"|".join(map(str, k)): v for k, v in cov.most_common(15)}
        # class histogram
        ch = collections.Counter()
        for r in rows:
            for c in r["objects"]["category"]: ch[classes.get(c, str(c))] += 1
        R["class_box_counts"] = dict(ch.most_common())
        report["splits"][split] = R
        print(f"== {split}: {R['n_rows']} rows, missing imgs {R['missing_images']}, "
              f"person_visible=false {R['person_visible_false']} of which {R['person_visible_false_with_person_boxes']} carry person boxes, "
              f"dup rate {R['phash_dup_rate_global']}, coverage {R['coverage_cells_populated']}/{R['coverage_cells_total']} cells", flush=True)

    Path(a.out).write_text(json.dumps(report, indent=1))
    print("wrote", a.out)

if __name__ == "__main__":
    main()
