#!/usr/bin/env python3
"""Select the frozen evaluation (L1) and quantisation-calibration frame slices, reproducibly.

  python scripts/select_frame_slices.py --index <frames_index.csv> --out data/slices

Both slices are frame-id lists only; the images they name are never committed. Re-running with the same index and
seeds reproduces them byte for byte, which is what makes a bake-off number checkable by someone else.

Guards, in the order they are applied:
  max per segment   at most 2 frames from any one rover segment, so a single long segment cannot dominate and
                    inflate apparent agreement between arms.
  dHash distance    every accepted frame is >= MIN_HAMMING (12) bits from every frame already accepted, across the
                    whole slice rather than within a segment. The index is already deduped at Hamming <= 6; 12 is a
                    deliberately stricter bar for an evaluation set, where near-duplicates would silently correlate
                    the errors that a paired test assumes are independent.
  disjointness      the calibration slice is drawn from what L1 did not take, and the result is asserted disjoint.
                    INT4 AWQ fits activation ranges to its calibration data, so any overlap would flatter the
                    quantised model on exactly the frames used to judge it.
"""
import argparse, csv, json, hashlib, os, random, sys
from pathlib import Path

def hamming(a, b):
    return bin(a ^ b).count("1")

def select(rows, n_target, seed, max_per_segment, min_hamming, taken=frozenset(), taken_hashes=()):
    """Breadth first: one frame from as many distinct segments as possible, then a second, and so on.

    A plain shuffle-and-greedy takes two frames from a segment whenever the Hamming guard allows, which spends the
    budget on fewer segments (474 frames over 309 segments here). Filling by rank instead reaches 349 segments for
    the same 474 frames -- more independent scenes for the same evaluation cost, which is the point of the slice.
    """
    rnd = random.Random(seed)
    pool = [r for r in rows if r["file"] not in taken]
    rnd.shuffle(pool)
    by_seg = {}
    for r in pool: by_seg.setdefault(r["segment"], []).append(r)
    order = sorted(by_seg)
    rnd.shuffle(order)
    chosen, hashes = [], list(taken_hashes)
    for rank in range(max_per_segment):
        for seg in order:
            if len(chosen) >= n_target: break
            bucket = by_seg[seg]
            if sum(1 for c in chosen if c["segment"] == seg) > rank: continue
            for r in bucket:
                if r in chosen: continue
                h = int(r["dhash"])
                if any(hamming(h, k) < min_hamming for k in hashes): continue
                chosen.append(r); hashes.append(h); break
        if len(chosen) >= n_target: break
    return chosen, hashes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--l1-n", type=int, default=474); ap.add_argument("--l1-seed", type=int, default=0)
    ap.add_argument("--calib-n", type=int, default=256); ap.add_argument("--calib-seed", type=int, default=17)
    ap.add_argument("--max-per-segment", type=int, default=2); ap.add_argument("--min-hamming", type=int, default=12)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.index)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"index: {len(rows)} frames, {len({r['segment'] for r in rows})} segments")

    l1, l1_hashes = select(rows, a.l1_n, a.l1_seed, a.max_per_segment, a.min_hamming)
    l1_files = {r["file"] for r in l1}
    calib, _ = select(rows, a.calib_n, a.calib_seed, a.max_per_segment, a.min_hamming,
                      taken=l1_files, taken_hashes=l1_hashes)
    calib_files = {r["file"] for r in calib}

    assert not (l1_files & calib_files), "L1 and calibration overlap"
    for name, sel in (("l1", l1), ("calib", calib)):
        segs = {r["segment"] for r in sel}
        counts = {}
        for r in sel: counts[r["segment"]] = counts.get(r["segment"], 0) + 1
        assert max(counts.values()) <= a.max_per_segment
        print(f"{name}: {len(sel)} frames over {len(segs)} segments (max {max(counts.values())}/segment)")

    def write(name, sel, seed):
        p = out / f"{name}_slice_v1.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["file", "segment", "dhash"])
            for r in sorted(sel, key=lambda x: x["file"]): w.writerow([r["file"], r["segment"], r["dhash"]])
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        print(f"  wrote {p}  sha256:{digest}")
        return {"file": p.name, "n": len(sel), "segments": len({r['segment'] for r in sel}),
                "seed": seed, "sha256_16": digest}
    meta = {"index": os.path.basename(a.index), "index_frames": len(rows),
            "index_segments": len({r["segment"] for r in rows}),
            "max_per_segment": a.max_per_segment, "min_hamming": a.min_hamming,
            "l1": write("l1", l1, a.l1_seed), "calib": write("calib", calib, a.calib_seed),
            "disjoint": True}
    (out / "slices_manifest.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / "slices_manifest.json")

if __name__ == "__main__":
    main()
