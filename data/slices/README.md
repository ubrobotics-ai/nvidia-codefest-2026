# Frozen frame slices

Two frame-id lists drawn from the deduped rover blackbox set. **The lists are tracked here; the images they name
are not** — they show identifiable people and stay private (see the privacy note below).

| slice | file | frames | segments | seed | purpose |
|---|---|---|---|---|---|
| **L1** | `l1_slice_v1.csv` | 474 | 363 | 0 | frozen evaluation set — every bake-off number is quoted against this |
| **Calibration** | `calib_slice_v1.csv` | 256 | 171 | 17 | INT4 AWQ activation-range fitting only |

Regenerate both with:

```bash
python scripts/select_frame_slices.py \
  --index $TEAM/data/blackbox_frames/frames_index.csv --out data/slices
```

`slices_manifest.json` carries the SHA-256 of each CSV. If a number cannot be reproduced, check the digest first —
a silently regenerated slice is the likeliest cause.

## Guards

**At most 2 frames per segment.** A rover segment is one continuous run, so its frames share lighting, location and
often the same people. Without a cap, a few long segments would dominate the slice and two models would appear to
agree simply because they were being asked about the same scene repeatedly.

**dHash Hamming ≥ 12 across the whole slice.** The source index is already deduped at Hamming ≤ 6; 12 is a stricter
bar chosen for evaluation, where near-duplicates correlate errors and quietly violate the independence that a paired
test assumes. The guard is applied across the entire slice, not within each segment.

**Breadth-first over segments.** Frames are taken one per segment across all segments before any segment gives a
second. A plain shuffle-and-greedy selection reaches the same 474 frames over only 309 segments; filling by rank
reaches 363 — more independent scenes for the same evaluation cost.

**Calibration is disjoint from L1, and the selector asserts it.** INT4 AWQ is activation-aware: it fits ranges to
whatever it is calibrated on. Any overlap would flatter the quantised model on precisely the frames used to judge
it, and the quantisation delta would be worthless. Calibration is drawn only from what L1 did not take.

## Power

For a proportion measured on the 474-frame L1 slice:

| true rate | 95% CI half-width |
|---|---|
| 0.30 | ±4.1 pp |
| 0.50 | ±4.5 pp |
| 0.65 | ±4.3 pp |
| 0.90 | ±2.7 pp |

Minimum detectable difference at 80% power, α = 0.05 two-sided:

| comparison | MDE |
|---|---|
| two independent arms, p ≈ 0.5 | **9.1 pp** |
| two independent arms, p ≈ 0.65 | 8.7 pp |
| same frames, paired (McNemar), 10% discordant | **4.1 pp** |
| same frames, paired, 20% discordant | 5.8 pp |
| same frames, paired, 30% discordant | 7.0 pp |

Two consequences worth stating before anyone quotes a result. **Run every arm on the same frames and test paired** —
the unpaired MDE (9.1 pp) is more than double the paired one, and 474 frames cannot resolve a 5-point difference
between independent arms. And **a difference below ~4 pp is not measurable on this slice at all**, however precisely
it prints; report it as indistinguishable rather than as a win.

The calibration slice carries no power claim — 256 frames is a sample for fitting activation ranges, not for
estimating a rate.

## Provenance

Drawn from `frames_index.csv`: **2,223 frames over 512 segments**, extracted at 0.5 fps and 800 px wide from
blackbox segments with `person_events > 0` and `standby == False`, then deduped at dHash Hamming ≤ 6.

**Privacy.** The frames show colleagues who are identifiable and have not consented to publication. The filenames
here are opaque segment ids and carry no personal data, which is why the lists can be public while the images
cannot. Do not add the images to this repository, and do not publish them elsewhere.

**Discrepancy on record.** The task specification that requested these slices described L1 as *474 frames over 349
segments*. This selector produces 474 frames over **363** segments. The frame count matches exactly; the segment
count does not, so the original was produced by a slightly different rule. The selector here was **not** tuned to
reproduce 349 — fitting an algorithm to a remembered number would make it unreproducible for the next person. If a
canonical 349-segment list exists elsewhere, it should replace these files and the digests in the manifest should be
updated.
