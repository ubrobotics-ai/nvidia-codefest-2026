# Cosmos Transfer 2.5 sim-to-real pilot, eight Isaac clips

Bench result on one B300 (SM103), against `ubr-physical-ai/isaac-sdg-rescue-target`, to decide
whether a video-to-video translation pass is worth running over all 109 clips.

**Decision: run it, as a 20-clip subset first, gated on a detector retrain.** Reasoning in
"The decision" below.

Three of this document's earlier conclusions were wrong and have been withdrawn. The way they
were wrong is the most transferable part and is kept in "What I got wrong".

## Configuration

93 frames, 1280x720, 36 steps, `seg` control at strength 0.7, `nvidia/Cosmos-Transfer2.5-2B`
via `scripts/cosmos_transfer_batch.py`. Source masks are `mode=L` instance ids `{0,1}`, which
are visually black; `colourise_ids` maps them to saturated colours, without which the control
carries nothing.

`--height 720`, not the script default 704. 720 is a multiple of 16, so no resample is needed;
a 720→704 squash shifts every mask by ~2% vertically and would corrupt an IoU test for reasons
unrelated to translation. **The earlier night-20260910 run used 1280x704 and therefore carried
that squash.**

## Results, all eight clips, both instruments

`COCO` is COCO-pretrained Faster R-CNN, person class, conf 0.5 — trained on neither domain
(real-photo-native, which is the point). `Isaac` is `ubr-physical-ai/rescue-target-yolo26n`,
fine-tuned on these very renders. Cells are `median IoU / fraction of frames at IoU >= 0.50`.

**Denominators differ**: COCO columns sample every 6th masked frame (n per row below); Isaac
columns score every masked frame (76–93 per clip). Do not read a COCO fraction and an Isaac
fraction as the same measurement.

| clip | px | n | COCO translated | COCO source | Isaac translated | Isaac source |
|---|---:|---:|---|---|---|---|
| hospital 5.0 m | 7190 | 13 | 0.000 / 0.31 | 0.730 / 0.54 | 0.000 / 0.34 | 0.960 / 0.63 |
| warehouse 5.0 m | 6757 | 16 | **0.894 / 0.94** | 0.865 / 0.75 | 0.886 / 0.94 | 0.975 / 1.00 |
| warehouse standing 7.5 m | 4350 | 16 | **0.879 / 1.00** | 0.896 / 1.00 | 0.000 / 0.00 | 0.053 / 0.00 |
| hall night 7.5 m | 3007 | 16 | **0.776 / 0.75** | 0.000 / 0.06 | 0.000 / 0.49 | 0.959 / 1.00 |
| hall fog 10 m | 1692 | 16 | **0.881 / 0.88** | 0.000 / 0.12 | 0.000 / 0.23 | 0.938 / 0.94 |
| hall no-vest 10 m | 1502 | 15 | 0.000 / 0.00 | 0.000 / 0.07 | 0.000 / 0.00 | 0.824 / 0.78 |
| rivermark 15 m | 620 | 16 | 0.000 / 0.00 | 0.000 / 0.06 | 0.000 / 0.00 | 0.000 / 0.41 |
| hall 20 m | 390 | 15 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.06 | 0.834 / 0.66 |

Hallucinations on empty-mask frames: 6 across all eight clips (Isaac detector), 0–3 per clip.

### What the table says

**Translation makes low-visibility casualties findable.** Night 0.000 → **0.776** and fog
0.000 → **0.881** by COCO. A real-photo detector essentially cannot see the casualty in the
Isaac source and can see it clearly after translation. That is the sim-to-real gap closing,
and it is the strongest result here.

**One clip regresses: hospital**, COCO source 0.730 → translated 0.000 median. It is the
LARGEST subject in the set at 7190 px, so this is not a size effect and I cannot explain it.

**Three clips are below resolution in both domains** — no-vest 10 m, rivermark 15 m, hall 20 m
— where COCO scores ~0 on source and translated alike. Those are a detection-resolution limit,
not a translation result, and the subset should not be judged on them.

**The two instruments disagree in both directions**, which is why both columns are shown:
- hall night and fog: Isaac says translation destroyed the casualty (0.959 → 0.000), COCO says
  it created a findable one (0.000 → 0.776). Exactly mirrored.
- warehouse standing: Isaac scores 0.053 on the SOURCE — it is fine-tuned on lying casualties
  and upright poses are its hard negatives — while COCO scores 0.896/1.00 on the same frames.
  I previously called this clip untestable; it is testable, with an instrument that can see it.
- warehouse 5 m: Isaac scores 0.886 on translated. So a sim-tuned detector does **not** always
  fail on good translation — size dominates, and the mirror appears only in the low-visibility
  cases.

## Control strength: inconclusive, and my earlier claim withdrawn

The earlier version concluded "no single strength works", from Isaac-detector numbers. That is
not safe: by this document's own argument a lower Isaac score on translated output is
ambiguous, and strength 1.0 adheres harder to the seg control, which plausibly keeps the frame
closer to the source look that an Isaac-tuned detector rewards.

Re-scored with COCO, same clips, n=15–16 per cell:

| clip | s0.7 COCO | s1.0 COCO |
|---|---|---|
| hall night 7.5 m | 0.776 / 0.75 | 0.731 / 0.75 |
| hall fog 10 m | 0.881 / 0.88 | 0.866 / **0.94** |
| hall 20 m | 0.000 / 0.00 | **0.375** / 0.00 |

The two are close, with a slight edge to 1.0 on the harder cases. **Nothing here supports the
earlier claim that 1.0 hurts.** 0.7 is retained as the default only because the subset was
already specified that way; a 0.7-vs-1.0 comparison on the 20-clip subset would settle it.

Note the earlier night-20260910 survival curve, scored with Grounding DINO (domain-neutral),
was monotonic in strength: 0.0% kept at 0.4, 15.9% at 0.7, 41.3% at 1.0. That also points at
1.0, from a third instrument.

## Reconciling the earlier "~13% of labels survive, 0% keep the vest"

That figure is `logs/night-20260910/B_curve_transfer.csv`, produced by
`scripts/transfer_fidelity.py`. Checked rather than assumed:

- **The scorer was NOT the Isaac detector.** It is Grounding DINO
  (`IDEA-Research/grounding-dino-base`) with the text prompt `"a person lying on the ground"`,
  and the script documents that a generic "person" prompt scores a prone figure at ~0.27
  against ~0.76 for that phrasing. So the instrument was domain-neutral and deliberately
  chosen. **That artefact does not apply.**
- **The prompt artefact DOES apply.** Those runs used
  `'industrial hall interior, low camera height, overcast daylight through high windows, dust
  in the air'` and `'industrial interior, ...'`. Neither names a person. The script's own
  docstring says so explicitly: *"the prompt rule forbids naming the subject, so nothing
  instructs Cosmos to keep the garment; this quantifies whether it does anyway."*
- The resolution was **1280x704**, carrying the vertical squash described above.

So the two runs are not in contradiction — they answer different questions. The old run
measured *does Cosmos keep the casualty when nothing tells it to*, and the answer was mostly
no. This pilot measures *does it keep the casualty when the prompt names it*, and the answer
is substantially yes.

**Consequence for the multi-control prescription.** Prescribing edge + seg + depth was a fix
for "nothing instructs Cosmos to keep the garment". The cheaper fix is to instruct it. That
does not prove multi-control is unnecessary — it has not been tested against a naming prompt —
but the evidence it was based on has a simpler explanation, so it should not be treated as
settled. Tracker A8.

## The prompt must name the casualty

The first run of this pilot used the script's default prompt for all eight clips. The night
clip rendered as a bright daylight warehouse, the outdoor street rendered as a warehouse, and
the casualty was erased in every clip: median IoU 0.000 on all eight, Isaac and COCO alike.

Re-run with per-clip prompts naming environment, lighting, posture and clothing, the Isaac
fraction went to 0.94 (warehouse 5 m), 0.49 (night), 0.34 (hospital), 0.23 (fog), 0.06 (20 m)
and 0.00 (no-vest, rivermark, standing) — per clip, not a uniform range.

The `PROMPT RULE` bans naming the vehicle and says geometry comes from the masks. In practice
**geometry from masks is not sufficient**: if the text does not name a person, the model draws
none. That is a limitation of the control path, not a style preference.

## What I got wrong

Three retractions, in the order they happened. Two are `§03` triage records — a rig fault then
an instrument fault — both caught before "Transfer destroys distant casualties" hardened into
a design decision.

1. **Rig.** Ran all eight clips with one default prompt. Produced eight zeros and would have
   read as total failure.
2. **Instrument.** Scored translated frames with a detector fine-tuned on the source domain,
   producing an apparent size cliff and the conclusion "Transfer destroys far casualties".
   Visual inspection of the 20 m frame disproved it: the casualty is plainly there, on the mask
   contour, in a hi-vis vest. The detector could not see it.
3. **Instrument, again.** Labelled two clips "untestable" on the same detector's source scores.
   COCO scores the standing-person clip 0.896/1.00 on source. Only rivermark is genuinely
   marginal.

The rule that came out of it: **never score sim-to-real output with a model trained on the
source domain, and always score the source with the same instrument.**

## Cost and the cluster behaviour that dominates it

~245 s per clip. 109 clips is **~7.4 GPU-hours of compute**.

The kill rate is **16.1%** (5 of 31 attempts across every Transfer run today), Wilson 95% CI
**7.1% to 32.6%**; `sacct` independently gives 6 FAILED of 38 steps = 15.8%. An earlier version
of this document said ~25%, which came from a single 8-run repeat (2 kills) and sat at the top
of that interval — the wider sample brings it down.

Practically the kills barely cost compute: at 16% they add ~130 submissions instead of 109 and
about **0.35 GPU-hours** of wasted work, because a kill lands 37–96 s in rather than near the
end. **The cost is orchestration, not compute** — a run without job-level retry loses the
whole batch, which is what makes this worth designing around rather than absorbing.

`exit 137` therefore hits roughly 1 job step in 6. Ruled out by measurement:

| candidate | evidence |
|---|---|
| GPU memory | 39 GB peak of 275 GB |
| Host memory | MaxRSS 3.5–6.2 GB against `--mem=800G` |
| `/dev/shm` | 1008 GB, 967 GB free, 18 GB peak |
| Duration | kills land 37–96 s in; successes run 127–292 s |
| Workload size | identical config: 6 successes, 2 kills in 8 runs |
| **Preemption** | **`PreemptMode=OFF`, `PreemptType=(null)` cluster-wide** |
| **Another tenant on the node** | **`OverSubscribe=NO`** — nodes are not shared between jobs |

`dmesg` is restricted on the nodes (`dmesg_restrict=1`) and journald shows nothing, so the
kernel's own account is unavailable. **No hypothesis survives.** It is recorded as unexplained.

Mitigations that work regardless:
- **One job submission per clip.** The SIGKILL takes down the whole `srun` step, so an in-job
  retry loop never reaches attempt 2. A job array with `--array=<failed ids>` for retries is
  the cleaner form of this.
- **Propagate the step exit code.** `sacct` reported `FAILED 0:9` correctly for a killed step
  that srun surfaced, but reports `COMPLETED 0:0` when a wrapper script swallows it. Echo
  `$?` explicitly.

## The decision

**Run a 20-clip subset now; run the remaining 89 only if the retrain arm shows a gain.**

Supporting it: translation demonstrably closes the domain gap on the clips that matter
(night 0.000 → 0.776, fog 0.000 → 0.881 by a detector trained on neither domain), the prompt
fix is understood, and 20 clips is ~1.4 GPU-hours.

Against running all 109 now: the endpoint is unproven. "More findable by COCO" is not "more
useful for training the rover's detector". Seven to eight GPU-hours would buy data we cannot
yet evaluate, and one clip (hospital) regresses for reasons unknown.

Configuration for the subset: per-clip prompts naming environment, lighting, posture and
clothing; strength 0.7 with a 1.0 comparison; one job submission per clip with retry; COCO and
Isaac scores recorded per clip on both source and translated, so the set ships with its own
evidence.

**Gates.** This pilot closes **A3b** (do labels survive Transfer — yes, when the prompt names
the casualty, on clips a neutral detector can resolve). It leaves **A5** open (does training on
translated frames help — TSTR), which is exactly what the retrain arm answers. Size that arm
against the real evaluation data we actually have: the 24-frame L1 set with 12 positives, plus
the judge set at 40 per lighting condition — not the "~17 rover frames" figure quoted earlier,
which is stale.

## What this does not establish

That translated frames improve the deployed detector on real footage. Settling it needs the
retrain with a translated-augmentation arm measured against real frames.
