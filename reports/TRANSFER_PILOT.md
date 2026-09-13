# Cosmos Transfer 2.5 sim-to-real pilot, eight Isaac clips

Bench result. Run on one B300 (SM103) against `ubr-physical-ai/isaac-sdg-rescue-target`
clips, at the request of the Isaac SDG session, to decide whether a video-to-video
translation pass is worth running over all 109 clips.

**Headline: yes it closes the domain gap where a subject is resolvable, and the first two
conclusions I drew were both wrong.** They are kept below, because the way they were wrong is
the transferable part.

## Configuration

93 frames, 1280x720, 36 steps, `seg` control at strength 0.7, `nvidia/Cosmos-Transfer2.5-2B`
through `scripts/cosmos_transfer_batch.py`. Source masks are `mode=L` instance ids `{0,1}`,
which are visually black; the script's `colourise_ids` maps them to saturated palette colours
before conditioning, without which the control signal carries nothing.

`--height 720`, not the script's default 704. 720 is a multiple of 16 so no resample is
needed, and a 720 to 704 squash would shift every mask by up to 2% vertically and corrupt an
IoU test for reasons unrelated to translation.

## The prompt has to name the casualty

The first run used the script's default prompt for all eight clips. The night clip rendered as
a bright daylight warehouse, the outdoor street rendered as a warehouse, and **the casualty was
erased in every clip** — every clip scored median IoU 0.000. That run measured the prompt.

Re-run with per-clip prompts naming environment, lighting, posture and clothing. Detection
went from uniformly 0.00 to 0.23–0.49 of frames immediately.

The script's `PROMPT RULE` bans naming the vehicle (it once rendered as a wheelchair) and says
geometry comes from the masks. In practice **geometry from masks is not sufficient**: if the
text does not name a person, the model draws none, whatever the segmentation control says.
That is a limitation of the control path, not a style preference.

## The acceptance test was built out of the wrong instrument

Scored with `ubr-physical-ai/rescue-target-yolo26n`, which is **fine-tuned on these Isaac
renders**, the result looked like a size cliff: 0.94 of frames at 5 m falling to 0.06 at 20 m,
and a conclusion that Transfer destroys distant casualties.

Then visual inspection of the 20 m frame showed a person in a hi-vis vest lying exactly on the
mask contour. The casualty was never erased. The detector could not see it.

**Scoring sim-to-real output with a detector trained on the source domain conflates "the
translation destroyed the subject" with "a sim-trained detector does not transfer" — and the
second is guaranteed if the translation works.** The better the translation, the worse a
sim-only detector performs on it.

Re-scored with COCO-pretrained Faster R-CNN, person class, neutral to both domains, every 6th
frame:

| clip | subject px | COCO translated | COCO source | Isaac-detector translated |
|---|---:|---|---|---|
| warehouse 5 m | 6811 | 0.894 / **0.94** | 0.865 / 0.75 | 0.886 / 0.94 |
| **hall fog 10 m** | 1616 | **0.881 / 0.88** | 0.000 / **0.12** | 0.000 / 0.23 |
| hall 20 m | 370 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.06 |
| rivermark 15 m | 608 | 0.000 / 0.00 | 0.000 / 0.06 | 0.000 / 0.00 |

*(median IoU / fraction of frames at IoU >= 0.50)*

The fog row is the result. A general detector finds the casualty in **88% of translated frames
and 12% of source frames** — translation took a hard case and made it findable to a detector
that had never seen Isaac. The Isaac-tuned detector scores the same clip 0.23 translated
against 0.94 source, exactly mirrored.

Where no detector resolves the subject (370–883 px), it fails in **both** domains. That is a
detection-resolution limit the SDG session already measures (0.27 recall at 20 m), not
something Transfer introduces.

## Control strength is not a knob that fixes it

| clip | strength 0.7 | strength 1.0 |
|---|---:|---:|
| hall night 7.5 m | 0.49 | **0.55** (median IoU 0.000 to 0.762) |
| hall fog 10 m | 0.23 | 0.12 |
| hall 20 m | 0.06 | 0.00 |

Helps the largest, hurts the two smaller. No single strength works across the range.

## Cost, and the cluster behaviour that dominates it

~245 s per clip. 109 clips is about 7.4 GPU-hours on a card four people share.

`exit 137` hits roughly **25% of job steps** with no memory pressure of any kind: 39 GB GPU of
275, 6 GiB host of 800 requested, `/dev/shm` 1008 GB with 967 free. Kills land 37–96 s in while
successes run 127–292 s, so it is neither duration nor workload size — an identical config run
eight times gave 6 successes and 2 kills. **The SIGKILL takes down the whole `srun` step**, so
an in-job retry loop never reaches its second attempt; retries must be separate job
submissions. `sacct` reports `COMPLETED 0:0` over a SIGKILLed payload.

## What this does not establish

That translated frames improve **our deployed detector on real footage**. "More findable by
COCO" is not "more useful for training the rover's detector", and the real evaluation set is
about 17 rover frames. Settling it needs a detector retrain with a translated augmentation arm
measured against real frames — a different and much larger piece of work.
