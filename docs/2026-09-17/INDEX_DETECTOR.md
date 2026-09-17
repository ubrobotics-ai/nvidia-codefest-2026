# Index: lying-person detector tests (B0, B1, D1, arm A/B/C, N0, N1, N2)

Snapshot of 2026-09-17. This page is an index. The sources of record are the bench harness
(`baseline/report.py`, private `ubr-isaac`), its write-up `docs/baseline-b0-vs-d1.md`, and
the run directories listed below.

## The task and the deciding metric

- **Task:** one binary decision per frame: *does this frame contain a person lying on the
  ground?*
- **Deciding metric:** recall at a matched false-alarm rate, where false alarms are measured
  on **hard negatives only** (frames holding a person in a non-lying pose). Empty frames are
  reported separately and are not in the denominator.
- **Rule `+r`:** a box counts only if its width/height ratio is at least 1.3. The rule was
  fixed in advance.
- **Threshold:** the lowest confidence (searched in 0.005 steps) whose hard-negative false-
  alarm rate is at or below the target.
- **Headline budget:** FA <= 0.005. The budget the mission can actually afford is still owed.

## Three instruments. Never mix them.

| instrument | frames | positives / hard / empty | what it is |
|---|---|---|---|
| **A** | 2,396 | 386 / 1,362 / 648 | v2 validation split of the Isaac renders |
| **B** | 13,376 | — | v3 split, used for the posture-ceiling (oracle) and rule sweeps only |
| **H** | 22,060 | 6,724 / 4,550 / 10,786 | **held-out hospital** split of the v3/rebal data. Training data contains no hospital frames |

Split H is built by `prepare_from_hub.py` / `prepare_coco_for_tao.py` with
`--empty-ratio 0.25 --seed 1234`: 91,647 train images, 56,912 train boxes and 6,724 val
boxes. All hard negatives are kept; person-free frames are sampled at 25%.

## Arm inventory

| id | model | init | trained on | input | notes |
|---|---|---|---|---|---|
| B0 | yolo26n | COCO | — | 640 | off the shelf |
| B1 | PeopleNet (DetectNet_v2) | NVIDIA | — | 960x544 | off the shelf. FP32; ONNX Runtime and TensorRT agree |
| D1 | `rescue_person80` (yolo26n) | COCO | v2 Isaac | 640 | bench-trained |
| D1far | `rescue_person80_far` | COCO | v2 Isaac | 640 | **80% contaminated** on instrument A |
| ref | yolo26n | COCO | v3, all negatives | 640 | 12 epochs, batch 24 |
| rebal | yolo26n | COCO | v3, empties cut to 25% | 640 | 12 epochs |
| armB | yolo26n | COCO | rebal | 1280 | 12 epochs, batch 24. HF `rescue-det-armB` |
| armA | yolo26m | COCO | rebal | 1280 | 16 epochs (3.5 h time budget). HF `rescue-det-armA` |
| armC-cos / armC-ctl | armA + second stage | armA | 698 Cosmos frames / 698 Isaac frames | 1280 | 8 epochs. HF `rescue-det-armC` |
| N0 | RT-DETR rn50 warehouse (NGC `deployable_rn50_v1.0.2`) | NVIDIA | — | 640 | off the shelf. 7 classes, Person is index 0 |
| N1-sq | RT-DETR rn50 | warehouse `trainable_rn50_v1.0.2` | rebal | **544 wide x 960 high (squashed by mistake)** | 12 epochs |
| N1 | RT-DETR rn50 | warehouse | rebal | 1280x704 | 4 epochs. 2 classes, person is index 1 |
| N2 | RT-DETR **resnet_18** | **none (from scratch)** | rebal | 1280x704 | 4 epochs. Person is index 1 |
| N1-e12 | as N1 | warehouse | rebal | 1280x704 | **12 epochs, still running** |
| N2-e12 | as N2 | none | rebal | 1280x704 | **12 epochs, finished, not yet scored** |

N1 and N2 differ in backbone **and** in initialisation. The rn50 warehouse checkpoint cannot
initialise a resnet_18: it leaves 41 tensors with mismatched shapes and TAO segfaults on
them.

## Results: instrument A (bench harness, rule +r)

| arm | FA <= 0.002 | FA <= 0.005 | FA <= 0.010 | FA <= 0.020 |
|---|---|---|---|---|
| B0+r | 0.365@0.145 | 0.402@0.090 | 0.440@0.045 | 0.459@0.030 |
| B1+r | 0.218@0.325 | 0.251@0.180 | 0.272@0.130 | 0.290@0.100 |
| **D1+r** | 0.562@0.465 | **0.575@0.400** | 0.611@0.260 | 0.632@0.145 |
| D1far+r (contaminated) | 0.723@0.730 | 0.736@0.700 | 0.754@0.600 | 0.754@0.525 |

Every off-the-shelf arm fails mainly at detection rather than posture. A cascade in which
PeopleNet proposes and D1 decides scores 0.241, below D1 alone. Of D1's misses, PeopleNet
recovers 2.

## Results: instrument H, recall at FA <= 0.005

### Reported by the bench harness

| arm | recall |
|---|---|
| B1 PeopleNet | 0.021 |
| B0 COCO yolo26n | 0.024 |
| rebal yolo26n@640 | 0.362 |
| armB yolo26n@1280 | 0.393 |
| yolo26m@640 | 0.429 |
| armA yolo26m@1280 | **0.431** |

The bench also reported an arm labelled "AMR" at 0.014. Its definition is held by the bench.

B0 and B1 are a measured tie: paired test, 192 discordant frames, p = 0.130.

### N arms, scored with a replica of the bench harness

The replica uses the same rule, the same denominator and the same threshold search. The
bench has not yet scored these rows itself.

| arm | FA <= 0.002 | **FA <= 0.005** | FA <= 0.010 | FA <= 0.020 |
|---|---|---|---|---|
| N0 | 0.123@0.725 | **0.152@0.640** | 0.173@0.565 | 0.204@0.485 |
| N2 (4 epochs) | 0.135@0.710 | **0.167@0.655** | 0.194@0.610 | 0.221@0.560 |
| N1-sq (12 epochs) | 0.200@0.660 | **0.217@0.605** | 0.238@0.545 | 0.260@0.485 |
| N1 (4 epochs) | 0.243@0.730 | **0.274@0.685** | 0.292@0.655 | 0.314@0.625 |

Without the rule, N0 collapses to 0.002, because it fires on upright people and those
detections consume the whole hard-negative budget. N1 moves to 0.293 and N2 to 0.173. The
rule is what makes N0 usable at all.

### Correction: superseded N numbers

An earlier scoring pass reported **N0 0.0097, N1-sq 0.2412, N1 0.2863 and N2 0.1703**. That
pass used box-level recall at IoU 0.5 and counted **all 15,336 non-positive frames** as
negatives, without the +r rule. **It is not the harness metric** and must not be put in a
column with B0, B1, armA or armB. It also produced the claim that "warehouse pretraining is
the worst off-the-shelf arm", and **that claim is withdrawn.** Under the harness definition,
N0 at 0.152 sits *above* the reported B0 (0.024) and B1 (0.021). Treat that ordering as
provisional until the bench scores all three on one run.

One finding from the N0 output stands, because it does not depend on the metric. On hospital
frames N0 fired `Forklift` 1,477 times, at up to 0.938 confidence, and 673 of those were on
frames containing a casualty. N0 mislabels casualties as equipment with confidence.

## Other detector measurements

**Ablation.** These figures come from Ultralytics `results.csv` recall on split H. That is a
different number from the harness column above.

| step | delta |
|---|---|
| negative balance (all negatives to empties cut to 25%) | +0.002, null |
| resolution 640 to 1280 (yolo26n) | +0.014 recall, +0.024 mAP50-95 |
| capacity n to m at 1280, epoch-matched at 12 | +0.036 recall |

**Arm C** (second stage on armA, `results.csv` on split H):

| arm | P | R | mAP50-95 |
|---|---|---|---|
| armA start | .887 | .395 | .362 |
| armC-ctl | .894 | .394 | .365 |
| armC-cos | .913 | .384 | .356 |

The translated-frame arm moves recall about 14x more than the Isaac control does, drifting
down while the control stays flat. Whether that shift is toward reality needs real frames.

**Validation mAP50 by epoch** (TAO; this is not recall):

- N1-sq: .292 .358 .348 .353 .335 .350 .305 .351 .356 .335 .358
- N1: .358 .355 .346 .369
- N2: .156 .186 .160 .263
- N1-e12 (running): .335 .371 .359 .339 .365 .388 .343 .382 .387
- N2-e12: .191 .214 .136 .211 .232 .263 .260 .265 .224 .259 .299 .258 (never converged)

**Posture bands on 698 known-lying translated frames** (gym classifier). Horizontal 146,
unclear 549, upright (wrong) 3 (0.4%). A single rule at 1.3 calls 29.9% of these people
upright (5th-percentile aspect 0.70). Treating unclear as negative misses 79.1%.

**Latency on jetson0, p95:**

| model | GFLOPs | latency |
|---|---|---|
| deployed yolo26n@640, `cv2.dnn` on **CPU** | 5.5 | 271.5 ms |
| yolo26n TensorRT FP16, idle | 5.5 | 6.32 ms |
| yolo26n TensorRT FP16, with VLM running | 5.5 | 14.57 ms |
| armB | 22.7 | 20.25 ms |
| RT-DETR rn50@640 | 121.7 | 29.23 ms |
| armA (idle) | 275.3 | 65.35 ms |

## Prediction files

The HF dataset `ubr-physical-ai/rescue-det-preds` (private) holds N0 (all classes, and a
person-only version), N1-sq, a manifest and the inference patch. N1 and N2 JSONL files are on
the cluster at `codefest/ubr-det/preds/` and have not been uploaded yet.

## Open

1. Score N1-e12 and N2-e12. For N2, report both the best epoch (11) and the final epoch.
2. Have the bench score N0, N1 and N2 on its own harness in one run with B0 and B1, to settle
   the ordering.
3. A fair backbone arm (resnet_18 with an ImageNet backbone init) is still untested.
4. Real-frame recall needs the judge-set capture. It does not exist yet.
