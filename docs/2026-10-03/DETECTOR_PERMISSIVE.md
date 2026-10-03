# Permissive detector, multi-class model, and G1 runs (2026-09-17 to 2026-10-03)

Follows `docs/2026-09-17/HANDOFF.md` and `INDEX_DETECTOR.md`. Everything here is a bench demo.
Nothing is operationally ready, and every recall figure below comes from Isaac renders or COCO
photos, not rover footage.

## 1. Why

The deployed detector (yolo26n) and every YOLO arm are AGPL-3.0. The goal was a detector with a
permissive licence for both code and weights, at comparable recall and Jetson latency.

| family | licence | used |
|---|---|---|
| RF-DETR N / S / M / L | Apache-2.0 | yes |
| RF-DETR XL / 2XL | PML 1.0 (not permissive) | no; do not install `rfdetr[plus]` |
| D-FINE | Apache-2.0 | yes |
| YOLOv8 / yolo26 | AGPL-3.0 | no |
| TAO RT-DETR (N arms) | NVIDIA model terms | baseline only |

## 2. Metric

Unchanged from the ladder: frame-level recall at FA <= 0.005 on hard negatives (frames holding a
non-lying person), held-out hospital split (22,060 frames: 6,724 positive, 4,550 hard, 10,786
empty), rule `+r` (w/h >= 1.3), lowest threshold in 0.005 steps. Scored with the replica of the
bench harness. The bench has not yet re-scored these rows itself.

## 3. NVIDIA stack, longer training (closes INDEX_DETECTOR open item 1)

| arm | recall |
|---|---|
| N2 resnet_18 scratch, 12 epochs: final / best (epoch 11) | 0.163 / 0.185 |
| N2, 16 epochs | 0.180 |
| N1 rn50 warehouse, 12 epochs | 0.264 |
| N1, 16 epochs | 0.264 |

More epochs do not help (N1 at 4 epochs was 0.274). The TAO schedule keeps the LR constant, so
a decayed schedule is still untested.

## 4. Permissive detectors

All 12 epochs from COCO weights on the same rebal split, final EMA weights unless noted.

| model | input | recall (rule) | recall (no rule) |
|---|---|---|---|
| D-FINE-S | 1280x704 | 0.325 | |
| RF-DETR-M, trained at 1280, run at 640 | 640 | 0.343 | |
| D-FINE-M | 1280x704 | 0.369 | |
| **RF-DETR-S, trained at 640** | 640 | **0.370** | **0.437** |
| RF-DETR-S, trained at 768 | 768 | 0.372 | |
| RF-DETR-M, run at 960 | 960 | 0.395 | |
| RF-DETR-M | 1280 | 0.417 | 0.483 |
| reference: armB yolo26n@1280 / armA yolo26m@1280 (bench) | | 0.393 / 0.431 | |

- **Epochs.** RF-DETR-M at epochs 13-16 scored 0.392-0.409, and 0.393 with EMA weights at epoch
  16. Recall wanders by about 0.025 between epochs, so 12 is enough, and 0.417 is a good draw;
  0.39-0.42 is the fair range.
- **Checkpoint choice.** The epoch with the best COCO mAP was never the best on the harness
  (RF-DETR-M: 0.394 vs 0.417; S@640: 0.350 vs 0.370). Pick checkpoints with the harness.
- **The w/h rule.** It was set for off-the-shelf detectors that fire on upright people. On
  detectors trained for lying people it mostly removes true casualties seen end-on: RF-DETR-S
  gains 0.067 without it. Whether to keep it is the bench's call; YOLO arms must be re-scored
  under the same rule before comparing.

## 5. Jetson latency

jetson-0 (Orin Nano, 15 W), TensorRT 10.16 FP16, p95 from the saved engine, measured by the
sim session. The VLM column runs Cosmos3-Edge generating back to back.

| model | idle | with VLM |
|---|---|---|
| yolo26n@640, cv2.dnn on CPU (the rover today) | 140.3 ms | 178.5 ms |
| yolo26n@640, TensorRT | 7.8 ms | 18.8 ms |
| armB yolo26n@1280 (15 Sept) | 20.25 ms | |
| **RF-DETR-S@640** | **27.9 ms** | **65.7 ms** |
| RF-DETR-M@640 | 28.8 ms | 66.6 ms |
| D-FINE-S@1280x704 | 33.3 ms | 76.7 ms |
| RF-DETR-S@768 | 38.6 ms | 90.6 ms |
| D-FINE-M@1280x704 | 49.0 ms | 115.7 ms |
| RF-DETR-M@960 | 64.2 ms | 150.6 ms |
| armA yolo26m@1280 (15 Sept) | 65.35 ms | |
| RF-DETR-M@1280 | 135.1 ms | 315.8 ms |

- The VLM slows every model by 2.30-2.36x, across four model families.
- The contending VLM held about 3.1 GB, not the 4.6 GB recorded on 15 Sept, so it may not be the
  same model; the matching factor suggests an equivalent load.
- All engines build with no plugins. RT-DETR needs one.
- **INT8 does not help RF-DETR on Orin.** ModelOpt INT8 costs under 0.01 recall but runs 2-7%
  slower than FP16: the fused attention stays FP16, and quantize steps cancel the GEMM savings.
- RF-DETR-N / S / M share one backbone; at a given input size they cost about the same. Input
  size is the lever, not model size.

## 6. The deployed detector on the same split

Stock yolo26n, end-to-end export `[1,300,6]`, decoded as the rover does (confidence filter, no
NMS, person = class 0).

| at confidence >= 0.25 | casualty frames found | standing-person frames fired | empty frames fired |
|---|---|---|---|
| yolo26n, any person box | 0.127 | 0.268 | 0.015 |
| RF-DETR-S@640, lying class | 0.617 | 0.069 | 0.025 |

On the harness: yolo26n 0.122 with the rule, 0.008 without it. The bench's earlier B0 row for
the same model was 0.024; the decode paths differ and the two figures should be reconciled
before either is quoted.

## 7. Ten-class model (for the detector + Cosmos3-Edge pairing)

RF-DETR-S@640, 12 epochs on 208,584 images: COCO 2017, LVIS v1 box / crate / carton, and the
Isaac rebal split relabelled from the full annotations. Channels: 0 person (any pose),
1 person_lying, 2 backpack, 3 handbag, 4 suitcase, 5 dog, 6 car, 7 truck, 8 box, 9 object (every
other COCO class plus Isaac distractor primitives), 10 unused. A lying person carries both a
person and a person_lying box.

| class | AP50:95 | AP50 | threshold for 80% precision | found at that threshold |
|---|---|---|---|---|
| person_lying | harness 0.374 (rule) / 0.435 (no rule) | | 0.54 / 0.575 | |
| person | 0.596 | 0.835 | 0.40 | 0.75 |
| backpack | 0.242 | 0.425 | 0.60 | 0.15 |
| handbag | 0.244 | 0.416 | 0.55 | 0.19 |
| suitcase | 0.520 | 0.733 | 0.45 | 0.56 |
| dog | 0.732 | 0.858 | 0.40 | 0.82 |
| car | 0.488 | 0.734 | 0.45 | 0.61 |
| truck | 0.473 | 0.625 | 0.55 | 0.38 |
| box | 0.271 | 0.440 | 0.40 | 0.20 |
| object | 0.485 | 0.691 | 0.45 | 0.55 |

- Casualty recall equals the single-class model's, so the extra classes cost nothing there.
- Box is measured only on the 351 images that carry LVIS box labels (optimistic). It has no
  warehouse cartons in training; labelled warehouse renders were requested from the gym session.
- Label gaps remain: LVIS is federated, and Isaac scenes contain unlabelled props. OWLv2 was
  calibrated as a pre-labeller and rejected: at 80% precision it finds 86% of dogs but 5% of
  boxes and handbags, and never reaches 80% on backpacks.

**Can Cosmos3-Edge name the `object` boxes?** 476 COCO objects (6 per class), TensorRT INT4:

| shown as | base | SFT v4 |
|---|---|---|
| crop | 0.775 | 0.769 |
| full frame, one red outline | 0.813 | 0.815 |

Lexical scoring, so roughly a few points either way. Weak on small kitchen items and bag types.
Crops fail on people (the model names what they hold). Numbered regions were near chance in the
earlier L0 test, so one outlined box per query is the recommended form.

## 8. Cosmos Transfer, batch 2

81 more clips (hospital excluded), seg control at 0.7: 7,533 frames. Per-frame label check
(COCO detector at IoU >= 0.5): 2,726 kept (2,186 lying, 540 standing or kneeling hard negatives),
1,753 lost, 2,036 too small to check, 1,018 without a label. The size filter biases the kept set
toward near range. Not yet used in training.

## 9. G1 Cosmos3-Edge action policy (for the box-lifting workstream)

Run here at that workstream's request, from its recipe (cosmos-framework `0c60e98` plus its
patch): 2,000 iterations, global batch 2,048, 4x B300 each.

| run | view | status |
|---|---|---|
| ego v1 | head camera only | loss 18.22 / 8.22 / 1.80 at 1 / 25 / 100, 1.06 at 1,135 |
| concat v1 | head + both wrists | started 2026-10-03 |

Checkpoints go to private model repos `ubr-physical-ai/g1-hug-edge-ego-v1` and `-concat-v1`.
Closed-loop evaluation needs RTX GPUs (Isaac Sim cameras) and runs elsewhere.

## 10. Assets

Private HF model repo `ubr-physical-ai/rescue-det-permissive`: ONNX exports of RF-DETR-M (640,
960, 1280, and INT8 960 / 1280), RF-DETR-S (640, 768), D-FINE-M and D-FINE-S (single-input
variants for the rover runner), and `rfdetr_s640_10class_v1.onnx` with its class map and
thresholds. Prediction files stay on the cluster.

## 11. New gotchas

1. **Named pyxis containers live on shared storage.** Two concurrent jobs using the same
   `--container-name` lock each other ("Could not acquire rootfs lock") or interfere. Give every
   concurrent job its own name; a new name needs `--container-image` the first time.
2. **torchrun's default port collides** between jobs on one node. Set `--master_port` from the job
   id.
3. **`srun --overlap` into a running job does not see that job's GPUs.** It reports idle GPUs. Read
   utilisation from the job's own logs.
4. **Watch for `NODE_FAIL`.** One run died on a node fault and went unnoticed for hours. Watchers
   must cover every terminal state, and timers must count from RUNNING, not submission.
5. **D-FINE on non-square input and multi-GPU** needed three local patches: the FLOPs counter
   builds a square dummy input; COCO evaluation gathers CPU tensors (init with
   `cpu:gloo,cuda:nccl`); and ranks reload `best_stg1.pth` at the stage switch with no barrier
   (hung the 4-GPU run).
6. **RF-DETR's venv** needs `numpy<2` and `opencv-python-headless<4.11` on the NGC 25.08 image.
7. **Ultralytics 8.4 exports the raw `[1,84,8400]` head by default.** `nms=False` selects the
   YOLO26 end-to-end `[1,300,6]` head that the rover uses.
8. **transformers 5** renamed OWLv2's post-processing to `post_process_grounded_object_detection`.
