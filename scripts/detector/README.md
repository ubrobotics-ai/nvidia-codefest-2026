# Detector scripts (permissive replacement for yolo26n)

The code behind `docs/2026-10-03/DETECTOR_PERMISSIVE.md`. Bench demo only; every result comes from
Isaac renders or COCO photos. Paths are the Codefest cluster's team storage; datasets and weights are
not in this repo (weights and ONNX exports: private HF `ubr-physical-ai/rescue-det-permissive`).

| folder | what |
|---|---|
| `rfdetr/` | RF-DETR (Apache-2.0) training (M at 1280, S at 640 / 768), harness inference, ONNX export, and the ModelOpt INT8 + TensorRT pipeline that measured INT8 recall |
| `dfine/` | D-FINE (Apache-2.0) configs (M and S at 1280x704), inference, ONNX export (two-input and the single-input variant for the rover runner), and `dfine_local_fixes.patch` |
| `multiclass/` | the 10-class set (person, person_lying, backpack, handbag, suitcase, dog, car, truck, box, object): dataset builders (v1; v2 adds the warehouse box renders), training, inference, COCO evaluation, per-class thresholds, export, and the OWLv2 pre-labeller with its calibration (rejected; see the doc) |
| `eval/` | `harness_replica.py` (recall at matched FA on the held-out hospital split, rule `+r` and variants), the fixed-threshold operating point, the deployed yolo26n scored with the rover's end-to-end decode, and the Cosmos3-Edge object-naming test |
| `cosmos_transfer/` | Cosmos Transfer 2.5 batch 2: clip download, per-clip prompts, the 8-GPU worker, and the per-frame label check and staging |

`dfine_local_fixes.patch` applies to D-FINE `956d170` and fixes three things that broke multi-GPU training
on a non-square input: the FLOPs counter's square dummy input, the CPU all-gather in COCO evaluation
(`cpu:gloo,cuda:nccl`), and a missing barrier before ranks reload `best_stg1.pth` at the stage switch.
