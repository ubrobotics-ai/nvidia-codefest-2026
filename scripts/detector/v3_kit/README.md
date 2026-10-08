# Detector v3 kit: re-textured warehouse boxes (Cosmos-Transfer 2.5) + RF-DETR-S retrain

Goal: improve the `box` class of the 10-class rescue detector (v2b: 0.333 AP50 on an unseen warehouse stage) by
re-texturing the warehouse box renders with Cosmos-Transfer 2.5, then retraining with the same recipe as v1/v2/v2b.
Deadline: a scored, exported v3 by Wed 7 Oct ~04:00 UTC; otherwise the rover keeps v2b. The curve-2 evals keep GPU priority.

## Files
- `manifest.jsonl`: every v2b image -> its source (`coco`, `rescue`, `wbox`) and path inside it (249,325 lines).
- `train_annotations_v2b.coco.json`, `valid_annotations_v2b.coco.json`: v2b's labels (10 classes, ids 1..10).
- `retexture_list.json`: the 3,000 renders to re-texture (seed 1234, all contain boxes), each with its prompt.
- `pick_retexture.py`, `make_manifest.py`: how those were made (cluster paths; for the record only).
- `build_v3.py`, `check_retexture.py`, `train_mc_v3.py`, `infer_mc_any.py`: the steps below.

## Steps
1. Sources (about 60 GB):
   - COCO 2017: `train2017.zip`, `val2017.zip` from images.cocodataset.org -> `COCO/train2017`, `COCO/val2017`.
   - `ubr-physical-ai/isaac-sdg-rescue-target` @ `92bb89405242f204dd8d76be45877808bbdd98dd`: extract all `train-*.tar` and
     `validation-*.tar` -> `RESCUE/train`, `RESCUE/validation` (the manifest paths are `train/<f>`, `validation/<f>`).
   - `ubr-physical-ai/isaac-sdg-warehouse-boxes` @ `3988dba` -> `WBOX/train`, `WBOX/validation`.
2. RF-DETR: `pip install git+https://github.com/roboflow/rf-detr@8913f7ba5d5534613474d6b818fe43ab18edcce6` (the exact
   commit v1/v2/v2b were trained with), plus pycocotools.
3. Re-texture: for each entry of `retexture_list.json`, run Cosmos-Transfer 2.5 on `WBOX/<rel>` with EDGE control at high
   strength (geometry must not move; start at 1.0, lower only if the output copies the stock look), its `prompt`, and
   write the result as `RETEX/<basename of rel>.jpg` at EXACTLY the source size (1280x720; resize back if the model
   works at another size). A still image can go in as a single frame or a short repeated clip; keep one output frame.
   First test 8 renders and time them; size the batch to finish by ~15:30 UTC.
4. Check before training: `python check_retexture.py <v2b .pth> WBOX RETEX 300`. Expect re-textured box recall below the
   source's but not collapsed (if it drops by more than about half, the boxes moved: raise the edge strength). Look at
   `check_sheet.jpg`: red label rectangles must still sit on boxes in the right-hand images.
5. Build: `python build_v3.py data_v3 COCO RESCUE WBOX RETEX` (prints the images and boxes added).
6. Train on 4 GPUs (keep 8 images per GPU: the recipe's global batch is 32):
   `torchrun --nproc_per_node=4 --master_port=<free> train_mc_v3.py data_v3 runs/mc_v3`. It resumes after interruptions.
7. Score on GPU (CPU inference gives different numbers; do not use it): `mkdir preds`, then one shard per GPU:
   `CUDA_VISIBLE_DEVICES=k python infer_mc_any.py runs/mc_v3/last_ema.pth v3 data_v3 k 4 - .` for k = 0..3.
   Send back `preds/mc_v3.s*.jsonl` and `preds/mc_v3.s*.coco.json` (the B300 side runs the comparison and the hospital
   harness), and keep `runs/mc_v3/last_ema.pth`.
