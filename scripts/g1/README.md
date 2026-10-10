# G1 box lift: Cosmos3-Edge action-policy post-training (B300 cluster)

Everything used to post-train Cosmos3-Edge as an action policy on the simulated Unitree G1 box-lift demos
(datasets `ubr-physical-ai/g1_hug_v3`, `_v4`, `_v5`, `_v6`, all private), extracted from the B300 cluster before access ended.

## cosmos-framework changes
Base: `NVIDIA/cosmos-framework` at commit `0c60e98` (head of PR #278). `cosmos_framework_0c60e98.patch` re-applies every
local change on a clean checkout of that commit (`git apply cosmos_framework_0c60e98.patch`); `cosmos_framework/` holds the
same files as plain copies. Contents:
- the G1 recipe: `G1HugLeRobotDataset` (LeRobot v3 loader, concat/ego viewpoints), the `action_policy_g1_hug_edge`
  experiment, domain 15 registration, the smoke launcher and every G1 TOML;
- `G1_HUG_EPISODES` (JSON list of episode indices; the train split keeps only those): data-efficiency subsets;
- `G1_MAX_SAMPLES_PER_BATCH` (default 128): per-GPU micro-batch override;
- the policy server's `viewpoint` / view-description fields, for the closed-loop client.

TOMLs: `v1_4gpu` (2,000 iterations, used for v1-v4), `v5full_4gpu` (the same, checkpoints every 250), `curve500_4gpu`
(500 iterations, LR cycle 500: the curve points), `ego_*` (head camera only), `smoke`. `v5full_fast_4gpu` (no activation
checkpointing, 256 per micro-batch) ran out of memory on B300 and `v5full_noac_4gpu` gave no speed-up; kept for the record.

## tools/
- `make_curve_subsets.py`, `make_v6_subsets.py`: nested / stratified episode subsets.
- `openloop_eval.py`: open-loop score of a policy server against recorded val episodes (per-phase errors, box frame).
- `upload_g1_v5.py`, `upload_g1_v6.py`: weights-only upload with the model card.

## launch/
The Slurm/pyxis launch, resume, chain and upload-watch scripts as run (cluster paths, job names and node exclusions
included as they were).

## Results (closed loop, 4 heights x 64 envs, held-out seed 9001, level lifts)
- v4 (measured-pose actions): 0/16. Replaying the recorded actions also gave 0/256: the actions lacked the controller's squeeze.
- v5 (commanded wrist targets), 500 iterations: N = 64 217/256 (84.8 %), N = 256 155/256 (60.5 %), N = 965 145/256 (56.6 %).
  At this fixed budget N = 64 sees each training window ~76 times, N = 965 ~5 times.
- v5, 2,000 iterations: N = 64 232/256 (90.6 %), N = 965 247/256 (96.5 %). With enough training more demos help;
  the 500-iteration ordering was undertraining. The scripted controller scores 256/256 on the same boxes.
- v6 (warehouse scenes, open hands), 500 iterations: lifts only at 0.65 m (N = 64 44/256 plain, 8/240 warehouse;
  N = 971 18/256 plain, 6/256 warehouse). Open-loop on recorded v6 val frames the N = 64 policy tracks the per-height
  wrist trajectory (16-frame chained error 1-2 cm, smallest at 0.80 m), so the failure is closed-loop (walk-up drift
  or the open-hand grip), not a failure to learn height.

## Checkpoints (Hugging Face, ubr-physical-ai)
Public: g1-hug-edge-concat-v5-n64 / -n256 / -n965 (500 it), -v5-n64-2k (2,000 it), -v6-n64, -v6-n971.
Private: -v5-full (N = 965, 2,000 it), -v5-n64-2k-iter1000 (mid-schedule fallback).
