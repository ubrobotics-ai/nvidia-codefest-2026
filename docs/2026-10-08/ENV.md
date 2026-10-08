# B300 cluster environment notes (extracted 2026-10-08)

## Cosmos3-Edge (G1 action policy)
- `cosmos-framework` at `0c60e98` + `scripts/g1/cosmos_framework_0c60e98.patch`.
- uv venv in the checkout (`.venv`): Python 3.13, torch 2.10 (cu130), Transformer Engine 2.12, flash_attn 2.7.4;
  `opencv-python` replaced by `opencv-python-headless==4.13.0.92` (the policy server failed on missing libxcb).
- Runs inside named pyxis containers on the team's shared NFS (`pt_g1`, `pt_g1b`, `pt_build`). Named containers are
  shared across jobs: give each concurrent job its own name; a new name needs `--container-image` once.
- `LD_LIBRARY_PATH` adds the venv's `nvidia/cu13/lib`, an ffmpeg-links directory and `av.libs`; see `scripts/g1/launch/`.
- Every srun from a non-interactive shell needs `ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH`.
- Unique torchrun port per job: `MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))`.

## RF-DETR (rescue detector)
- `roboflow/rf-detr` at `8913f7b`, venv `venvs/rfdetr` inside `nvcr.io/nvidia/pytorch:25.08-py3` (container `pt_rfinf`).
- Score on GPU only: CPU inference of the same checkpoint gave clearly different detections and AP.

## Nodes and performance
- Exclude dgx03 (NCCL all-gather stall at start-up, 2026-10-06), dgx06, dgx07 and dgx11 (NODE_FAIL or silent stalls);
  dgx08 is suspect.
- G1 training is CPU data-bound: 4 loader workers per rank run at ~93 % CPU while the GPUs wait (about 45 of 275 GB
  and ~190 W per B300). 12 workers ran 1.3x faster but changed the loss curve (it rose from ~1.4 to 2-3.4), so the runs
  keep 4. Turning activation checkpointing off gave no speed-up.
- One G1 iteration (global batch 2048, 4 x B300) takes about 60-67 s: 500 iterations about 8.5 h, 2,000 about 34 h.
- A submit plugin counts K8s and Slurm GPUs together; the effective per-team cap was about 8 GPUs.
