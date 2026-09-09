# nvidia-codefest-2026 — cluster scripts

Cluster-side scripts from **Team UBR Stack** (UB Robotics) for the NVIDIA / OpenHackathons / Oracle
**Open Models Codefest 2026** — theme: *agentic AI for your region's public-sector needs*, built on open models.

Our entry is an offline-first **physical AI search-and-rescue teammate**: a rescue coordinator briefs a mission, and
the mission then runs entirely on the edge (Jetson Orin Nano), so the robot keeps searching when the network degrades
or drops. Feeding it is a hybrid data pipeline — real UGV/UAV footage plus synthetic scenes — and that pipeline is
what runs on the cluster.

This repository holds **only the scripts that run on the NVIDIA B300 nodes**. Edge/robot code and project
documentation live elsewhere.

## Scripts

| Script | What it does |
|---|---|
| [`scripts/cluster_build_container.sh`](scripts/cluster_build_container.sh) | Builds the named pyxis container `ubr` once — NGC PyTorch base + Diffusers, vLLM and the Cosmos Framework — and pre-downloads the model weights. |
| [`scripts/cosmos_i2v_batch.py`](scripts/cosmos_i2v_batch.py) | Generates N image→video clips with Cosmos 3 from real UGV frames, four condition variants (day, dusk smoke, night, dense smoke), and times every clip. |

## Environment

Slurm with **pyxis + enroot** (no Docker), one **B300 SXM6** per job, CUDA 13.0. Written against
`nvcr.io#nvidia/pytorch:25.08-py3`.

Models used: `nvidia/Cosmos3-Nano` (16B image→video) and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
(prompt upsampling, served with vLLM). Both need their licences accepted on Hugging Face first.

## Usage

Build the container once, from a login node inside `tmux`:

```bash
export HF_TOKEN=hf_...          # accept the Cosmos 3 + Nemotron licences on huggingface.co first
./scripts/cluster_build_container.sh
```

Put source frames in `data/real_frames/` (jpg/png), then generate:

```bash
srun --gres=gpu:1 --pty --container-name=ubr --container-mounts=$HOME:$HOME \
  python scripts/cosmos_i2v_batch.py \
    --frames data/real_frames --out data/synthetic --n 20 \
    [--nemotron-url http://<vllm-host>:8000/v1]
```

Without `--nemotron-url` the script falls back to a fixed structured prompt instead of upsampling one.

Each clip writes an `.mp4` plus a `.json` sidecar (prompt + ground-truth tags). `data/synthetic/timing.csv` ends with
`mean_s_per_clip`, which is the number that decides how much synthetic data is affordable within the GPU budget.

Useful knobs: `--model`, `--frames-per-clip` (default 189), `--steps` (35), `--height` / `--width` (720×1280),
`--seed`.

## Data

No datasets are committed. `data/real_frames/` and `data/synthetic/` are placeholders — see the READMEs in each.

## Licence

Apache-2.0 — see [LICENSE](LICENSE). The NVIDIA models the scripts pull are covered by their own licences
(NVIDIA Open Model License, OpenMDW 1.1).
