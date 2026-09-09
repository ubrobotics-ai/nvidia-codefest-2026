#!/bin/bash
# Persistent single-GPU job for the night: serves Nemotron with vLLM at a reduced memory share so Cosmos / YOLO steps can
# run inside the SAME allocation via `srun --jobid=<id> --overlap` (one GPU, no second job queued).
#   sbatch --job-name=llm --gres=gpu:1 --time=24:00:00 --output=$WORK/logs/night-20260909/llm_%j.log scripts/llm_job.sbatch
set -uo pipefail
export TEAM=${TEAM:-/storage/hackathon_teams/omc-team15}
export WORK=${WORK:-$TEAM/codefest}
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=/tmp/enroot-$USER/run   # /run/user/<uid> is unwritable on DGX nodes
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
PORT=${PORT:-8150}          # 8000 is taken by other teams on the shared nodes
GPU_FRAC=${GPU_FRAC:-0.35}  # ~96 GB of 275: weights 64 GB + KV cache; leaves ~180 GB for the overlapping steps
echo "== start $(date -Is) job $SLURM_JOB_ID node $(hostname) git $(git -C $WORK/repo rev-parse HEAD) container ubr port $PORT gpu_frac $GPU_FRAC"
srun --container-name=ubr --container-mounts=$HOME:$HOME,$TEAM:$TEAM --container-workdir=$WORK \
  bash -c '
export HF_HOME=$WORK/hf-cache HF_HUB_OFFLINE=1 PATH=$HOME/.local/bin:$PATH
nvidia-smi -L
SNAP=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(\"'"$MODEL"'\"))")
exec vllm serve "'"$MODEL"'" --served-model-name nemotron \
  --max-num-seqs 8 --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization '"$GPU_FRAC"' \
  --host 0.0.0.0 --port '"$PORT"' --trust-remote-code \
  --reasoning-parser-plugin "$SNAP/nano_v3_reasoning_parser.py" --reasoning-parser nano_v3'
echo "== end $(date -Is) exit $?"
