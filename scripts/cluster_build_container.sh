#!/bin/bash
# Build the named pyxis container `ubr` once. Run from the LOGIN node, inside tmux.
# Prereqs: export HF_TOKEN=hf_... (licences for nvidia/Cosmos3-Nano accepted on huggingface.co)
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
export TEAM=${TEAM:-/storage/hackathon_teams/omc-team15}
export WORK=${WORK:-$TEAM/codefest}
# enroot must not use /run/user/<uid> (not writable on dgx01) and home is only 50 GB: keep the big stuff on team storage
# runtime path should be node-local (enroot creates mounts under it); data/cache are the big ones and go on the share
export ENROOT_RUNTIME_PATH=${ENROOT_RUNTIME_PATH:-/tmp/enroot-$USER/run}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$TEAM/enroot/data}
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$TEAM/enroot/cache}
export XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_DATA_PATH $ENROOT_CACHE_PATH $WORK/hf-cache $WORK/repo
module load python312 py-uv tree 2>/dev/null || true

srun --gres=gpu:1 --time=01:30:00 --job-name=build --pty \
  --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
  --container-name=ubr \
  --container-mounts=$HOME:$HOME,$TEAM:$TEAM \
  --container-workdir=$WORK \
  --container-env=HF_TOKEN \
  bash -c '
set -e
export HF_HOME='"$WORK"'/hf-cache
echo "== GPU"; nvidia-smi -L
echo "== python deps"
pip install -U pip
pip install -U "diffusers>=0.36" transformers accelerate imageio[ffmpeg] huggingface_hub cosmos_guardrail
pip install -U "vllm[omni]" || pip install -U vllm
[ -d '"$WORK"'/cosmos-framework ] || git clone https://github.com/NVIDIA/cosmos-framework.git '"$WORK"'/cosmos-framework
pip install -e '"$WORK"'/cosmos-framework
echo "== models"
huggingface-cli download nvidia/Cosmos3-Nano
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
echo "== sanity"
python - <<PY
import torch, diffusers, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)
from diffusers import Cosmos3OmniPipeline; print("Cosmos3OmniPipeline OK")
PY
du -sh '"$WORK"'/hf-cache
echo READY'
