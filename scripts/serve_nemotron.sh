#!/bin/bash
# Serve Nemotron-3-Nano with vLLM from the `ubr` container. Run from the LOGIN node inside tmux; it blocks
# for the life of the server. Find the node afterwards with `squeue -u $USER -h -n nemotron -o %N`.
# Flags follow the model card (https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16); the
# reasoning parser ships inside the model repo, so it is taken from the local snapshot -- no extra download.
set -euo pipefail
export TEAM=${TEAM:-/storage/hackathon_teams/omc-team15}
export WORK=${WORK:-$TEAM/codefest}
# Same enroot runtime redirect as cluster_build_container.sh: srun propagates XDG_RUNTIME_DIR, and the default
# /run/user/<uid> is not writable on the DGX nodes, so pyxis fails in task_init() before anything runs.
export ENROOT_RUNTIME_PATH=${ENROOT_RUNTIME_PATH:-/tmp/enroot-$USER/run}
export XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
# Scratch: node-local /tmp is 1.6 TB free at 1.9 GB/s; team storage is NFS from dgx02 at 871 MB/s, and CUDA JIT
# plus enroot unpacking are latency-sensitive. (.bashrc sets TMPDIR for interactive shells only -- it returns early
# when non-interactive, which is why every script must set this itself.) Note /tmp INSIDE the container is tmpfs,
# i.e. RAM (1008 GB cap on a 1.8 TB node): fast, but do not write hundreds of GB of scratch there.
export TMPDIR=${TMPDIR_OVERRIDE:-/tmp/$USER}; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$TEAM/pip-cache}   # keep pip wheels off the 50 GB home quota
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
PORT=${PORT:-8150}     # NOT 8000: nodes are shared and other teams run vLLM on 8000 (SO_REUSEPORT lets two servers share it and split traffic)
MAX_LEN=${MAX_LEN:-32768}     # card says 262144; prompt upsampling needs a few KB, keep KV cache small and startup quick

srun --gres=gpu:1 --time=${TIME:-04:00:00} --job-name=nemotron --pty \
  --container-name=ubr \
  --container-mounts=$HOME:$HOME,$TEAM:$TEAM \
  --container-workdir=$WORK \
  bash -c '
set -e
export HF_HOME='"$WORK"'/hf-cache HF_HUB_OFFLINE=1 PATH=$HOME/.local/bin:$PATH TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
echo "== node $(hostname)"; nvidia-smi -L
SNAP=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(\"'"$MODEL"'\"))")
echo "== snapshot $SNAP"
exec vllm serve "'"$MODEL"'" \
  --served-model-name nemotron \
  --max-num-seqs 8 --tensor-parallel-size 1 --max-model-len '"$MAX_LEN"' \
  --host 0.0.0.0 --port '"$PORT"' --trust-remote-code \
  --reasoning-parser-plugin "$SNAP/nano_v3_reasoning_parser.py" --reasoning-parser nano_v3'
