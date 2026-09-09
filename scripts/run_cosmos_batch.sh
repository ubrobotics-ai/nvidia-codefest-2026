#!/bin/bash
# Run cosmos_i2v_batch.py on one B300 from the `ubr` container. Run from the LOGIN node inside tmux.
#   FRAMES=... OUT=... N=20 NEMOTRON_URL=http://dgx01:8000/v1 ./scripts/run_cosmos_batch.sh [extra cosmos_i2v_batch.py args]
# Datasets live on team storage, not in the repo and not in $HOME.
set -euo pipefail
export TEAM=${TEAM:-/storage/hackathon_teams/omc-team15}
export WORK=${WORK:-$TEAM/codefest}
# srun propagates XDG_RUNTIME_DIR; the default /run/user/<uid> is unwritable on the DGX nodes and pyxis dies in task_init().
export ENROOT_RUNTIME_PATH=${ENROOT_RUNTIME_PATH:-/tmp/enroot-$USER/run}
export XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
# Scratch: node-local /tmp is 1.6 TB free at 1.9 GB/s; team storage is NFS from dgx02 at 871 MB/s, and CUDA JIT
# plus enroot unpacking are latency-sensitive. (.bashrc sets TMPDIR for interactive shells only -- it returns early
# when non-interactive, which is why every script must set this itself.) Note /tmp INSIDE the container is tmpfs,
# i.e. RAM (1008 GB cap on a 1.8 TB node): fast, but do not write hundreds of GB of scratch there.
export TMPDIR=${TMPDIR_OVERRIDE:-/tmp/$USER}; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$TEAM/pip-cache}   # keep pip wheels off the 50 GB home quota
FRAMES=${FRAMES:-$WORK/repo/data/real_frames}
OUT=${OUT:-$WORK/repo/data/synthetic}
N=${N:-20}
export NEMOTRON_URL=${NEMOTRON_URL:-}   # empty -> cosmos_i2v_batch.py uses its fixed fallback prompt
REPO=$(cd "$(dirname "$0")/.." && pwd)

srun --gres=gpu:1 --time=${TIME:-03:00:00} --job-name=cosmos --pty \
  --container-name=ubr \
  --container-mounts=$HOME:$HOME,$TEAM:$TEAM \
  --container-workdir=$REPO \
  --container-env=NEMOTRON_URL \
  bash -c '
set -e
export HF_HOME='"$WORK"'/hf-cache HF_HUB_OFFLINE=1 PATH=$HOME/.local/bin:$PATH NLTK_DATA=$HOME/nltk_data TMPDIR=/tmp/$USER   # real-file copy, see cluster_build_container.sh
mkdir -p $TMPDIR
echo "== node $(hostname)"; nvidia-smi -L
python scripts/cosmos_i2v_batch.py --frames '"$FRAMES"' --out '"$OUT"' --n '"$N"' '"$*"'
echo "== timing"; tail -3 '"$OUT"'/timing.csv'
