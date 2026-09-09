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
export HF_HOME='"$WORK"'/hf-cache HF_HUB_OFFLINE=1 PATH=$HOME/.local/bin:$PATH NLTK_DATA=$HOME/nltk_data   # real-file copy, see cluster_build_container.sh
echo "== node $(hostname)"; nvidia-smi -L
python scripts/cosmos_i2v_batch.py --frames '"$FRAMES"' --out '"$OUT"' --n '"$N"' '"$*"'
echo "== timing"; tail -3 '"$OUT"'/timing.csv'
