#!/bin/bash
T=/storage/hackathon_teams/omc-team15
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
mkdir -p $T/tmp/xfer/b2 $T/tmp/xfer/b2log
srun -N1 --gres=gpu:8 --mem=1500G --cpus-per-task=96 --time=4:00:00 --job-name=xfer_b2 \
  --container-name=ubr --container-mounts=$HOME:$HOME,$T:$T --container-workdir=$T/codefest/repo \
  bash -c "for w in 0 1 2 3 4 5 6 7; do bash $T/tmp/xfer_b2_worker.sh \$w 8 & done; wait" > $T/tmp/xfer/xfer_b2.log 2>&1
echo "== xfer_b2 state=$(sacct -n -X --name=xfer_b2 -o State -P | tail -1) mp4=$(find $T/tmp/xfer/b2 -name '*.mp4' | wc -l)/81"
