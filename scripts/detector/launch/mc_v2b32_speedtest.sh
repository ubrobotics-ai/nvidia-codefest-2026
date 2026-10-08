#!/bin/bash
# 20-minute speed test of v2b-32 on the 4 idle GPUs (pt_rfinf container, not v2b's): steps/s from metrics.csv.
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
rm -rf $M/runs/v2b32_speedtest
srun -N1 --gres=gpu:4 --mem=600G --cpus-per-task=64 --time=0:22:00 --job-name=v2b32_speed --exclude=dgx06 \
  --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache MP=\$((20000 + SLURM_JOB_ID % 20000)); cd $M && $T/venvs/rfdetr/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=\$MP train_mc_v2b32.py $M/runs/v2b32_speedtest 1" \
  > $T/tmp/xfer/v2b32_speedtest.log 2>&1
echo SPEEDTESTDONE >> $T/tmp/xfer/v2b32_speedtest.log
