#!/bin/bash
# v2b training driver: up to 3 x 12 h runs, each resuming from the newest checkpoint, until last_ema.pth after epoch 12.
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
for run in 1 2 3; do
  srun -N1 --gres=gpu:4 --mem=600G --cpus-per-task=64 --time=12:00:00 --job-name=mc_v2b --exclude=dgx06 \
    --container-image='nvcr.io#nvidia/pytorch:25.08-py3' --container-name=pt_rfv2b \
    --container-mounts=$HOME:$HOME,$T:$T \
    bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache MP=\$((20000 + SLURM_JOB_ID % 20000)); cd $M && $T/venvs/rfdetr/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=\$MP train_mc_v2b.py" \
    >> $T/tmp/xfer/mc_v2b.log 2>&1
  echo "== run $run ended $(date -u +%FT%TZ)" >> $T/tmp/xfer/mc_v2b.log
  ls $M/runs/mc_v2b/checkpoint_11.ckpt >/dev/null 2>&1 && { echo "MCV2BDONE" >> $T/tmp/xfer/mc_v2b.log; exit 0; }
done
