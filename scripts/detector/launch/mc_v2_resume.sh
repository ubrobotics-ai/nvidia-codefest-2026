#!/bin/bash
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH

srun -N1 --gres=gpu:4 --mem=600G --cpus-per-task=64 --time=12:00:00 --job-name=mc_v2r --exclude=dgx06 \
  --container-image='nvcr.io#nvidia/pytorch:25.08-py3' --container-name=pt_rfinf \
  --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache MP=\$((20000 + SLURM_JOB_ID % 20000)); cd $M && $T/venvs/rfdetr/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=\$MP train_mc_v2_resume.py" \
  > $T/tmp/xfer/mc_v2_resume.log 2>&1
echo "== mc_v2r state=$(sacct -n -X --name=mc_v2r -o State -P | tail -1)"
