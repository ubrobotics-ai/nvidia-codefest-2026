#!/bin/bash
M=/storage/hackathon_teams/omc-team15/codefest/multiclass; T=/storage/hackathon_teams/omc-team15
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 --gres=gpu:4 --mem=200G -c 32 --time=2:00:00 --job-name=mc_infer --exclude=dgx06 --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache; cd $M
  for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$s $T/venvs/rfdetr/bin/python infer_mc.py runs/mc_v1/last_ema.pth v1 \$s 4 > preds/infer_v1.s\$s.log 2>&1 & done; wait"
cat $M/preds/mc_v1.s[0-3].jsonl > $M/preds/mc_v1.jsonl; wc -l $M/preds/mc_v1.jsonl; grep -h "done\|Traceback\|Error" $M/preds/infer_v1.s*.log | head -5; echo MCINFERDONE
