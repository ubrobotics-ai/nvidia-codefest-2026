#!/bin/bash
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass; PY=$T/venvs/rfdetr/bin/python
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 --gres=gpu:3 -c 36 --mem=200G --time=2:00:00 --job-name=v2b_pseudo --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache; cd $M
  for s in 0 1 2; do CUDA_VISIBLE_DEVICES=\$s $PY pseudolabel_boxes.py \$s 3 > pseudo/log.s\$s 2>&1 & done; wait"
cat $M/pseudo/boxes.s[0-2].jsonl > $M/pseudo/boxes.jsonl; wc -l $M/pseudo/boxes.jsonl; grep -h "done\|Traceback" $M/pseudo/log.s*; echo PSEUDODONE
