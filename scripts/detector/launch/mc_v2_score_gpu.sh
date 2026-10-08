#!/bin/bash
# GPU scoring: v2 on the whole v2 valid split (3 shards, GPUs 0-2), v1 on the warehouse box renders only (GPU 3).
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass; PY=$T/venvs/rfdetr/bin/python
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 --gres=gpu:4 -c 48 --mem=300G --time=2:00:00 --job-name=mc_score --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache THREADS=8; cd $M
  for s in 0 1 2; do CUDA_VISIBLE_DEVICES=\$s $PY infer_mc_any.py runs/mc_v2/last_ema.pth v2 $M/data_v2 \$s 3 > preds/infer_v2.s\$s.log 2>&1 & done
  CUDA_VISIBLE_DEVICES=3 $PY infer_mc_any.py runs/mc_v1/last_ema.pth v1wbox $M/data_v2 0 1 wbox_ > preds/infer_v1wbox.s0.log 2>&1 &
  wait"
cat $M/preds/mc_v2.s[0-2].jsonl > $M/preds/mc_v2.jsonl; cp $M/preds/mc_v1wbox.s0.jsonl $M/preds/mc_v1wbox.jsonl
wc -l $M/preds/mc_v2.jsonl $M/preds/mc_v1wbox.jsonl; grep -h "done\|Traceback\|Error" $M/preds/infer_v2.s*.log $M/preds/infer_v1wbox.s*.log | sort | uniq -c | head; echo SCOREINFERDONE
