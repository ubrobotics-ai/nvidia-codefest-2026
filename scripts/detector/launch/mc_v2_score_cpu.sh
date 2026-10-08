#!/bin/bash
# CPU-only scoring (the 4-GPU per-user quota is held by the G1 v3fix run): v2 on the whole v2 valid split,
# v1 on the warehouse box renders only (v1 was never run on them), 8 shards x 12 threads.
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass; PY=$T/venvs/rfdetr/bin/python
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 -c 100 --mem=400G --time=8:00:00 --job-name=mc_score_cpu --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache CUDA_VISIBLE_DEVICES='' THREADS=12 OMP_NUM_THREADS=12; cd $M
  for s in 0 1 2 3 4 5; do $PY infer_mc_any.py runs/mc_v2/last_ema.pth v2 $M/data_v2 \$s 6 > preds/infer_v2.s\$s.log 2>&1 & done
  for s in 0 1; do $PY infer_mc_any.py runs/mc_v1/last_ema.pth v1wbox $M/data_v2 \$s 2 wbox_ > preds/infer_v1wbox.s\$s.log 2>&1 & done
  wait"
cat $M/preds/mc_v2.s[0-5].jsonl > $M/preds/mc_v2.jsonl; cat $M/preds/mc_v1wbox.s[0-1].jsonl > $M/preds/mc_v1wbox.jsonl
wc -l $M/preds/mc_v2.jsonl $M/preds/mc_v1wbox.jsonl; grep -h "done\|Traceback\|Error" $M/preds/infer_v2.s*.log $M/preds/infer_v1wbox.s*.log | sort | uniq -c | head; echo SCOREINFERDONE
