#!/bin/bash
# CPU-only scoring of v2b on the v2 valid split (same split as v1 and v2; all 8 GPUs are on the G1 v5 runs), 6 shards x 12 threads.
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass; PY=$T/venvs/rfdetr/bin/python
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 -c 100 --mem=400G --time=8:00:00 --exclude=dgx06,dgx08,dgx11 --job-name=mc_score_v2b --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache CUDA_VISIBLE_DEVICES='' THREADS=12 OMP_NUM_THREADS=12; cd $M
  for s in 0 1 2 3 4 5; do $PY infer_mc_any.py runs/mc_v2b/last_ema.pth v2b $M/data_v2 \$s 6 > preds/infer_v2b.s\$s.log 2>&1 & done
  wait"
cat $M/preds/mc_v2b.s[0-5].jsonl > $M/preds/mc_v2b.jsonl
wc -l $M/preds/mc_v2b.jsonl; grep -h "done\|Traceback\|Error" $M/preds/infer_v2b.s*.log | sort | uniq -c | head; echo SCOREINFERDONE
