#!/bin/bash
# GPU scoring of v2b, plus a same-conditions re-score of v2 (tag v2r), as overlap steps inside the G1 v5 full-run job
# (Filipe: "Yes", 2026-10-05; that run uses ~45 of 275 GB per GPU). 2 shards each, one GPU per shard.
T=/storage/hackathon_teams/omc-team15; M=$T/codefest/multiclass; PY=$T/venvs/rfdetr/bin/python
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
J=$(squeue -h -u $USER -n g1_concat_v5 -o %i | head -1)
srun --jobid=$J --overlap -N1 --container-name=pt_rfinf --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache THREADS=8; cd $M
  for s in 0 1; do CUDA_VISIBLE_DEVICES=\$s $PY infer_mc_any.py runs/mc_v2b/last_ema.pth v2b $M/data_v2 \$s 2 > preds/infer_v2b.s\$s.log 2>&1 & done
  for s in 0 1; do CUDA_VISIBLE_DEVICES=\$((s + 2)) $PY infer_mc_any.py runs/mc_v2/last_ema.pth v2r $M/data_v2 \$s 2 > preds/infer_v2r.s\$s.log 2>&1 & done
  wait"
cat $M/preds/mc_v2b.s[0-1].jsonl > $M/preds/mc_v2b.jsonl; cat $M/preds/mc_v2r.s[0-1].jsonl > $M/preds/mc_v2r.jsonl
wc -l $M/preds/mc_v2b.jsonl $M/preds/mc_v2r.jsonl; grep -h "done\|Traceback\|Error" $M/preds/infer_v2b.s*.log $M/preds/infer_v2r.s*.log | sort | uniq -c; echo SCOREINFERDONE
