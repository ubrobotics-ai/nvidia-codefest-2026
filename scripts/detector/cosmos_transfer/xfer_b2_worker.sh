#!/bin/bash
# Runs INSIDE the container. Worker $1 of $2 on GPU $1: translates its share of prompts_b2.json,
# one clip per invocation (each clip has its own prompt), skipping clips that already have an mp4.
W=$1; N=$2; T=/storage/hackathon_teams/omc-team15; D=$T/tmp/xfer
SNAP=$T/codefest/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd/clips
export CUDA_VISIBLE_DEVICES=$W HF_HOME=$T/codefest/hf-cache HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $T/codefest/repo
python3 -c "import json;[print(k) for i,k in enumerate(sorted(json.load(open('$D/prompts_b2.json')))) if i % $N == $W]" | while read cid; do
  out=$D/b2/$cid
  [ -n "$(find $out -name '*.mp4' 2>/dev/null | head -1)" ] && { echo "w$W $cid skip"; continue; }
  p=$(python3 -c "import json;print(json.load(open('$D/prompts_b2.json'))['$cid'])")
  mkdir -p $D/one_b2/$cid; ln -sfn $SNAP/$cid $D/one_b2/$cid/$cid
  for a in 1 2 3; do
    t0=$(date +%s)
    python scripts/cosmos_transfer_batch.py --clips $D/one_b2/$cid --out $out --control seg --frames 93 \
      --height 720 --width 1280 --steps 36 --strengths 0.7 --prompt "$p" > $D/b2log/${cid}_a$a.log 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -n "$(find $out -name '*.mp4' 2>/dev/null | head -1)" ]; then
      echo "w$W $cid OK a$a $(( $(date +%s) - t0 ))s"; break; fi
    echo "w$W $cid FAIL a$a rc=$rc"
  done
done
echo "w$W finished"
