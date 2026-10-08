#!/bin/bash
# Publishes each v5 checkpoint once, as it lands: the three curve points (from curve_v5/driver.log) and the full run.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; PY=$T/tools/hfvenv/bin/python; L=$G/curve_v5/upload.log
export HF_HOME=$T/codefest/hf-cache
FULL=$G/runs/g1-edge-concat-v5/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu/checkpoints/iter_000002000
done_list=" "
while true; do
  while read -r _ nn _ dir _; do
    N=${nn#N=}; [[ "$done_list" == *" $N "* ]] && continue
    NAME=$N; [ $N = all ] && NAME=965
    echo "$(date -u +%FT%TZ) uploading N=$NAME from $dir" >> $L
    $PY $G/upload_g1_v5.py $dir ubr-physical-ai/g1-hug-edge-concat-v5-n$NAME $NAME 500 1 $dir/episodes.json 2>&1 | grep -E "UPLOADED|failed|Error|Traceback" >> $L
    done_list="$done_list$N "
  done < <(grep " CHECKPOINT " $G/curve_v5/driver.log 2>/dev/null)
  if [[ "$done_list" != *" full "* ]] && [ -d $FULL/model ] && grep -q "g1_concat_v5 state=" $G/concat_v5.log 2>/dev/null; then
    echo "$(date -u +%FT%TZ) uploading full run" >> $L
    $PY $G/upload_g1_v5.py $FULL ubr-physical-ai/g1-hug-edge-concat-v5-full 965 2000 0 $G/curve_v5/episodes_all.json 2>&1 | grep -E "UPLOADED|failed|Error|Traceback" >> $L
    done_list="${done_list}full "
  fi
  [[ "$done_list" == *" all "*" 64 "*" 256 "*" full "* ]] && { echo ALLUPLOADED >> $L; exit 0; }
  sleep 300
done
