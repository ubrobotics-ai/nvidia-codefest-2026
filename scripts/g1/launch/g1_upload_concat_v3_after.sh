#!/bin/bash
G=/storage/hackathon_teams/omc-team15/codefest/g1_cosmos
C=$G/runs/g1-edge-concat-v3/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu/checkpoints
until squeue -h -u $USER -n g1_concat_v3 | grep -q .; do sleep 120; grep -q "NOT starting" /storage/hackathon_teams/omc-team15/tmp/xfer/g1_concat_v3_after.out 2>/dev/null && exit 1; done
J=$(squeue -h -u $USER -n g1_concat_v3 -o %i | head -1); echo "watching job $J"
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 120; done
echo "job $J: $(sacct -j $J -X -o State -n)"; ls $C
[ -d $C/iter_000002000/model ] || { echo "NO iter_000002000, not uploading"; exit 1; }
sleep 60
HF_HOME=/storage/hackathon_teams/omc-team15/codefest/hf-cache /storage/hackathon_teams/omc-team15/tools/hfvenv/bin/python $G/upload_g1_concat_v3.py 2>&1 | grep -vE "Warning|%\|" | tail -5
echo UPLOADDONE
