#!/bin/bash
G=/storage/hackathon_teams/omc-team15/codefest/g1_cosmos
C=$G/runs/g1-edge-ego-v1/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_ego_v1_4gpu/checkpoints
while squeue -h -j 9084 2>/dev/null | grep -q .; do sleep 120; done
echo "job 9084: $(sacct -j 9084 -X -o State -n)"; ls $C
[ -d $C/iter_000002000/model ] || { echo "NO iter_000002000, not uploading"; exit 1; }
sleep 60   # let the final checkpoint write settle
HF_HOME=/storage/hackathon_teams/omc-team15/codefest/hf-cache /storage/hackathon_teams/omc-team15/tools/hfvenv/bin/python $G/upload_g1.py 2>&1 | grep -vE "Warning|%\|" | tail -5
echo UPLOADDONE
