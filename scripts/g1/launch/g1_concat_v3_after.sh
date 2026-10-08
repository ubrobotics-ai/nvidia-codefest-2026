#!/bin/bash
# Start the v3 concat run when job 9084 (ego v1) has ended with iter_000002000 saved (Filipe approved 2026-10-03).
C=/storage/hackathon_teams/omc-team15/codefest/g1_cosmos/runs/g1-edge-ego-v1/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_ego_v1_4gpu/checkpoints
while squeue -h -j 9084 2>/dev/null | grep -q .; do sleep 60; done
[ -d $C/iter_000002000/model ] || { echo "9084 ended without iter_000002000; NOT starting v3"; exit 1; }
echo "starting v3 at $(date -u +%T) UTC"
/storage/hackathon_teams/omc-team15/tmp/g1_concat_v3_run.sh
