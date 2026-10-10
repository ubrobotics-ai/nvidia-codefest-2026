#!/bin/bash
# Filipe 2026-10-09: v5 N = 64 at 2,000 iterations public. Waits for job 9810 to end, then publishes iter_000002000.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; O=$G/runs/g1-v5-n64-2k
C=$O/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu/checkpoints/iter_000002000
while squeue -h -u $USER -n g1v5_n64_2k 2>/dev/null | grep -q . || ! squeue -h -u $USER >/dev/null 2>&1; do sleep 120; done
sleep 120
[ -d $C/model ] || { echo "NO iter_000002000 $(date -u +%FT%TZ)" >> $O/upload.log; exit 1; }
[ -f $C/episodes.json ] || cp $O/episodes.json $C/episodes.json
HF_HOME=$T/codefest/hf-cache $T/tools/hfvenv/bin/python $G/upload_g1_v5_n64_2k_public.py $C ubr-physical-ai/g1-hug-edge-concat-v5-n64-2k 64 2>&1 | grep -E "UPLOADED|Error|Traceback" >> $O/upload.log
