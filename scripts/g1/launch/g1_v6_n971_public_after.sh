#!/bin/bash
# Filipe 2026-10-09: v6-n971 public. Waits for the v6 chain to log the N=971 checkpoint, then publishes it with
# upload_g1_v6_public.py (weights only, redacted config, NOT YET EVALUATED card).
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; LOG=$G/curve_v6/driver.log
until grep -q "N=971 CHECKPOINT\|N=971 FAILED" $LOG; do sleep 120; done
grep -q "N=971 FAILED" $LOG && { echo "N=971 FAILED, nothing to publish" >> $LOG; exit 1; }
C=$G/runs/g1-v6-n971/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu/checkpoints/iter_000000500
sleep 60
HF_HOME=$T/codefest/hf-cache $T/tools/hfvenv/bin/python $G/upload_g1_v6_public.py $C ubr-physical-ai/g1-hug-edge-concat-v6-n971 971 2>&1 | grep -E "UPLOADED|Error|Traceback" >> $LOG
