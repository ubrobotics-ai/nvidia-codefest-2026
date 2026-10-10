#!/bin/bash
# Runs the v6-n64 open-loop test when job 9810 (v5 N = 64, 2,000 it) ends and frees its GPUs.
T=/storage/hackathon_teams/omc-team15
while squeue -h -u $USER -n g1v5_n64_2k 2>/dev/null | grep -q . || ! squeue -h -u $USER >/dev/null 2>&1; do sleep 120; done
sleep 60; exec $T/tmp/g1_openloop_v6n64.sh
