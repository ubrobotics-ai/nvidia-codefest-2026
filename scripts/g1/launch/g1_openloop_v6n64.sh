#!/bin/bash
# Open-loop score of G1 v6-n64 (iter 500) on the 51 g1_hug_v6 val episodes (Filipe 2026-10-09: does it reproduce wrist height per box height?): NVIDIA's policy server (+ the box-lifting
# session's viewpoint patch) on localhost, then openloop_eval.py against it. One GPU.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos
R=$G/runs/g1-v6-n64/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
mkdir -p $G/openloop_v6n64
srun -N1 --gres=gpu:1 -c 32 --mem=200G --time=3:00:00 --job-name=g1_openloop_v6n64 --container-name=pt_g1 --exclude=dgx03,dgx06,dgx07,dgx08,dgx11 --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  export G1_HUG_ROOT=$G/g1_data_v6/g1_hug_v6 G1_HUG_STATS=$G/g1_data_v6/g1_hug_v6/action_stats.json WAN_VAE_PATH=$G/wan_vae/Wan2.2_VAE.pth
  PORT=\$((20000 + SLURM_JOB_ID % 20000))
  .venv/bin/python -m cosmos_framework.scripts.action_policy_server_libero --checkpoint-path $R/checkpoints/iter_000000500 \
     --config-file $R/config.yaml --action-stats-path $G/g1_data_v6/g1_hug_v6/action_stats.json --fps 10 --num-steps 30 --port \$PORT \
     > $G/openloop_v6n64/server.log 2>&1 &
  S=\$!
  for i in \$(seq 360); do grep -q 'Server accessible at' $G/openloop_v6n64/server.log && break; kill -0 \$S 2>/dev/null || { echo 'server died'; tail -30 $G/openloop_v6n64/server.log; exit 1; }; sleep 10; done
  echo \"server up after \$((i*10)) s\"
  .venv/bin/python $G/openloop_eval.py http://127.0.0.1:\$PORT $G/g1_data_v6/g1_hug_v6 $G/openloop_v6n64/chunks.json 16
  kill \$S" > $G/openloop_v6n64/run.log 2>&1
echo OPENLOOPJOBDONE >> $G/openloop_v6n64/run.log
