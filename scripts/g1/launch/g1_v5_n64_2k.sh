#!/bin/bash
# Curve 2, second budget (lw-lab1 PROTOCOL.md addendum, ubr-isaac codefest-curves 3429870; Filipe "yes" 2026-10-07):
# N = 64 (curve_v5/episodes_64.json) at 2,000 iterations with job 9674's recipe (v5full TOML: LR cycle 2000, warm-up 100,
# gbs 2048, save every 250), full-v5 action_stats. Publishes iter 2000 to a public repo when done.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; R=$G/g1_data_v5/g1_hug_v5; O=$G/runs/g1-v5-n64-2k
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
mkdir -p $O; cp $G/curve_v5/episodes_64.json $O/episodes.json
srun -N1 --gres=gpu:4 -c 64 --mem=800G --time=48:00:00 --job-name=g1v5_n64_2k --exclude=dgx03,dgx06,dgx07,dgx08,dgx11 --container-name=pt_g1b --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PATH=\$PWD/.venv/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  export G1_HUG_ROOT=$R G1_HUG_STATS=$R/action_stats.json G1_HUG_EPISODES=$G/curve_v5/episodes_64.json
  export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$G/wan_vae/Wan2.2_VAE.pth
  export OUTPUT_ROOT=$O IMAGINAIRE_OUTPUT_ROOT=$O MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
  TOML_FILE=examples/toml/sft_config/action_policy_g1_hug_edge_v5full_4gpu.toml NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_g1_hug_edge_smoke.sh" > $O/train.log 2>&1
C=$O/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu/checkpoints/iter_000002000
if [ -d $C/model ]; then
  cp $O/episodes.json $C/episodes.json
  HF_HOME=$T/codefest/hf-cache $T/tools/hfvenv/bin/python $G/upload_g1_v5.py $C ubr-physical-ai/g1-hug-edge-concat-v5-n64-2k 64 2000 0 $C/episodes.json 2>&1 | grep -E "UPLOADED|failed|Error|Traceback" >> $O/upload.log
else echo "NO iter_000002000 $(date -u +%FT%TZ)" >> $O/upload.log; fi
