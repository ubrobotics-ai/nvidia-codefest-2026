#!/bin/bash
G=/storage/hackathon_teams/omc-team15/codefest/g1_cosmos; T=/storage/hackathon_teams/omc-team15
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 --gres=gpu:4 -c 64 --mem=800G --time=40:00:00 --job-name=g1_concat_v4 --exclude=dgx06 --container-name=pt_g1 --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PATH=\$PWD/.venv/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  export G1_HUG_ROOT=$G/g1_data_v4/g1_hug_v4 G1_HUG_STATS=$G/g1_data_v4/g1_hug_v4/action_stats.json
  export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$G/wan_vae/Wan2.2_VAE.pth
  export OUTPUT_ROOT=$G/runs/g1-edge-concat-v4 IMAGINAIRE_OUTPUT_ROOT=$G/runs/g1-edge-concat-v4 MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
  nvidia-smi --query-gpu=name --format=csv,noheader | head -1
  TOML_FILE=examples/toml/sft_config/action_policy_g1_hug_edge_v1_4gpu.toml NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_g1_hug_edge_smoke.sh" > $G/concat_v4.log 2>&1
echo "== g1_concat_v4 state=$(sacct -n -X --name=g1_concat_v4 -o State -P | tail -1)" >> $G/concat_v4.log
