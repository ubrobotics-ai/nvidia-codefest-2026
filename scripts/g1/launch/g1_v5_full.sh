#!/bin/bash
# Full v5 run (Filipe: "when v2b finish launch v5 in those GPUs"): waits for the v2b driver to end, then the unchanged
# 2000-iteration Edge recipe on the v5 train split.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; R=$G/g1_data_v5/g1_hug_v5; O=$G/runs/g1-edge-concat-v5
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
until grep -q "MCV2BDONE\|== run 3 ended" $T/tmp/xfer/mc_v2b.log; do sleep 60; done
sleep 60; mkdir -p $O
srun -N1 --gres=gpu:4 -c 64 --mem=800G --time=40:00:00 --job-name=g1_concat_v5 --exclude=dgx06,dgx11 --container-name=pt_g1 --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PATH=\$PWD/.venv/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  export G1_HUG_ROOT=$R G1_HUG_STATS=$R/action_stats.json
  export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$G/wan_vae/Wan2.2_VAE.pth
  export OUTPUT_ROOT=$O IMAGINAIRE_OUTPUT_ROOT=$O MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
  TOML_FILE=examples/toml/sft_config/action_policy_g1_hug_edge_v1_4gpu.toml NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_g1_hug_edge_smoke.sh" > $G/concat_v5.log 2>&1
echo "== g1_concat_v5 state=$(sacct -n -X --name=g1_concat_v5 -o State -P | tail -1)" >> $G/concat_v5.log
