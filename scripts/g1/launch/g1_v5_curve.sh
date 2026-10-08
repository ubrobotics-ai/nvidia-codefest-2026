#!/bin/bash
# v5 data-efficiency curve (Filipe, 2026-10-05): 500-iteration runs of the Edge recipe on nested subsets, in the
# order N=all, 64, 256, one at a time on 4 GPUs. Each run's episode list is copied next to its checkpoint.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; R=$G/g1_data_v5/g1_hug_v5
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
for N in all 64 256; do
  O=$G/runs/g1-v5-curve-n$N; mkdir -p $O; cp $G/curve_v5/episodes_$N.json $O/episodes.json
  SUB=$G/curve_v5/episodes_$N.json; [ $N = all ] && SUB=""
  echo "== N=$N start $(date -u +%FT%TZ)" >> $G/curve_v5/driver.log
  srun -N1 --gres=gpu:4 -c 64 --mem=800G --time=12:00:00 --job-name=g1v5_curve_n$N --exclude=dgx06,dgx11 --container-name=pt_g1b --container-mounts=$HOME:$HOME,$T:$T \
    bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
    export PATH=\$PWD/.venv/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache PYTHONUNBUFFERED=1
    export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
    export G1_HUG_ROOT=$R G1_HUG_STATS=$R/action_stats.json G1_HUG_EPISODES=$SUB
    export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$G/wan_vae/Wan2.2_VAE.pth
    export OUTPUT_ROOT=$O IMAGINAIRE_OUTPUT_ROOT=$O MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
    TOML_FILE=examples/toml/sft_config/action_policy_g1_hug_edge_curve500_4gpu.toml NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_g1_hug_edge_smoke.sh" > $O/train.log 2>&1
  C=$(ls -d $O/cosmos3_action_g1/action_sft/*/checkpoints/iter_000000500 2>/dev/null)
  if [ -n "$C" ]; then cp $O/episodes.json $C/episodes.json; echo "== N=$N CHECKPOINT $C $(date -u +%FT%TZ)" >> $G/curve_v5/driver.log
  else echo "== N=$N FAILED (no iter_000000500) $(date -u +%FT%TZ)" >> $G/curve_v5/driver.log; fi
done
echo CURVEDONE >> $G/curve_v5/driver.log
