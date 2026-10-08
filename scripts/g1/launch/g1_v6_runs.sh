#!/bin/bash
# v6 (warehouse) runs, Filipe option A (2026-10-08): when job 9674 (v5 full) ends and frees its 4 GPUs, train
# N=64 then N=971 (all train) on ubr-physical-ai/g1_hug_v6 @ c3da28de, each 500 iterations with the curve recipe
# (action_policy_g1_hug_edge_curve500_4gpu.toml: LR cycle 500, warm-up 100, gbs 2048, saves at 250/500), v6 action_stats.
# Each lands as a PRIVATE repo ubr-physical-ai/g1-hug-edge-concat-v6-n{64,971}.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; R=$G/g1_data_v6/g1_hug_v6; LOG=$G/curve_v6/driver.log
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
until grep -q "g1_concat_v5 state=" $G/concat_v5.log 2>/dev/null; do sleep 60; done
sleep 60
for N in 64 all; do
  NAME=$N; [ $N = all ] && NAME=971
  O=$G/runs/g1-v6-n$NAME; mkdir -p $O; cp $G/curve_v6/episodes_$N.json $O/episodes.json
  SUB=$G/curve_v6/episodes_$N.json; [ $N = all ] && SUB=""
  echo "== N=$NAME start $(date -u +%FT%TZ)" >> $LOG
  srun -N1 --gres=gpu:4 -c 64 --mem=800G --time=14:00:00 --job-name=g1v6_n$NAME --exclude=dgx03,dgx06,dgx07,dgx08,dgx11 --container-name=pt_g1 --container-mounts=$HOME:$HOME,$T:$T \
    bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
    export PATH=\$PWD/.venv/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache PYTHONUNBUFFERED=1
    export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
    export G1_HUG_ROOT=$R G1_HUG_STATS=$R/action_stats.json G1_HUG_EPISODES=$SUB
    export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$G/wan_vae/Wan2.2_VAE.pth
    export OUTPUT_ROOT=$O IMAGINAIRE_OUTPUT_ROOT=$O MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
    TOML_FILE=examples/toml/sft_config/action_policy_g1_hug_edge_curve500_4gpu.toml NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_g1_hug_edge_smoke.sh" > $O/train.log 2>&1
  C=$O/cosmos3_action_g1/action_sft/action_policy_g1_hug_edge_v1_4gpu/checkpoints/iter_000000500
  if [ -d $C/model ]; then
    cp $O/episodes.json $C/episodes.json; echo "== N=$NAME CHECKPOINT $C $(date -u +%FT%TZ)" >> $LOG
    HF_HOME=$T/codefest/hf-cache $T/tools/hfvenv/bin/python $G/upload_g1_v6.py $C ubr-physical-ai/g1-hug-edge-concat-v6-n$NAME $NAME 500 $([ $NAME = 64 ] && echo 1 || echo 0) $C/episodes.json 2>&1 | grep -E "UPLOADED|failed|Error|Traceback|exists" >> $LOG
  else echo "== N=$NAME FAILED (no iter_000000500) $(date -u +%FT%TZ)" >> $LOG; fi
done
echo V6DONE >> $LOG
