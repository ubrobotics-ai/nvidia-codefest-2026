#!/bin/bash
# Filipe 2026-10-06, option B: when the N=all curve rerun ends (~14:40 UTC), use its 4 GPUs first for a small
# Cosmos3-Super edge-transfer test (8 warehouse renders, the RTX Transfer-2.5 test set), at most 1.5 h, then the full v5 run.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; O=$G/super_transfer_test
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
until grep -q "== N=all rerun ended" $G/curve_v5/driver.log; do sleep 60; done
echo "super test start $(date -u +%FT%TZ)" > $O/run.log
srun -N1 --gres=gpu:4 -c 64 --mem=800G --time=1:30:00 --job-name=super_xfer_test --exclude=dgx06,dgx07,dgx08,dgx11 --container-name=pt_g1 --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PATH=\$PWD/.venv/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  t0=\$(date +%s)
  torchrun --nproc-per-node=4 --master-port=\$((20000 + SLURM_JOB_ID % 20000)) -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput --dp-shard-size=4 --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 \
    -i '$O/inputs/*.json' -o $O/outputs --checkpoint-path Cosmos3-Super --seed=1 --no-guardrails
  echo \"SUPER rc=\$? wall \$(( \$(date +%s) - t0 )) s\"" >> $O/run.log 2>&1
echo "super test end $(date -u +%FT%TZ)" >> $O/run.log
exec $T/tmp/g1_v5_full_relaunch.sh
