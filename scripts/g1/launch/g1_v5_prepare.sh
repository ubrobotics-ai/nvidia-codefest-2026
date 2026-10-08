#!/bin/bash
# Download ubr-physical-ai/g1_hug_v5 at revision $1, run the loader check (train split and the 64-episode subset),
# and build the curve subsets. Prints PREPARE OK / PREPARE FAILED.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; PY=$T/tools/hfvenv/bin/python; REV=$1
export HF_HOME=$T/codefest/hf-cache ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=/tmp/enroot-$USER/run
D=$G/g1_data_v5; R=$D/g1_hug_v5
$PY -c "
from huggingface_hub import snapshot_download
print(snapshot_download('ubr-physical-ai/g1_hug_v5', repo_type='dataset', revision='$REV', local_dir='$D', max_workers=8))" 2>&1 | tail -1
echo "$REV" > $D/REVISION
mkdir -p $G/curve_v5 && python3 $G/make_curve_subsets.py $R $G/curve_v5 || { echo "PREPARE FAILED (subsets)"; exit 1; }
srun -N1 -c 16 --mem=128G --time=0:30:00 --exclude=dgx06,dgx11 --job-name=g1v5_loader --container-name=pt_build --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  for SUB in '' $G/curve_v5/episodes_64.json; do G1_HUG_EPISODES=\$SUB .venv/bin/python -c \"
from cosmos_framework.data.generator.action.datasets.g1_hug_lerobot_dataset import G1HugLeRobotDataset
d = G1HugLeRobotDataset(root='$R', viewpoint='concat_view', action_stats_path='$R/action_stats.json')
s = d[0]; print('LOADER', len(d), tuple(s['video'].shape), tuple(s['action'].shape), s['domain_id'].item())\" 2>&1 | grep -E 'LOADER|Loaded G1|Error|Traceback' | tail -3; done" 2>&1 | grep -v "^pyxis" | tee $D/loader_check.txt
n=$(grep -c "(16, 29) 15" $D/loader_check.txt)
[ "$n" = 2 ] && grep -q "kept_episodes=64/" $D/loader_check.txt && echo "PREPARE OK" || echo "PREPARE FAILED (loader)"
