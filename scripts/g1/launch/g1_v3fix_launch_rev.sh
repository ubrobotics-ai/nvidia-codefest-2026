#!/bin/bash
# Filipe: "Launch" (2026-10-03 20:18 UTC). Waits for ubr-physical-ai/g1_hug_v3 to finish uploading, downloads that
# revision, runs the loader check, and launches the v3fix concat run only if the check matches the expected layout.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; PY=$T/tools/hfvenv/bin/python
export HF_HOME=$T/codefest/hf-cache ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=/tmp/enroot-$USER/run
cur="82f256113f7cf5adfe07aee6baa08c0426db2827"   # revision sent by the box-lifting session, 2026-10-03
REV=${cur%% *}; echo "using revision $REV"
$PY -c "
from huggingface_hub import snapshot_download
print(snapshot_download('ubr-physical-ai/g1_hug_v3', repo_type='dataset', revision='$REV', local_dir='$G/g1_data_v3fix', max_workers=8))" 2>&1 | tail -1
echo "$REV" > $G/g1_data_v3fix/REVISION
srun -N1 -c 16 --mem=128G --time=0:30:00 --job-name=g1v3fix_loader --container-name=pt_build --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  .venv/bin/python -c \"
from cosmos_framework.data.generator.action.datasets.g1_hug_lerobot_dataset import G1HugLeRobotDataset
d = G1HugLeRobotDataset(root='$G/g1_data_v3fix/g1_hug_v3', viewpoint='concat_view', action_stats_path='$G/g1_data_v3fix/g1_hug_v3/action_stats.json')
s = d[0]; print('LOADER', len(d), tuple(s['video'].shape), tuple(s['action'].shape), s['domain_id'].item())\" 2>&1 | grep -E 'LOADER|Loaded G1|Error|Traceback' | tail -4" 2>&1 | grep -v "^pyxis" | tee $G/g1_data_v3fix/loader_check.txt
L=$(grep "^LOADER" $G/g1_data_v3fix/loader_check.txt)
K=$(grep -oE "kept_episodes=[0-9]+" $G/g1_data_v3fix/loader_check.txt | cut -d= -f2)
if [[ "$L" == *"(3, 17, 540, 640) (16, 29) 15"* && -n "$K" && "$K" -ge 900 && "$K" -le 998 ]]; then
  echo "loader check OK (kept $K): launching"
  $T/tmp/g1_concat_v3fix_run.sh &
  sleep 30; nohup $T/tmp/g1_upload_concat_v3fix_after.sh > $T/tmp/xfer/g1_upload_concat_v3fix.out 2>&1 &
  wait
else
  echo "loader check MISMATCH, NOT launching: $L kept=$K"
fi
