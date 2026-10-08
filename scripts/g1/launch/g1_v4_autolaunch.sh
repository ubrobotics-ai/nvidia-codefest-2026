#!/bin/bash
# Filipe: "Launch" (2026-10-03 20:18 UTC). Waits for ubr-physical-ai/g1_hug_v4 to finish uploading, downloads that
# revision, runs the loader check, and launches the v4 concat run only if the check matches the expected layout.
T=/storage/hackathon_teams/omc-team15; G=$T/codefest/g1_cosmos; PY=$T/tools/hfvenv/bin/python
export HF_HOME=$T/codefest/hf-cache ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=/tmp/enroot-$USER/run
prev=""; stable=0
while true; do
  cur=$($PY -c "
from huggingface_hub import HfApi
try:
  i=HfApi().dataset_info('ubr-physical-ai/g1_hug_v4'); fs=sorted(s.rfilename for s in i.siblings)
  ok=any(f.endswith('g1_hug_v4/action_stats.json') for f in fs) and any(f.endswith('g1_hug_v4/meta/info.json') for f in fs)
  print(i.sha if ok else 'incomplete', len(fs))
except Exception: print('missing 0')" 2>/dev/null)
  if [[ "$cur" != missing* && "$cur" != incomplete* && "$cur" == "$prev" ]]; then stable=$((stable+1)); else stable=0; fi
  prev="$cur"; echo "$(date -u +%T) $cur stable=$stable"
  [ $stable -ge 2 ] && break
  sleep 300
done
REV=${cur%% *}; echo "using revision $REV"
$PY -c "
from huggingface_hub import snapshot_download
print(snapshot_download('ubr-physical-ai/g1_hug_v4', repo_type='dataset', revision='$REV', local_dir='$G/g1_data_v4', max_workers=8))" 2>&1 | tail -1
echo "$REV" > $G/g1_data_v4/REVISION
srun -N1 -c 16 --mem=128G --time=0:30:00 --job-name=g1v4_loader --container-name=pt_build --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  .venv/bin/python -c \"
from cosmos_framework.data.generator.action.datasets.g1_hug_lerobot_dataset import G1HugLeRobotDataset
d = G1HugLeRobotDataset(root='$G/g1_data_v4/g1_hug_v4', viewpoint='concat_view', action_stats_path='$G/g1_data_v4/g1_hug_v4/action_stats.json')
s = d[0]; print('LOADER', len(d), tuple(s['video'].shape), tuple(s['action'].shape), s['domain_id'].item())\" 2>&1 | grep -E 'LOADER|Loaded G1|Error|Traceback' | tail -4" 2>&1 | grep -v "^pyxis" | tee $G/g1_data_v4/loader_check.txt
L=$(grep "^LOADER" $G/g1_data_v4/loader_check.txt)
K=$(grep -oE "kept_episodes=[0-9]+" $G/g1_data_v4/loader_check.txt | cut -d= -f2)
if [[ "$L" == *"(3, 17, 540, 640) (16, 29) 15"* && -n "$K" && "$K" -ge 900 && "$K" -le 1016 ]]; then
  echo "loader check OK (kept $K): launching"
  $T/tmp/g1_concat_v4_run.sh &
  sleep 30; nohup $T/tmp/g1_upload_concat_v4_after.sh > $T/tmp/xfer/g1_upload_concat_v4.out 2>&1 &
  wait
else
  echo "loader check MISMATCH, NOT launching: $L kept=$K"
fi
