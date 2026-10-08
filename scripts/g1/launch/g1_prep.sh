#!/bin/bash
G=/storage/hackathon_teams/omc-team15/codefest/g1_cosmos; T=/storage/hackathon_teams/omc-team15
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 -c 32 --mem=200G --time=1:00:00 --job-name=g1_prep --container-name=pt_g1 --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "cd $G/cosmos-framework; V=.venv/lib/python3.13/site-packages
  export PYTHONNOUSERSITE=1 PYTHONPATH=. TMPDIR=/tmp HF_HOME=$T/codefest/hf-cache LD_LIBRARY_PATH=\$PWD/\$V/nvidia/cu13/lib:$G/ffmpeg_links:\$PWD/\$V/av.libs:\$LD_LIBRARY_PATH
  export G1_HUG_ROOT=$G/g1_data/g1_hug_v1 G1_HUG_STATS=$G/g1_data/action_stats.json
  echo '--- loader check'
  .venv/bin/python -c \"
from cosmos_framework.data.generator.action.datasets.g1_hug_lerobot_dataset import G1HugLeRobotDataset
import os; d = G1HugLeRobotDataset(root=os.environ['G1_HUG_ROOT'], viewpoint='ego_view', action_stats_path=os.environ['G1_HUG_STATS'])
s = d[0]; print('LOADER', len(d), tuple(s['video'].shape), tuple(s['action'].shape), s['domain_id'].item())\" 2>&1 | grep -E 'LOADER|Loaded G1|Error|error' | tail -5
  echo '--- DCP conversion'
  .venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp -o examples/checkpoints/Cosmos3-Edge --checkpoint-path $G/Cosmos3-Edge 2>&1 | tail -6
  echo dcp_rc=\$?; du -sh examples/checkpoints/Cosmos3-Edge" > $G/prep.log 2>&1
echo PREPDONE >> $G/prep.log
