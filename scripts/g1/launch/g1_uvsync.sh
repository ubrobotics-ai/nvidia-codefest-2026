#!/bin/bash
G=/storage/hackathon_teams/omc-team15/codefest/g1_cosmos; T=/storage/hackathon_teams/omc-team15
export ENROOT_RUNTIME_PATH=/tmp/enroot-$USER/run XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
srun -N1 -c 32 --mem=128G --time=2:00:00 --job-name=g1_uvsync --container-image='nvcr.io#nvidia/pytorch:25.08-py3' --container-name=pt_g1 \
  --container-mounts=$HOME:$HOME,$T:$T \
  bash -c "export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin UV_CACHE_DIR=$T/uv-cache UV_PYTHON_INSTALL_DIR=$T/uv-python PYTHONNOUSERSITE=1 TMPDIR=/tmp MAX_JOBS=16
  cd $G/cosmos-framework && uv --version && uv sync --all-extras --group=cu130-train 2>&1 | tail -25
  echo uv_sync_rc=\$?
  .venv/bin/python -c 'import torch, sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)'
  .venv/bin/python -c 'import transformer_engine, flash_attn; print(\"te\", transformer_engine.__version__, \"fa\", flash_attn.__version__)' 2>&1 | tail -1" > $G/uvsync.log 2>&1
echo UVDONE >> $G/uvsync.log
