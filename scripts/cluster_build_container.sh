#!/bin/bash
# Build the named pyxis container `ubr` once. Run from the LOGIN node, inside tmux.
# Prereqs: export HF_TOKEN=hf_... (licences for nvidia/Cosmos3-Nano accepted on huggingface.co)
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
export TEAM=${TEAM:-/storage/hackathon_teams/omc-team15}
export WORK=${WORK:-$TEAM/codefest}
# enroot must not use /run/user/<uid> (not writable on dgx01) and home is only 50 GB: keep the big stuff on team storage
# NOTE: pyxis does NOT honour these ENROOT_* vars (enroot runs under slurmd). They are kept for direct `enroot` use only.
# The effective fix is symlinking ~/.cache/enroot and ~/.local/share/enroot to $TEAM/enroot/{cache,data} — see docs.
export ENROOT_RUNTIME_PATH=${ENROOT_RUNTIME_PATH:-/tmp/enroot-$USER/run}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$TEAM/enroot/data}
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$TEAM/enroot/cache}
export XDG_RUNTIME_DIR=$ENROOT_RUNTIME_PATH
# Scratch: node-local /tmp is 1.6 TB free at 1.9 GB/s; team storage is NFS from dgx02 at 871 MB/s, and CUDA JIT
# plus enroot unpacking are latency-sensitive. (.bashrc sets TMPDIR for interactive shells only -- it returns early
# when non-interactive, which is why every script must set this itself.) Note /tmp INSIDE the container is tmpfs,
# i.e. RAM (1008 GB cap on a 1.8 TB node): fast, but do not write hundreds of GB of scratch there.
export TMPDIR=${TMPDIR_OVERRIDE:-/tmp/$USER}; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$TEAM/pip-cache}   # keep pip wheels off the 50 GB home quota
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_DATA_PATH $ENROOT_CACHE_PATH $WORK/hf-cache $WORK/repo
module load python312 py-uv tree 2>/dev/null || true

srun --gres=gpu:1 --time=01:30:00 --job-name=build --pty \
  --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
  --container-name=ubr \
  --container-mounts=$HOME:$HOME,$TEAM:$TEAM \
  --container-workdir=$WORK \
  --container-env=HF_TOKEN \
  bash -c '
set -e
export HF_HOME='"$WORK"'/hf-cache
echo "== GPU"; nvidia-smi -L
# pip has no write access to the container site-packages, so it falls back to a --user install whose bin dir
# ($HOME/.local/bin) is not on PATH. Put it there rather than calling console scripts by bare name.
export PATH=$HOME/.local/bin:$PATH TMPDIR=/tmp/$USER PIP_CACHE_DIR='"$TEAM"'/pip-cache
mkdir -p $TMPDIR
echo "== python deps"
pip install -U pip
# One pip invocation: pip resolves each run independently, so a second run can silently downgrade the first.
pip install -U "diffusers>=0.36" transformers accelerate imageio[ffmpeg] huggingface_hub cosmos_guardrail "vllm[omni]" \
  || pip install -U "diffusers>=0.36" transformers accelerate imageio[ffmpeg] huggingface_hub cosmos_guardrail vllm
# cosmos-framework is deliberately NOT installed: cosmos_i2v_batch.py drives Cosmos 3 through Diffusers
# (Cosmos3OmniPipeline), and cosmos-framework 1.2.2 pins transformers<5 / huggingface_hub<1, which downgrades
# both out from under vllm (needs transformers>=5.10.4) and cosmos_guardrail (needs transformers>=5.0.0).
echo "== dependency check"
pip check || echo "[warn] pip check reported conflicts above - fix before trusting this container"
echo "== models"
python - <<PY
import pathlib
from huggingface_hub import snapshot_download
for repo in ("nvidia/Cosmos3-Nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"):
    print("downloading", repo, flush=True)
    snapshot_download(repo)
# Cosmos3OmniPipeline.__init__ builds the cosmos_guardrail CosmosSafetyChecker, which fetches these three at
# construction time. Mirror its exact calls (cosmos_guardrail.py, SigLIPEncoder / Blocklist / RetinaFace /
# Qwen3Guard) so a later HF_HUB_OFFLINE=1 run finds them in the cache.
print("downloading nvidia/Cosmos-1.0-Guardrail (+ siglip, Qwen3Guard)", flush=True)
g = snapshot_download("nvidia/Cosmos-1.0-Guardrail",
                      allow_patterns=["blocklist/*", "video_content_safety_filter/*", "face_blur_filter/Resnet50_Final.pth"])
snapshot_download("google/siglip-so400m-patch14-384", cache_dir=(pathlib.Path(g) / "video_content_safety_filter").as_posix())
snapshot_download("Qwen/Qwen3Guard-Gen-0.6B")
PY
# nltk>=3.10 (pathsec) refuses to open symlinked data files and group-writable roots, which is exactly what the HF cache
# on team storage is. Give the guardrail a real-file copy of its nltk_data in $HOME and point NLTK_DATA at it at run time.
G=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(\"nvidia/Cosmos-1.0-Guardrail\", allow_patterns=[\"blocklist/*\"]))")
rm -rf $HOME/nltk_data && cp -rL "$G/blocklist/nltk_data" $HOME/nltk_data && chmod -R u+rwX,go+rX,go-w $HOME/nltk_data
echo "nltk_data materialised at $HOME/nltk_data (export NLTK_DATA=\$HOME/nltk_data when running the guardrail)"
echo "== sanity"
python - <<PY
import torch, diffusers, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)
from diffusers import Cosmos3OmniPipeline; print("Cosmos3OmniPipeline OK")
PY
du -sh '"$WORK"'/hf-cache
echo READY'
