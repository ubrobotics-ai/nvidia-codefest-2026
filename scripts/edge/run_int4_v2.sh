#!/bin/bash
# Full corrected INT4 chain: quantise (projector excluded) -> stage (NO index JSONs) -> export (audited).
set -uo pipefail
TEAM=${TEAM:-/storage/hackathon_teams/omc-team15}
E=${EDGE_DIR:-$TEAM/codefest/edge}
export TMPDIR=/tmp/$USER PIP_CACHE_DIR=$TEAM/pip-cache
export HF_HOME=$TEAM/codefest/hf-cache HF_HUB_OFFLINE=1
export EDGELLM_QUANT_DATASET_MMMU=$E/calib_int4_mmmu_override.jsonl
mkdir -p $TMPDIR
CKPT=$E/Cosmos3-Edge
QCKPT=$E/Cosmos3-Edge/quantized-int4-awq-v2
ONNX=$E/Cosmos3-Edge/onnx-v2
rm -rf "$QCKPT" "$ONNX"

echo "### stage 1: INT4 AWQ quantize, projector excluded  $(date -Is)"
$E/venv/bin/python $E/l0/int4_quant_fix.py llm \
  --model_dir "$CKPT" --output_dir "$QCKPT" \
  --quantization int4_awq --lm_head_quantization int4_awq \
  --text_dataset cnn_dailymail --image_dataset mmmu \
  --num_samples 512 --dtype fp16 --device cuda
rc=$?; echo "### stage 1 exit $rc  $(date -Is)"; [ $rc -ne 0 ] && exit $rc

echo "### stage 2: stage auxiliary dirs -- NOT the index JSONs"
# The original model.safetensors.index.json describes the NATIVE checkpoint (698 flat keys pointing at
# transformer/ and vision_encoder/ shards). Symlinking it next to the quantised model.safetensors makes
# checkpoint/loader.py follow it instead of the real file: it then loads the diffusion and vision towers,
# silently skips every quantised tensor, and the export emits INT4 plugin nodes around dense FP16 weights.
# That is the 2026-09-10 defect. Stage everything the exporter needs EXCEPT the index.
for d in text_tokenizer transformer vae vision_encoder scheduler assets; do
  [ -e "$CKPT/$d" ] && ln -sfn "$CKPT/$d" "$QCKPT/$d"
done
for f in special_tokens_map.json model_index.json modular_model_index.json; do
  [ -e "$CKPT/$f" ] && ln -sfn "$CKPT/$f" "$QCKPT/$f"
done
ls -la "$QCKPT" | grep -c index.json | xargs -I{} echo "index JSONs staged: {} (model.safetensors.index.json must be absent)"
ls "$QCKPT" | grep 'model.safetensors.index.json' && { echo "ABORT: stale index present"; exit 2; }

echo "### stage 3: ONNX export, audited  $(date -Is)"
$E/venv/bin/python $E/l0/int4_export_diag.py "$QCKPT" "$ONNX" --externalize-weights int4_ffn
rc=$?; echo "### stage 3 exit $rc  $(date -Is)"
du -sh $ONNX/* 2>/dev/null
