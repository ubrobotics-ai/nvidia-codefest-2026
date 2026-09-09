#!/bin/bash
# One chat/completions call against the running Nemotron server (see serve_nemotron.sh). Prints the URL to
# pass to cosmos_i2v_batch.py --nemotron-url on success.
set -euo pipefail
NODE=${NODE:-$(squeue -u $USER -h -n nemotron -o %N | head -1)}
: "${NODE:?no running job named nemotron}"
URL=http://$NODE:${PORT:-8150}/v1
echo "== $URL/models"; curl -sf $URL/models | python3 -m json.tool | grep '"id"'
echo "== chat/completions"
curl -sf $URL/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron", "temperature": 0.3, "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "system", "content": "You write structured JSON prompts for a physical-AI video generator. Output only a JSON object with keys scene, camera, motion, subjects, style."},
               {"role": "user", "content": "Forward camera of a small tracked rescue robot in Portuguese scrubland at dusk with thin smoke; one person lying 12 m ahead."}]}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print(m["content"]); print("usage:", r["usage"])'
echo; echo "NEMOTRON_URL=$URL"
