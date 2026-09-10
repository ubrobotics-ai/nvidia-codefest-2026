#!/usr/bin/env python3
"""L0 row 3: score `gemma-4-E4B-it-qat-UD-Q4_K_XL` through llama.cpp -- the artefact actually on the rover.

Talks to a running `llama-server` (built with CUDA, `--mmproj` loaded) over its OpenAI-compatible endpoint. The
image goes in as a base64 data URI; everything else -- the 986 + 456 items, the rendered numbered region outlines,
the prompt text, the parser including the mcq region-index fix -- is imported from `spatial_qa_eval` so this row is
comparable with the transformers rows item for item.

This quantisation is **QAT** -- quantisation-aware trained upstream by Google, then packed to Q4_K_XL -- not
post-training quantisation like the Cosmos INT4 arm. The two 4-bit rows are not the same kind of object.

tok/s here is a llama.cpp-on-B300 figure. It is not an Orin figure and must not be read as one.

  python scripts/l0_gemma_gguf_eval.py --data $TEAM/data/nvidia/spatial-qa --out <dir> --url http://127.0.0.1:8151
"""
import argparse, base64, io, json, os, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_qa_eval import build_prompt, parse_answer, render_item


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8151")
    ap.add_argument("--tag", default="gemma4e4b_q4kxl")
    ap.add_argument("--categories", nargs="+", default=["distance", "left_right"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(os.path.join(a.data, "val.json")))
    items = [d for d in data if d["category"] in a.categories]
    if a.limit: items = items[: a.limit]
    print(f"{a.tag}: {len(items)} questions, categories {a.categories}", flush=True)

    props = post(a.url + "/v1/models", {}) if False else json.load(
        urllib.request.urlopen(a.url + "/v1/models", timeout=60))
    served = props["data"][0]["id"]
    print(f"server model: {served}", flush=True)

    preds, raws, tok_total, gen_s, truncated, empty = [], [], 0, 0.0, 0, 0
    for k, item in enumerate(items, 1):
        img, n_reg = render_item(a.data, item, a.max_side)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        prompt = build_prompt(item, n_reg)
        # Gemma-4 QAT ships thinking ON by default. Left alone, llama.cpp routes the whole generation into
        # `reasoning_content`, `content` comes back empty, and the run burns the entire token budget without ever
        # emitting an answer -- 96% truncation, every distance parsed as 0 and every left_right as the "left"
        # fallback. Round 1 ran `enable_thinking=False`; this is the same switch, so the row is comparable.
        payload = {"model": served, "temperature": 0.0, "top_k": 1, "seed": a.seed,
                   "max_tokens": a.max_new_tokens,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": uri}},
                       {"type": "text", "text": prompt}]}]}
        t = time.time()
        r = post(a.url + "/v1/chat/completions", payload)
        dt = time.time() - t
        msg = r["choices"][0]["message"]
        gen = msg.get("content") or ""
        if not gen.strip():
            empty += 1
            gen = msg.get("reasoning_content") or ""   # still score it, but the count below flags the run
        ntok = int(r.get("usage", {}).get("completion_tokens", 0))
        tok_total += ntok; gen_s += dt
        truncated += int(r["choices"][0].get("finish_reason") == "length")
        na = parse_answer(gen, item["category"])
        preds.append({"id": item["id"], "normalized_answer": na})
        raws.append({"id": item["id"], "category": item["category"], "image": item["image"],
                     "gt": item["normalized_answer"], "pred": na, "raw": gen.strip()[:400]})
        if k % 50 == 0 or k == len(items):
            print(f"  [{k}/{len(items)}] {tok_total/max(gen_s,1e-9):.1f} tok/s", flush=True)

    (out / f"E_{a.tag}_preds.json").write_text(json.dumps(preds))
    with open(out / f"E_{a.tag}_raw.jsonl", "w") as fh:
        for r in raws: fh.write(json.dumps(r) + "\n")
    stats = {"model": served, "tag": a.tag, "n": len(items), "categories": a.categories, "seed": a.seed,
             "runtime": "llama.cpp (llama-server, CUDA)", "quantization_kind": "QAT (upstream Google), packed uniform Q4_0 by Unsloth -- the UD-Q4_K_XL filename is a misnomer: 100% of parameters are stored Q4_0, no K-quants",
             "hardware": "B300 -- NOT Orin; deployment figures come from the Jetson side",
             "tokens_generated": tok_total, "generate_seconds": round(gen_s, 1),
             "tokens_per_s": round(tok_total / max(gen_s, 1e-9), 2), "peak_gb": None,
             "max_new_tokens": a.max_new_tokens, "truncated_at_budget": truncated,
             "truncated_pct": round(100 * truncated / max(len(items), 1), 2),
             "empty_content": empty, "empty_content_pct": round(100 * empty / max(len(items), 1), 2)}
    (out / f"E_{a.tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)
    if stats["empty_content_pct"] > 5:
        print(f"!! {stats['empty_content_pct']}% of generations returned empty content -- the chat template is "
              f"routing output somewhere the parser never sees. This row is NOT comparable; fix before reporting.",
              flush=True)


if __name__ == "__main__":
    main()
