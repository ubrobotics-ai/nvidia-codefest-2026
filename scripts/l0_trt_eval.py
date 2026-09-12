#!/usr/bin/env python3
"""L0 arm driven by a real TensorRT Edge-LLM engine (no fake quantisation).

The `int4` arm of l0_cosmos_int4_sim.py simulates W4A16 with ModelOpt's fake-quant:
the weights are rounded to the INT4 grid but the GEMM still runs in bf16, so it
measures the quantisation error and nothing about the deployed kernel. This driver
runs the same items through an actual TensorRT engine built from the Edge-LLM ONNX
export -- the same artefact the Jetson deploys -- so the INT4-AWQ row reflects the
real INT4 kernels, the real fragment weight layout and the real runtime.

Rendering, prompt and parser are imported from spatial_qa_eval, unchanged, so this
row is scored identically to the PyTorch rows and can be compared item-by-item.

  python scripts/l0_trt_eval.py --data <spatial-qa> --out <dir> \
      --engine-dir <engines/llm> --multimodal-engine-dir <engines> \
      --checkpoint-dir <onnx/llm> --llm-inference <path to llm_inference> --tag awq_trt
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_qa_eval import build_prompt, parse_answer, render_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--multimodal-engine-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--llm-inference", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--categories", nargs="+", default=["distance", "left_right"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--image-dir", default=None, help="where rendered frames are written")
    ap.add_argument("--chunk", type=int, default=100, help="requests per llm_inference invocation")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.image_dir or (out / "rendered")); img_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(os.path.join(a.data, "val.json")))
    items = [d for d in data if d["category"] in a.categories]
    if a.limit: items = items[: a.limit]
    print(f"{a.tag}: {len(items)} questions, categories {a.categories}", flush=True)

    # 1. Render exactly as the PyTorch arms do, but to disk so llm_inference can read it.
    reqs = []
    for item in items:
        p = img_dir / f"{item['id']}.png"
        # Size, not just existence: a run killed mid-render leaves a zero-byte PNG that
        # this skipped on the next pass, and llm_inference then died 450 items in with
        # 'Failed to load image'.
        if not p.exists() or p.stat().st_size == 0:
            img, n_reg = render_item(a.data, item, a.max_side)
            img.save(p)
        else:
            _, n_reg = render_item(a.data, item, a.max_side)
        reqs.append({"id": item["id"], "category": item["category"], "image": item["image"],
                     "gt": item["normalized_answer"], "path": str(p),
                     "prompt": build_prompt(item, n_reg)})
    print(f"rendered {len(reqs)} frames into {img_dir}", flush=True)

    # 2. Drive llm_inference in chunks so one failure does not cost the whole run.
    raw_path = out / f"E_{a.tag}_raw.jsonl"
    rawfh = open(raw_path, "w", buffering=1)
    preds, raws, gen_s = [], [], 0.0
    for start in range(0, len(reqs), a.chunk):
        chunk = reqs[start:start + a.chunk]
        payload = {"batch_size": 1, "temperature": 0.0,
                   "max_generate_length": a.max_new_tokens,
                   "requests": [{"messages": [{"role": "user", "content": [
                       {"type": "image", "image": r["path"]},
                       {"type": "text", "text": r["prompt"]}]}]} for r in chunk]}
        inp = out / f".req_{start}.json"; outp = out / f".out_{start}.json"
        inp.write_text(json.dumps(payload))
        t = time.time()
        proc = subprocess.run(
            [a.llm_inference, "--engineDir", a.engine_dir,
             "--multimodalEngineDir", a.multimodal_engine_dir,
             "--checkpointDir", a.checkpoint_dir,
             "--inputFile", str(inp), "--outputFile", str(outp)],
            capture_output=True, text=True)
        gen_s += time.time() - t
        if proc.returncode != 0 or not outp.exists():
            sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
            raise SystemExit(f"llm_inference failed on chunk at {start}")
        resp = json.load(open(outp))["responses"]
        by_idx = {int(r.get("request_idx", i)): r for i, r in enumerate(resp)}
        for i, r in enumerate(chunk):
            gen = (by_idx.get(i) or {}).get("output_text", "")
            na = parse_answer(gen, r["category"])
            rec = {"id": r["id"], "category": r["category"], "image": r["image"],
                   "gt": r["gt"], "pred": na, "raw": gen.strip()[:400]}
            preds.append({"id": r["id"], "normalized_answer": na}); raws.append(rec)
            rawfh.write(json.dumps(rec) + "\n")
        inp.unlink(missing_ok=True); outp.unlink(missing_ok=True)
        print(f"  [{min(start+a.chunk, len(reqs))}/{len(reqs)}]  {gen_s:.0f}s elapsed", flush=True)
    rawfh.close()

    (out / f"E_{a.tag}_preds.json").write_text(json.dumps(preds))
    empty = sum(1 for r in raws if not r["raw"])
    stats = {"model": a.engine_dir, "tag": a.tag, "arm": "trt", "n": len(items),
             "categories": a.categories, "runtime": "tensorrt-edgellm",
             "generate_seconds": round(gen_s, 1), "empty_generations": empty}
    (out / f"E_{a.tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    # The Gemma arm silently scored parser fallbacks when generations came back empty;
    # abort loudly here rather than publish a row built on them.
    if empty > 0.05 * len(raws):
        raise SystemExit(f"ABORT: {empty}/{len(raws)} empty generations (>5%)")


if __name__ == "__main__":
    main()
