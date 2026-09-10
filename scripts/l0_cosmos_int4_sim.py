#!/usr/bin/env python3
"""L0 row 4: score Cosmos3-Edge at simulated INT4 AWQ, in memory, before export.

Nothing on this host can execute the exported `W4A16_AWQ` checkpoint: transformers refuses the GPTQ packing,
vLLM's modelopt path takes only FP8/NVFP4/MXFP8, and TensorRT is not installed. But `mtq.quantize()` returns an
ordinary `torch.nn.Module` whose weights are quantised-then-dequantised on the fly, and that module generates like
any other. That is what this scores.

**These are simulated-quantisation numbers, not engine kernels.** They are the right number for the weight-precision
delta and the wrong number for runtime. tok/s and peak GB below describe a fake-quant module on a B300 and mean
nothing for a Jetson.

The quantisation config, the calibration set and the calibration loop are taken from the same
`tensorrt_edgellm` code that produced the shipped checkpoint (`--quantization int4_awq --lm_head_quantization
int4_awq --image_dataset mmmu`, with `EDGELLM_QUANT_DATASET_MMMU` pointing at our own frames), so the module scored
here is the same object the exporter serialised.

  python scripts/l0_cosmos_int4_sim.py --data $TEAM/data/nvidia/spatial-qa --out <dir> --arm int4
  python scripts/l0_cosmos_int4_sim.py --data ... --out <dir> --arm bf16     # same-environment control
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_qa_eval import build_prompt, parse_answer, render_item  # identical prompt / parser / rendering


def build_model(model_dir, arm, calib_samples, dtype):
    import torch
    import modelopt.torch.quantization as mtq
    from tensorrt_edgellm.quantization.quantize import (
        _load_model, _multimodal_calib_dataloader, _calibrate_multimodal,
        _is_image_blind_calibration)
    from tensorrt_edgellm.chat_template import _get_model_type
    from tensorrt_edgellm.quantization.quantization_configs import build_quant_config
    from tensorrt_edgellm.quantization.datasets import resolve_dataset, dataset_name
    from transformers import AutoProcessor

    model, tokenizer, _ = _load_model(model_dir, dtype=dtype, device="cuda")
    model.eval()
    # Evaluate through a PLAIN AutoProcessor, the way round 1 did. `_load_model` builds its own with
    # min_pixels/max_pixels set, which changes how many visual tokens an image becomes -- that would make this row
    # incomparable with the bf16 baseline for a reason that has nothing to do with quantisation. The exporter's
    # image-calibration branch also uses a plain processor, so nothing about the quantised object changes either.
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if arm == "bf16":
        return model, processor, {"quantized": False}

    quant_cfg = build_quant_config("int4_awq", "int4_awq", None)
    # Same 64-alignment exclusion the exporter applies: the cuteDSL INT4 GEMM repack needs N%64==0 && K%64==0.
    skipped = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and (module.out_features % 64 or module.in_features % 64):
            quant_cfg["quant_cfg"].append({"quantizer_name": f"*{name}.weight_quantizer", "enable": False})
            skipped.append(name)
    print(f"[int4] {len(skipped)} Linear(s) left fp16 for 64-alignment", flush=True)

    blind = _is_image_blind_calibration(model, quant_cfg)
    print(f"[int4] image-blind calibration required: {blind}  (awq_lite reshapes weights against the calib "
          f"distribution, so a text-only pass would optimise the image-token ranges away)", flush=True)
    image_ds = resolve_dataset("mmmu", "image")
    print(f"[int4] image calibration dataset: {dataset_name(image_ds)}", flush=True)
    try:
        from tensorrt_edgellm.quantization.gemma4_patch import apply as _gemma4_patch
        _gemma4_patch(model, _get_model_type(model_dir))
    except Exception as e:
        print(f"[int4] gemma4_patch skipped: {e}", flush=True)
    cproc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    batches = _multimodal_calib_dataloader(cproc, image_dataset=image_ds, num_samples=calib_samples)
    print(f"[int4] {len(batches)} calibration batches", flush=True)
    t0 = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=lambda m: _calibrate_multimodal(m, batches))
    print(f"[int4] mtq.quantize: {time.time()-t0:.1f}s", flush=True)
    mtq.print_quant_summary(model)
    return model, processor, {"quantized": True, "calib_batches": len(batches),
                              "alignment_skipped": len(skipped), "image_blind": bool(blind)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("WORK", "") + "/edge/Cosmos3-Edge")
    ap.add_argument("--arm", choices=["int4", "bf16"], required=True)
    ap.add_argument("--categories", nargs="+", default=["distance", "left_right"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or f"cosmos3edge_{a.arm}" + ("_think" if a.enable_thinking else "")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    torch.manual_seed(a.seed)
    data = json.load(open(os.path.join(a.data, "val.json")))
    items = [d for d in data if d["category"] in a.categories]
    if a.limit: items = items[: a.limit]
    print(f"{tag}: {len(items)} questions, categories {a.categories}", flush=True)

    model, processor, qinfo = build_model(a.model, a.arm, a.calib_samples, a.dtype)
    torch.cuda.reset_peak_memory_stats()

    preds, raws, tok_total, gen_s, truncated = [], [], 0, 0.0, 0
    for k, item in enumerate(items, 1):
        img, n_reg = render_item(a.data, item, a.max_side)
        prompt = build_prompt(item, n_reg)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=a.enable_thinking)
        except TypeError:
            text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            text = prompt
        inp = processor(images=img, text=text, return_tensors="pt").to("cuda")
        t = time.time()
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        dt = time.time() - t
        newt = o[0][inp["input_ids"].shape[1]:]
        gen = processor.decode(newt, skip_special_tokens=True)
        ntok = int(newt.shape[0])
        tok_total += ntok; gen_s += dt
        # A generation that used the whole budget probably never reached "ANSWER:"; the parser then
        # falls back to the last number/word in the reasoning text, which is not the model's answer.
        truncated += int(ntok >= a.max_new_tokens)
        na = parse_answer(gen, item["category"])
        preds.append({"id": item["id"], "normalized_answer": na})
        raws.append({"id": item["id"], "category": item["category"], "image": item["image"],
                     "gt": item["normalized_answer"], "pred": na, "raw": gen.strip()[:400]})
        if k % 50 == 0 or k == len(items):
            print(f"  [{k}/{len(items)}] {tok_total/max(gen_s,1e-9):.1f} tok/s", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    (out / f"E_{tag}_preds.json").write_text(json.dumps(preds))
    with open(out / f"E_{tag}_raw.jsonl", "w") as fh:
        for r in raws: fh.write(json.dumps(r) + "\n")
    stats = {"model": a.model, "tag": tag, "arm": a.arm, "n": len(items), "categories": a.categories,
             "dtype": a.dtype, "seed": a.seed, "enable_thinking": a.enable_thinking,
             "runtime": "transformers fake-quant (simulated INT4)" if a.arm == "int4" else "transformers",
             "quantization_kind": "PTQ (modelopt AWQ), SIMULATED -- not engine kernels" if a.arm == "int4" else "none",
             "hardware": "B300 -- NOT Orin; deployment figures come from the Jetson side",
             "tokens_generated": tok_total, "generate_seconds": round(gen_s, 1),
             "tokens_per_s": round(tok_total / max(gen_s, 1e-9), 2), "peak_gb": round(peak, 1),
             "max_new_tokens": a.max_new_tokens, "truncated_at_budget": truncated,
             "truncated_pct": round(100 * truncated / max(len(items), 1), 2), **qinfo}
    (out / f"E_{tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
