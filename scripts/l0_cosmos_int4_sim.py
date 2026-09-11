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

    if arm == "nf4":
        # Real 4-bit kernels, not simulation. bitsandbytes NF4 executes quantised matmuls on device, so tok/s
        # and peak memory here MEAN something -- unlike the int4 arm, whose fake-quant dequantises every matmul.
        # It is a different quantiser from the shipped modelopt AWQ checkpoint (NormalFloat4, blockwise, no AWQ
        # smoothing), so this arm answers "what does naive 4-bit PTQ cost?", not "what does our checkpoint cost?".
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        # `llm_int8_skip_modules` patterns are matched by `should_convert_module` with
        # `re.match(f"{key}\\.", full_name)` -- ANCHORED AT THE START of the module path -- or an exact suffix.
        # Module names here are `model.visual.encoder.layers.N...`, so a bare "visual" matches nothing and
        # silently skips nothing: the first attempt quantised the vision tower too (333 Linear4bit instead of
        # ~169) and the model emitted degenerate repetition on 98% of items. Patterns must be rooted.
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["model.visual", "model.projector"])
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, quantization_config=qc, device_map="cuda", trust_remote_code=True).eval()
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        q4 = [n for n, m in model.named_modules() if type(m).__name__ == "Linear4bit"]
        leaked = [n for n in q4 if ".visual." in n or ".projector." in n]
        print(f"[nf4] Linear4bit modules: {len(q4)}  (AWQ quantises 169: 28 layers x 6 + lm_head)", flush=True)
        print(f"[nf4] vision/projector layers wrongly quantised: {len(leaked)}", flush=True)
        # Too FEW was the only case the first guard caught. Too MANY is what actually happened.
        if not q4:
            raise SystemExit("FAIL: no Linear4bit modules -- nothing was quantised.")
        if leaked:
            for n in leaked[:5]: print(f"[nf4]     {n}", flush=True)
            raise SystemExit(f"FAIL: {len(leaked)} vision/projector Linears were quantised. AWQ leaves them "
                             f"FP16, so this arm would not be comparable -- and a 4-bit vision tower produces "
                             f"degenerate output. Aborting instead of scoring it.")
        if not (150 <= len(q4) <= 180):
            raise SystemExit(f"FAIL: {len(q4)} Linear4bit modules, expected ~169 to match the AWQ arm.")
        return model, processor, {"quantized": True, "quantizer": "bitsandbytes NF4", "linear4bit": len(q4),
                                  "real_kernels": True}

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
    ap.add_argument("--arm", choices=["int4", "bf16", "nf4"], required=True)
    ap.add_argument("--categories", nargs="+", default=["distance", "left_right"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="reuse items already present in E_<tag>_raw.jsonl and score only the rest")
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

    # Stream results to disk as they are produced, and skip ids already scored on a previous attempt.
    # Three runs of the thinking arm were SIGKILLed near item 50 and lost ~50 completed evaluations each,
    # because outputs were only written after the final item. Appending per item makes a kill cost the
    # remainder rather than the whole run, and makes the arm resumable in chunks.
    raw_path = out / f"E_{tag}_raw.jsonl"
    done = {}
    if a.resume and raw_path.exists():
        for line in open(raw_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a kill mid-write can leave one truncated line
            done[r["id"]] = r
        if done:
            print(f"[resume] {len(done)} items already scored in {raw_path.name}; skipping those", flush=True)
    preds, raws, tok_total, gen_s, truncated = [], [], 0, 0.0, 0
    rawfh = open(raw_path, "a" if a.resume else "w", buffering=1)
    for k, item in enumerate(items, 1):
        if item["id"] in done:
            r = done[item["id"]]
            preds.append({"id": r["id"], "normalized_answer": r["pred"]}); raws.append(r)
            continue
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
        rec = {"id": item["id"], "category": item["category"], "image": item["image"],
               "gt": item["normalized_answer"], "pred": na, "raw": gen.strip()[:400]}
        preds.append({"id": item["id"], "normalized_answer": na}); raws.append(rec)
        rawfh.write(json.dumps(rec) + "\n")      # line-buffered: survives a kill
        if k % 50 == 0 or k == len(items):
            print(f"  [{k}/{len(items)}] {tok_total/max(gen_s,1e-9):.1f} tok/s", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    (out / f"E_{tag}_preds.json").write_text(json.dumps(preds))
    rawfh.close()   # already written incrementally above
    stats = {"model": a.model, "tag": tag, "arm": a.arm, "n": len(items), "categories": a.categories,
             "dtype": a.dtype, "seed": a.seed, "enable_thinking": a.enable_thinking,
             "runtime": {"int4": "transformers fake-quant (SIMULATED INT4)",
                         "nf4": "transformers + bitsandbytes NF4 (REAL 4-bit kernels)",
                         "bf16": "transformers"}[a.arm],
             "quantization_kind": {"int4": "PTQ (modelopt AWQ), SIMULATED -- not engine kernels",
                                   "nf4": "PTQ (bitsandbytes NF4), real kernels -- a DIFFERENT quantiser from "
                                          "the shipped AWQ checkpoint",
                                   "bf16": "none"}[a.arm],
             "hardware": "B300 -- NOT Orin; deployment figures come from the Jetson side",
             "tokens_generated": tok_total, "generate_seconds": round(gen_s, 1),
             "tokens_per_s": round(tok_total / max(gen_s, 1e-9), 2), "peak_gb": round(peak, 1),
             "max_new_tokens": a.max_new_tokens, "truncated_at_budget": truncated,
             "truncated_pct": round(100 * truncated / max(len(items), 1), 2), **qinfo}
    (out / f"E_{tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)
    # A model that runs to the token cap on nearly every item is looping, not answering. Scoring that produces
    # confident nonsense -- the NF4 first attempt read as "quantisation destroyed the model" when the real
    # cause was a config error. Flag it loudly rather than let the numbers stand.
    if stats["truncated_pct"] > 50:
        print(f"!! {stats['truncated_pct']}% of generations ran to the {a.max_new_tokens}-token cap. That is "
              f"degenerate output, not an answer. These accuracies are NOT a measurement of this model.",
              flush=True)


if __name__ == "__main__":
    main()
