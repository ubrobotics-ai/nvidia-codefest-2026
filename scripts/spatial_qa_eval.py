#!/usr/bin/env python3
"""Task E: zero-shot Spatial-QA bake-off between two VLMs on the PhysicalAI Spatial-Intelligence-Warehouse val split.

  python scripts/spatial_qa_eval.py --data $TEAM/data/nvidia/spatial-qa --out <dir> \
      --model nvidia/Cosmos3-Edge --tag cosmos3edge [--categories distance left_right] [--limit N]

The questions reference regions as `<mask>` tokens whose RLE masks ship with each item. A model shown the bare image
cannot know which object is meant, so each `<mask>` is replaced by `[Region i]` in the text and region i is drawn on
the image as a coloured outline with its number -- the same referring convention the ground-truth answers use. This
keeps the task well posed; it is a zero-shot baseline, not a leaderboard submission.

Writes <out>/E_<tag>_preds.json ({id, normalized_answer}), <out>/E_<tag>_raw.jsonl (full generations),
<out>/E_gt_<categories>.json (matching gt subset) and <out>/E_<tag>_stats.json (tokens/s, peak GB).
"""
import argparse, json, os, re, sys, time
from pathlib import Path

COLORS = [(255,64,64),(64,160,255),(64,255,96),(255,208,48),(220,96,255),(64,240,240),(255,144,32),(180,180,180),
          (255,96,160),(128,255,32),(96,128,255),(240,120,120)]

def render_item(data_dir, item, max_side=1024):
    """Draw each region's numbered bounding box on the image, then downscale to `max_side`.

    Module level so every arm of the L0 bake-off -- transformers, vLLM, simulated INT4, llama.cpp -- renders
    the identical image. Round 1 called this as a closure; the body is unchanged.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as maskutil
    img = Image.open(os.path.join(data_dir, "val", "images", item["image"])).convert("RGB")
    rles = item["rle"]
    if isinstance(rles, str): rles = eval(rles)
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rles):
        rr = dict(r)
        if isinstance(rr.get("counts"), str): rr["counts"] = rr["counts"].encode()
        m = maskutil.decode(rr)
        ys, xs = np.nonzero(m)
        if len(xs) == 0: continue
        c = COLORS[i % len(COLORS)]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        d.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=c, width=4)
        d.text((int(x0) + 5, int(y0) + 3), str(i), fill=c)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img, len(rles)


def build_prompt(item, n_regions):
    conv = item["conversations"]
    if isinstance(conv, str): conv = eval(conv)
    q = conv[0]["value"].replace("<image>", "").strip()
    i = 0
    def sub(_):
        nonlocal i
        s = f"[Region {i}]"; i += 1; return s
    q = re.sub(r"<mask>", sub, q)
    cat = item["category"]
    if cat == "distance":
        instr = "Answer with the distance in metres as a single number, for example: 12.3"
    elif cat == "left_right":
        instr = "Answer with exactly one word: left or right."
    elif cat == "count":
        instr = "Answer with a single integer."
    else:  # mcq: the answer is a region index, not a letter
        instr = "Answer with the number of the correct region, for example: 3"
    return (f"{q}\n\nThe numbered coloured outlines in the image mark the regions referred to above. {instr}\n"
            f"Reply with the answer only, on one line, in the form: ANSWER: <value>")

def parse_answer(text, cat):
    t = text.strip()
    # prefer the explicit marker; otherwise fall back to the LAST match, since any reasoning text mentions the
    # region indices ("[Region 0]") and the words left/right long before it commits to an answer.
    m = re.findall(r"ANSWER\s*:\s*(.+)", t, flags=re.I)
    if m: t = m[-1].strip()
    if cat in ("distance", "count"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
        if not nums: return "0"
        return nums[-1] if cat == "distance" else str(int(float(nums[-1])))
    if cat == "left_right":
        low = t.lower()
        fl, fr = low.rfind("left"), low.rfind("right")
        if fl < 0 and fr < 0: return "left"
        return "right" if fr > fl else "left"
    # mcq answers are region indices ("0".."10"), not option letters
    nums = re.findall(r"\d+", t)
    return nums[-1] if nums else "0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--categories", nargs="+", default=["distance", "left_right"])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-side", type=int, default=1024)
    # A modelopt INT4 checkpoint has GPTQ-packed weights whose shapes do not match the fp16 originals, so plain
    # transformers refuses to load it. vLLM understands the packing. Everything else -- region rendering, prompt,
    # parser -- is shared with the transformers path, which is what keeps the quantisation delta comparable.
    ap.add_argument("--vllm", action="store_true", help="load via vLLM (required for modelopt-quantised checkpoints)")
    ap.add_argument("--quantization", default="modelopt")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(os.path.join(a.data, "val.json")))
    items = [d for d in data if d["category"] in a.categories]
    if a.limit: items = items[: a.limit]
    print(f"{len(items)} questions, categories {a.categories}", flush=True)

    import torch, numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as maskutil
    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained(a.model)
    llm = sp = None
    if a.vllm:
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, quantization=a.quantization, trust_remote_code=True,
                  max_model_len=8192, gpu_memory_utilization=0.55, limit_mm_per_prompt={"image": 1})
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens)
        model = None
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    def render(item):
        return render_item(a.data, item, a.max_side)

    preds, raws, tok_total, gen_s = [], [], 0, 0.0
    for k, item in enumerate(items, 1):
        img, n_reg = render(item)
        prompt = build_prompt(item, n_reg)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            text = prompt
        t = time.time()
        if llm is not None:
            out = llm.generate([{"prompt": text, "multi_modal_data": {"image": img}}], sp)
            dt = time.time() - t
            gen = out[0].outputs[0].text
            ntok = len(out[0].outputs[0].token_ids)
        else:
            inp = proc(images=img, text=text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
            dt = time.time() - t
            newt = o[0][inp["input_ids"].shape[1]:]
            gen = proc.decode(newt, skip_special_tokens=True)
            ntok = int(newt.shape[0])
        tok_total += ntok; gen_s += dt
        na = parse_answer(gen, item["category"])
        preds.append({"id": item["id"], "normalized_answer": na})
        raws.append({"id": item["id"], "category": item["category"], "image": item["image"],
                     "gt": item["normalized_answer"], "pred": na, "raw": gen.strip()[:400], "prompt": prompt[:400]})
        if k % 25 == 0 or k == len(items):
            print(f"  [{k}/{len(items)}] {tok_total/max(gen_s,1e-9):.1f} tok/s", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    (out / f"E_{a.tag}_preds.json").write_text(json.dumps(preds))
    with open(out / f"E_{a.tag}_raw.jsonl", "w") as fh:
        for r in raws: fh.write(json.dumps(r) + "\n")
    gt = [{"id": d["id"], "normalized_answer": d["normalized_answer"], "category": d["category"]} for d in items]
    (out / f"E_gt_{'_'.join(a.categories)}.json").write_text(json.dumps(gt))
    stats = {"model": a.model, "tag": a.tag, "n": len(items), "categories": a.categories,
             "tokens_generated": tok_total, "generate_seconds": round(gen_s, 1),
             "tokens_per_s": round(tok_total / max(gen_s, 1e-9), 2), "peak_gb": round(peak, 1),
             "max_new_tokens": a.max_new_tokens}
    (out / f"E_{a.tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)

    # quick per-category accuracy (the official scorer is run separately on these files)
    from collections import defaultdict
    ok, tot = defaultdict(int), defaultdict(int)
    for r in raws:
        tot[r["category"]] += 1
        if r["category"] == "distance":
            try: ok[r["category"]] += int(abs(float(r["pred"]) - float(r["gt"])) / max(float(r["gt"]), 1e-9) < 0.10)
            except Exception: pass
        else:
            ok[r["category"]] += int(str(r["pred"]).lower() == str(r["gt"]).lower())
    for c in tot: print(f"== {a.tag} {c}: {ok[c]}/{tot[c]} = {100*ok[c]/tot[c]:.1f}%", flush=True)

if __name__ == "__main__":
    main()
