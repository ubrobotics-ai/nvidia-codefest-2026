#!/usr/bin/env python3
"""Order-swap probe for the left_right prior.

left_right asks "does pallet A appear on the right-hand side of pallet B?", with A and B
substituted as [Region 0] and [Region 1] in order of appearance. Swapping the two
substitutions asks the mirrored question about the SAME image, so the correct answer
inverts. A model judging geometry flips with it; a model leaning on a "left" prior does
not. Averaging the two runs is the standard order-debiasing move, and this measures what
it is worth before anyone trains anything.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["TEAM"] + "/codefest/repo/scripts")
from spatial_qa_eval import render_item, parse_answer

def build_prompt_swapped(item, swap):
    conv = item["conversations"]
    if isinstance(conv, str): conv = eval(conv)
    q = conv[0]["value"].replace("<image>", "").strip()
    order = [1, 0] if swap else [0, 1]
    i = 0
    def sub(_):
        nonlocal i
        s = f"[Region {order[i]}]" if i < 2 else f"[Region {i}]"
        i += 1
        return s
    q = re.sub(r"<mask>", sub, q)
    return (f"{q}\n\nThe numbered coloured outlines in the image mark the regions referred to above. "
            f"Answer with exactly one word: left or right.\n"
            f"Reply with the answer only, on one line, in the form: ANSWER: <value>")

ap = argparse.ArgumentParser()
for a in ("--data","--out","--engine-dir","--multimodal-engine-dir","--checkpoint-dir","--llm-inference"):
    ap.add_argument(a, required=True)
ap.add_argument("--image-dir", required=True); ap.add_argument("--chunk", type=int, default=100)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
items = [d for d in json.load(open(os.path.join(a.data, "val.json"))) if d["category"] == "left_right"]
print(f"left_right items: {len(items)}", flush=True)

def run(swap, tag):
    recs = []
    for s in range(0, len(items), a.chunk):
        chunk = items[s:s+a.chunk]
        payload = {"batch_size": 1, "temperature": 0.0, "max_generate_length": 192,
                   "requests": [{"messages":[{"role":"user","content":[
                       {"type":"image","image": str(Path(a.image_dir)/f"{it['id']}.png")},
                       {"type":"text","text": build_prompt_swapped(it, swap)}]}]} for it in chunk]}
        ip = out/f".sw_{tag}_{s}.json"; op = out/f".swo_{tag}_{s}.json"
        ip.write_text(json.dumps(payload))
        p = subprocess.run([a.llm_inference,"--engineDir",a.engine_dir,
                            "--multimodalEngineDir",a.multimodal_engine_dir,
                            "--checkpointDir",a.checkpoint_dir,
                            "--inputFile",str(ip),"--outputFile",str(op)],
                           capture_output=True, text=True)
        if p.returncode or not op.exists():
            sys.stderr.write(p.stdout[-2000:]+p.stderr[-2000:]); raise SystemExit(f"failed at {s}")
        resp = json.load(open(op))["responses"]
        by = {int(r.get("request_idx", i)): r for i, r in enumerate(resp)}
        for i, it in enumerate(chunk):
            recs.append({"id": it["id"], "gt": it["normalized_answer"],
                         "pred": parse_answer((by.get(i) or {}).get("output_text",""), "left_right")})
        ip.unlink(missing_ok=True); op.unlink(missing_ok=True)
        print(f"  {tag}: {min(s+a.chunk,len(items))}/{len(items)}", flush=True)
    return {r["id"]: r for r in recs}

t0 = time.time()
A = run(False, "orig"); B = run(True, "swap")
inv = {"left": "right", "right": "left"}
n = agree_flip = agree_same = 0
acc_A = acc_deb = acc_consistent = 0; n_consistent = 0
rows = []
for k, ra in A.items():
    rb = B.get(k)
    if not rb: continue
    n += 1; gt = str(ra["gt"]); pa, pb = str(ra["pred"]), str(rb["pred"])
    flipped = (pb == inv.get(pa, None))
    agree_flip += flipped; agree_same += (pa == pb)
    acc_A += (pa == gt)
    # debiased: trust the pair when it flips; on a non-flip the prior is showing, so take
    # the answer the model is biased AGAINST rather than keeping the biased one.
    deb = pa if flipped else "right"
    acc_deb += (deb == gt)
    if flipped: n_consistent += 1; acc_consistent += (pa == gt)
    rows.append({"id": k, "gt": gt, "orig": pa, "swap": pb, "flipped": flipped, "debiased": deb})
(out/"E_swap_probe.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n  n = {n}   elapsed {time.time()-t0:.0f}s")
print(f"  order-consistent (answer flipped) : {agree_flip}/{n} = {100*agree_flip/n:.1f}%")
print(f"  order-invariant  (same answer)    : {agree_same}/{n} = {100*agree_same/n:.1f}%  <- the prior")
print(f"  accuracy, original run            : {acc_A}/{n} = {100*acc_A/n:.2f}%")
print(f"  accuracy, order-debiased          : {acc_deb}/{n} = {100*acc_deb/n:.2f}%   "
      f"({100*(acc_deb-acc_A)/n:+.2f} pts)")
print(f"  accuracy on the consistent subset : {acc_consistent}/{n_consistent} = "
      f"{100*acc_consistent/max(n_consistent,1):.2f}%")
