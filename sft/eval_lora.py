#!/usr/bin/env python3
"""Score the base model and the LoRA on the held-out dev slice.

This measures whether the adapter learned the generated distribution. It is NOT the
Phase 6 gate: the real command set (15 items), the tool set (10) and the grounded
benchmark (24 frames) live on the robot. Treat a win here as necessary, not sufficient.
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import SYSTEM as PROMPT, user_turn  # production prompt, brain/prompts.py:74-96
LABELS = ["FORWARD","BACK","LEFT","RIGHT","STOP","SEARCH","REPORT","UNKNOWN"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("WORK","")+"/edge/Cosmos3-Edge")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    a = ap.parse_args()

    dev = json.load(open(a.dev))["dev"]
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", proc)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if a.adapter:
        import peft.import_utils as _piu; _piu.is_torchao_available = lambda: False
        try:
            import peft.tuners.lora.torchao as _plt; _plt.is_torchao_available = lambda: False
        except Exception: pass
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    def predict(instr):
        msgs = [{"role":"system","content":[{"type":"text","text":PROMPT}]},
                {"role":"user","content":[{"type":"text","text":user_turn(instr)}]}]
        try:
            p = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        except TypeError as ex:
            # Do NOT silently fall back. Without enable_thinking the template emits
            # "<think>\n" instead of "<think></think>", i.e. thinking ON -- and every number
            # here is a thinking-off number. A reply that reasons aloud over two candidate
            # actions also trips the production commitment gate into UNKNOWN, so a silent
            # flip would penalise whichever arm reasoned more and inflate the delta.
            raise SystemExit(f"chat template rejected enable_thinking, refusing to guess: {ex}")
        enc = tok(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        txt = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        up = txt.strip().upper()
        # NOTE: this scans LABELS order, not text order, so a reply naming two actions
        # resolves to whichever appears earlier in LABELS. That differs from the production
        # commitment gate, which returns UNKNOWN when two distinct actions are named. Both
        # arms are scored by this identical rule, so the comparison is fair, but a number
        # here is not a prediction of what command_intent.py would do with the same text.
        for L in LABELS:
            if up.startswith(L): return L, txt.strip()
        for L in LABELS:
            if L in up: return L, txt.strip()
        return None, txt.strip()

    from collections import Counter, defaultdict
    ok = 0; per = defaultdict(lambda: [0,0]); unparsed = 0
    for r in dev:
        pred, raw = predict(r["instruction"])
        gt = r["label"]; per[gt][1] += 1
        if pred is None: unparsed += 1
        if pred == gt: ok += 1; per[gt][0] += 1
    tag = "LoRA" if a.adapter else "base"
    print(f"  {tag:5s} overall {ok}/{len(dev)} = {100*ok/len(dev):.1f}%   unparsed {unparsed}")
    for L in LABELS:
        if per[L][1]:
            print(f"    {L:8s} {per[L][0]:3d}/{per[L][1]:<3d} = {100*per[L][0]/per[L][1]:5.1f}%")

if __name__ == "__main__":
    main()
