#!/usr/bin/env python3
"""LoRA on the bf16 Cosmos3-Edge reasoner — command vocabulary and UNKNOWN.

Trains on bf16, not on a quantised checkpoint: the AWQ artefact is GPTQ-packed uint8 that
transformers cannot load, and NF4 is the measurement control rather than a deployment
target. L0 showed INT4-AWQ is statistically indistinguishable from bf16 once k is re-fitted
(p = 0.931), which is the evidence that quantising AFTER training will not eat the adapter.

PILOT SCOPE. The system prompt below is a placeholder. L1 measured the prompt as
load-bearing for this model -- a bare list scores 9/15 where a conversational prompt with
worked examples scores 13/15 -- so the production prompt from prompts.py must replace it
before any run whose numbers are meant to transfer. Training under one prompt and serving
under another measures nothing useful.
"""
import argparse, json, os, random, sys
import torch

PROMPT = ("You control a ground robot. Map the operator's instruction to exactly one "
          "action: FORWARD, BACK, LEFT, RIGHT, STOP, SEARCH, REPORT, or UNKNOWN. "
          "Answer UNKNOWN if the instruction is not one of these actions. "
          "Reply with the action word only.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("WORK","")+"/edge/Cosmos3-Edge")
    ap.add_argument("--data", default="sft/data/commands.jsonl")
    ap.add_argument("--out", default="sft/adapter")
    ap.add_argument("--dev-frac", type=float, default=0.20, help="fraction of SEEDS held out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke-test cap")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data)]

    # Split by SEED PHRASE, not by row. The rows are decorated variants of ~109 seeds
    # ("advance" -> "please advance now"), so a row-level split puts every dev seed in
    # training and measures memorised decorations: it scored 98.9% that way against a
    # 48.3% base, which is not a generalisation result. Holding out whole seeds asks the
    # question that matters -- does it handle a phrasing it has never seen?
    MODS = ["please ", "can you ", "now ", "quickly ", "robot, ", "okay ", "right, "]
    SUFF = [" please", " now", " right away", "!", ".", " immediately"]
    def seed_of(s):
        t = s.lower()
        for m in MODS:
            if t.startswith(m): t = t[len(m):]
        for x in SUFF:
            if t.endswith(x): t = t[:-len(x)]
        return t.strip()
    by_seed = {}
    for r in rows: by_seed.setdefault(seed_of(r["instruction"]), []).append(r)
    per_label = {}
    for s, rs in by_seed.items(): per_label.setdefault(rs[0]["label"], []).append(s)
    dev_seeds = set()
    for lab, seeds in per_label.items():       # hold out seeds from every class
        seeds = sorted(seeds); random.shuffle(seeds)
        k = max(1, int(round(len(seeds) * a.dev_frac)))
        dev_seeds.update(seeds[:k])
    dev  = [r for s in dev_seeds for r in by_seed[s]]
    train = [r for s, rs in by_seed.items() if s not in dev_seeds for r in rs]
    random.shuffle(train); random.shuffle(dev)
    assert not ({seed_of(r["instruction"]) for r in train} & dev_seeds), "seed leak"
    print(f"  train {len(train)} rows / {len(by_seed)-len(dev_seeds)} seeds   "
          f"dev {len(dev)} rows / {len(dev_seeds)} seeds  (seed-disjoint)", flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", proc)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.config.use_cache = False

    # peft's LoRA dispatcher probes torchao and raises on the container's 0.12.0+git.
    # That dispatcher only matters for torchao-quantised bases; this one is bf16, so
    # reporting it unavailable is correct rather than a workaround.
    import peft.import_utils as _piu
    _piu.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as _plt
        _plt.is_torchao_available = lambda: False
    except Exception:
        pass
    from peft import LoraConfig, get_peft_model
    # Anchored to the language model on purpose. A bare suffix list ("q_proj", "fc1", ...)
    # also matches model.visual.encoder.layers.*, and the first run adapted 135 vision-tower
    # modules as well as the 168 language ones. That run was harmless only by accident --
    # the data is text-only, so the tower never executed and its lora_B stayed exactly zero
    # -- but with any image in the data it would have quietly retrained the perception stack
    # that L0 and L1 measure.
    targets = r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.fc[12])$"
    cfg = LoraConfig(r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={a.rank} on {targets}", flush=True)
    print(f"  trainable {trainable/1e6:.2f}M / {total/1e9:.2f}B = {100*trainable/total:.3f}%", flush=True)

    def encode(batch):
        texts, labels = [], []
        for r in batch:
            msgs = [{"role":"system","content":[{"type":"text","text":PROMPT}]},
                    {"role":"user","content":[{"type":"text","text":r["instruction"]}]}]
            try:
                p = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
            except TypeError:
                p = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            texts.append(p); labels.append(r["label"])
        full = [t + l + tok.eos_token for t, l in zip(texts, labels)]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=512)
        lab = enc["input_ids"].clone()
        # supervise the answer only, not the prompt
        for i, t in enumerate(texts):
            plen = len(tok(t, truncation=True, max_length=512)["input_ids"])
            lab[i, :plen] = -100
        lab[enc["attention_mask"] == 0] = -100
        enc["labels"] = lab
        return {k: v.to("cuda") for k, v in enc.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps_per_epoch = max(1, len(train) // a.batch)
    total_steps = a.max_steps or int(steps_per_epoch * a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps,
                                                pct_start=0.1)
    print(f"  {total_steps} steps  (batch {a.batch}, lr {a.lr})", flush=True)
    model.train(); step = 0
    while step < total_steps:
        random.shuffle(train)
        for i in range(0, len(train) - a.batch + 1, a.batch):
            if step >= total_steps: break
            out = model(**encode(train[i:i+a.batch]))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"    step {step}/{total_steps}  loss {out.loss.item():.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    json.dump({"dev": dev}, open(os.path.join(a.out, "dev_split.json"), "w"))
    print(f"  adapter -> {a.out}  (dev split saved beside it)", flush=True)

if __name__ == "__main__":
    main()
