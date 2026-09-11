#!/usr/bin/env python3
"""Phase 4a — fold the LoRA adapter into the bf16 checkpoint, in the checkpoint's own layout.

Does NOT go through transformers. `save_pretrained` would rewrite all 698 tensors into a
fresh shard set and lose the layout `tensorrt-edgellm-quantize` reads (decoder weights live
in `transformer/`, the vision tower in `vision_encoder/`, addressed by a flat index).

It rewrites the `transformer/` shards in place instead of writing a side file and re-pointing
the index. The first version did the latter, and the quantiser silently ignored it: it
resolves weights through the `transformer/` directory, not the index, so it loaded the
un-finetuned shards and produced a checkpoint byte-identical to the baseline. The engine
built from it scored 48.6% -- the base model's number -- and the merge had simply not
happened. Whichever path the loader takes now, it sees merged weights.

Name mapping, adapter -> checkpoint (the same remap the ONNX path uses):
    self_attn.{q,k,v,o}_proj -> self_attn.to_{q,k,v,out}
    mlp.fc1 / mlp.fc2        -> mlp.up_proj / mlp.down_proj

Vision-tower modules are skipped with a check, not an assumption: the adapter targets them
by suffix, but the training data is text-only so the tower never ran and their lora_B is
still exactly zero. If that ever stops being true this aborts rather than silently dropping
a trained delta.
"""
import argparse, json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

MAP = {"self_attn.q_proj": "self_attn.to_q", "self_attn.k_proj": "self_attn.to_k",
       "self_attn.v_proj": "self_attn.to_v", "self_attn.o_proj": "self_attn.to_out",
       "mlp.fc1": "mlp.up_proj", "mlp.fc2": "mlp.down_proj"}
LINK_DIRS = ["transformer", "vision_encoder", "vae", "scheduler", "assets", "text_tokenizer"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    print(f"  r={r} alpha={alpha} rslora={bool(cfg.get('use_rslora'))} -> scale={scale}")

    sd = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    mods = sorted({k.rsplit(".lora_", 1)[0] for k in sd})

    deltas, skipped = {}, 0
    for m in mods:
        A, B = sd[m + ".lora_A.weight"], sd[m + ".lora_B.weight"]
        if ".visual." in m:
            if bool((B != 0).any()):
                sys.exit(f"  ABORT: visual module {m} has a trained delta this script drops")
            skipped += 1
            continue
        short = m.replace("base_model.model.model.language_model.", "")   # layers.N.<mod>
        head, tail = short.rsplit(".", 2)[0], ".".join(short.rsplit(".", 2)[1:])
        if tail not in MAP:
            sys.exit(f"  ABORT: no checkpoint name for {tail}")
        deltas[f"{head}.{MAP[tail]}.weight"] = (B.float() @ A.float()) * scale
    print(f"  language modules merged: {len(deltas)}   visual skipped (lora_B == 0): {skipped}")

    idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    missing = [k for k in deltas if k not in wmap]
    if missing:
        sys.exit(f"  ABORT: {len(missing)} merged names absent from the base index, e.g. {missing[:3]}")

    by_file = {}
    for k in deltas: by_file.setdefault(wmap[k], []).append(k)
    print(f"  merged tensors live in {len(by_file)} shard(s): {sorted(by_file)}")

    os.makedirs(a.out, exist_ok=True)
    # Everything the adapter did not touch is shared with the base by symlink. The shards
    # that DO hold merged tensors are rewritten as real files below.
    touched_dirs = {os.path.dirname(f) for f in by_file}
    for d in LINK_DIRS:
        s = os.path.join(a.base, d)
        if not os.path.exists(s) or d in touched_dirs: continue
        dst = os.path.join(a.out, d)
        if os.path.islink(dst): os.remove(dst)
        if not os.path.exists(dst): os.symlink(os.path.realpath(s), dst)
    for f in os.listdir(a.base):
        p = os.path.join(a.base, f)
        if os.path.isfile(p): shutil.copy2(p, os.path.join(a.out, f))

    maxrel, nwritten = 0.0, 0
    for fn, keys in sorted(by_file.items()):
        src = load_file(os.path.join(a.base, fn))
        for k in keys:
            W = src[k]
            Wm = (W.float() + deltas[k].to(W.device)).to(W.dtype)
            rel = float((Wm.float() - W.float()).norm() / W.float().norm())
            maxrel = max(maxrel, rel)
            src[k] = Wm; nwritten += 1
        dst = os.path.join(a.out, fn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        save_file(src, dst, metadata={"format": "pt"})
        print(f"    rewrote {fn}  ({len(keys)} of {len(src)} tensors merged)")
        del src
    print(f"  max relative weight change: {maxrel:.4%}")
    if maxrel == 0.0: sys.exit("  ABORT: merge changed nothing")

    # Any shard in a rewritten directory that held no merged tensor still has to exist.
    for d in sorted(touched_dirs):
        for f in os.listdir(os.path.join(a.base, d)):
            s, dst = os.path.join(a.base, d, f), os.path.join(a.out, d, f)
            if not os.path.exists(dst): shutil.copy2(s, dst)

    # Verify on disk, from the files the quantiser will actually open.
    bad = 0
    for fn, keys in sorted(by_file.items()):
        chk = load_file(os.path.join(a.out, fn))
        ref = load_file(os.path.join(a.base, fn))
        for k in keys:
            if torch.equal(chk[k], ref[k]): bad += 1
        del chk, ref
    if bad: sys.exit(f"  ABORT: {bad} merged tensors are still identical to the base on disk")
    print(f"  verified: all {nwritten} merged tensors differ from the base on disk")
    print(f"  -> {a.out}")

if __name__ == "__main__":
    main()
