#!/usr/bin/env python3
"""Re-run the Cosmos3-Edge ONNX export with the silent failures made loud.

Three things went wrong quietly in the 2026-09-10 export and produced a 3.36 GB "INT4" artefact with zero INT8
tensors in it:

  1. `checkpoint.loader` skipped 471 of 698 tensors. Every skip is logged at DEBUG ("Key not found in model"),
     so at INFO the run looked clean. -> DEBUG is captured here and the skipped keys are summarised.
  2. `_cast_modelopt_awq_prepacked()` repacks the weight only `if w.dtype == torch.uint8`. When the loader has
     not populated the buffer, that guard is False, the function silently no-ops on the weight while still
     transposing and casting the scale -- which is why the graph ends up with FLOAT16 [2048, 2048] weights beside
     correctly shaped FLOAT16 [16, 2048] scales. -> asserted after repacking.
  3. The exporter then emits Int4GroupwiseGemmPluginV2 nodes around those dense FP16 weights without complaint.

Usage mirrors the real CLI:
  python int4_export_diag.py <quantized_ckpt_dir> <onnx_out_dir> [extra export args...]
"""
import collections, logging, sys, os

AUDIT = {"modules": 0, "uint8_before": 0, "bad_after": [], "skipped_keys": []}


class _SkipCollector(logging.Handler):
    """The loader announces every dropped tensor at DEBUG and nowhere else."""

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "Key not found in model" in msg or "Key not found in .bin shard" in msg:
            AUDIT["skipped_keys"].append(msg.split(": ", 1)[-1])


def install_audit():
    import torch
    from tensorrt_edgellm.checkpoint import repacking
    from tensorrt_edgellm.models.linear import ModelOptAWQPrepackedLinear

    loader_log = logging.getLogger("tensorrt_edgellm.checkpoint.loader")
    loader_log.setLevel(logging.DEBUG)
    loader_log.addHandler(_SkipCollector())

    original = repacking.apply_all_repacking

    def audited(model):
        mods = [m for m in model.modules() if isinstance(m, ModelOptAWQPrepackedLinear)]
        AUDIT["modules"] = len(mods)
        AUDIT["uint8_before"] = sum(
            1 for m in mods if getattr(m._buffers.get("weight"), "dtype", None) == torch.uint8)
        original(model)
        # dtype alone is not enough: 56 MLP tensors previously passed this check while being all-zero
        # buffers at the packed shape. Check dtype, the cuteDSL fragment shape (K dim == 512), and content.
        bad = []
        for n, mod in model.named_modules():
            if not isinstance(mod, ModelOptAWQPrepackedLinear): continue
            w = mod._buffers.get("weight")
            if w is None: continue
            why = []
            if w.dtype != torch.int8: why.append(f"dtype={w.dtype}")
            if w.dim() == 2 and w.shape[1] != 512: why.append(f"not fragment shape {tuple(w.shape)}")
            if int(w.abs().sum()) == 0: why.append("ALL ZERO")
            if why: bad.append((n, tuple(w.shape), " · ".join(why)))
        AUDIT["bad_after"] = bad
        print(f"\n[audit] ModelOptAWQPrepackedLinear modules: {AUDIT['modules']}", flush=True)
        print(f"[audit] holding uint8 weights BEFORE repack: {AUDIT['uint8_before']}", flush=True)
        print(f"[audit] FAILING (dtype / shape / all-zero): {len(bad)}", flush=True)
        for n, shp, why in bad[:6]:
            print(f"[audit]     {n}  {shp}  <- {why}", flush=True)
        if bad:
            pre = collections.Counter(k.split(".")[0] + "." + k.split(".")[1]
                                      for k in AUDIT["skipped_keys"] if "." in k)
            print(f"[audit] loader skipped {len(AUDIT['skipped_keys'])} keys; "
                  f"top prefixes: {pre.most_common(4)}", flush=True)
            for k in AUDIT["skipped_keys"][:5]:
                print(f"[audit]     skipped: {k}", flush=True)
            raise SystemExit(
                f"\nFAIL: {len(bad)} INT4 linears are not correctly packed after repacking.\n"
                "The plugin nodes would be emitted around dense FP16 tensors and the ONNX would be\n"
                "unbuildable on the target -- which is exactly the 2026-09-10 defect. Aborting instead\n"
                "of writing another plausible-looking broken artefact.")
        print("[audit] OK: every INT4 linear carries int8 fragment weights\n", flush=True)

    repacking.apply_all_repacking = audited



def install_key_remap_fix():
    """Add the MLP rename the Cosmos3-Edge remap is missing.

    `_cosmos3_edge_llm_key_remap` renames the four attention projections from the checkpoint's Qwen-VL style
    (`to_q/to_k/to_v/to_out`) onto the module tree's `q_proj/k_proj/v_proj/o_proj`. It renames nothing in the MLP —
    but the checkpoint stores `mlp.fc1` / `mlp.fc2` while `modeling_und_prefill.py` builds `mlp.up_proj` /
    `mlp.down_proj`. So all 56 MLP weights (2 per layer x 28) fail to bind, silently, and the modules keep the
    zero-filled INT8 buffers they were constructed with.

    The result passes a dtype check and an `onnx.checker` run: the graph carries 169 INT4 plugin nodes and 169
    INT8 tensors, of which 56 are **entirely zero** and still at the packed shape rather than the cuteDSL
    fragment shape. Every MLP in the decoder is a zero matrix.

    The visual tower also uses `mlp.fc1/fc2`, but those keys are dropped earlier by the `visual.` filter, so this
    rename cannot touch them.
    """
    from tensorrt_edgellm.scripts import export as _ex
    original = _ex._cosmos3_edge_llm_key_remap

    def fixed(key):
        if "visual." not in key and "projector." not in key:
            key = key.replace(".mlp.fc1.", ".mlp.up_proj.").replace(".mlp.fc2.", ".mlp.down_proj.")
        return original(key)

    _ex._cosmos3_edge_llm_key_remap = fixed
    print("[fix] remap extended: mlp.fc1 -> mlp.up_proj, mlp.fc2 -> mlp.down_proj", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install_audit()
    if os.environ.get("EDGELLM_FIX_REMAP", "1") == "1":
        install_key_remap_fix()
    from tensorrt_edgellm.scripts.export import main
    sys.argv[0] = "tensorrt-edgellm-export"
    main()
