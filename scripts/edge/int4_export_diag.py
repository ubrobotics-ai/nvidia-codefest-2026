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
        bad = [(n, tuple(m._buffers["weight"].shape), str(m._buffers["weight"].dtype))
               for n, m in model.named_modules()
               if isinstance(m, ModelOptAWQPrepackedLinear)
               and m._buffers.get("weight") is not None
               and m._buffers["weight"].dtype != torch.int8]
        AUDIT["bad_after"] = bad
        print(f"\n[audit] ModelOptAWQPrepackedLinear modules: {AUDIT['modules']}", flush=True)
        print(f"[audit] holding uint8 weights BEFORE repack: {AUDIT['uint8_before']}", flush=True)
        print(f"[audit] NOT int8 AFTER repack: {len(bad)}", flush=True)
        for n, shp, dt in bad[:5]:
            print(f"[audit]     {n}  {dt} {shp}", flush=True)
        if bad:
            pre = collections.Counter(k.split(".")[0] + "." + k.split(".")[1]
                                      for k in AUDIT["skipped_keys"] if "." in k)
            print(f"[audit] loader skipped {len(AUDIT['skipped_keys'])} keys; "
                  f"top prefixes: {pre.most_common(4)}", flush=True)
            for k in AUDIT["skipped_keys"][:5]:
                print(f"[audit]     skipped: {k}", flush=True)
            raise SystemExit(
                f"\nFAIL: {len(bad)} INT4 linears still hold non-int8 weights after repacking.\n"
                "The plugin nodes would be emitted around dense FP16 tensors and the ONNX would be\n"
                "unbuildable on the target -- which is exactly the 2026-09-10 defect. Aborting instead\n"
                "of writing another plausible-looking broken artefact.")
        print("[audit] OK: every INT4 linear carries int8 fragment weights\n", flush=True)

    repacking.apply_all_repacking = audited



def install_key_remap_fix():
    """Teach the Cosmos3-Edge remap about the ModelOpt-quantised layout.

    `_cosmos3_edge_llm_key_remap` was written for the *native* checkpoint, whose text tower is flat
    (`layers.N.*`, `embed_tokens.weight`, `norm.weight`) -- its docstring says so. A ModelOpt AWQ export nests the
    same tower under `model.language_model.*`, so the function's `startswith(("layers.", ...))` test is False,
    the key passes through unmapped, and `_set_tensor` then fails to bind it against a module tree that wants
    `model.layers.N.*`. The loader counts that as a skip and says nothing above DEBUG, which is how 471 tensors
    went missing without an error.
    """
    from tensorrt_edgellm.scripts import export as _ex
    original = _ex._cosmos3_edge_llm_key_remap
    NEST = "model.language_model."

    def fixed(key):
        if key.startswith(NEST):
            key = "model." + key[len(NEST):]
        return original(key)

    _ex._cosmos3_edge_llm_key_remap = fixed
    print("[fix] _cosmos3_edge_llm_key_remap wrapped: strips 'model.language_model.' -> 'model.'", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install_audit()
    if os.environ.get("EDGELLM_FIX_REMAP", "0") == "1":
        install_key_remap_fix()
    from tensorrt_edgellm.scripts.export import main
    sys.argv[0] = "tensorrt-edgellm-export"
    main()
