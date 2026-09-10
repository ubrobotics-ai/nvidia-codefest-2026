#!/usr/bin/env python3
"""Re-quantise Cosmos3-Edge with the multimodal projector excluded.

`build_quant_config` keeps the visual tower in FP16 by matching a fixed list of submodule prefixes
(`_VISUAL_PREFIXES`): visual, vision_tower, vision_model, multi_modal_projector, mlp1, image_embed, embed_vision.
Cosmos3-Edge names its projector plain **`projector`**, which matches none of them -- so `model.projector.*` gets
INT4-quantised along with the language model while `model.visual*` is correctly left alone.

Nothing complains at quantisation time. It surfaces two steps later, when the visual encoder export tries to load
`projector.linear_fc1` and finds a packed uint8 weight with half the rows it expects:

    RuntimeError: The size of tensor a (5760) must match the size of tensor b (11520)

The projector feeds the vision tower, so it belongs on the FP16 side of the line with it. This appends disable
rules for it (last-match-wins) and asserts afterwards that nothing under `projector.` carries a quantizer.
"""
import sys, os

PROJECTOR_PATTERNS = ("*projector*weight_quantizer", "*projector*input_quantizer")


def install_projector_exclusion():
    from tensorrt_edgellm.quantization import quantize as _q
    original = _q.build_quant_config

    def wrapped(*a, **kw):
        cfg = original(*a, **kw)
        if cfg.get("quant_cfg") is not None:
            for pat in PROJECTOR_PATTERNS:
                cfg["quant_cfg"].append({"quantizer_name": pat, "enable": False})
            print(f"[fix] excluded the projector from quantisation: {PROJECTOR_PATTERNS}", flush=True)
        return cfg

    _q.build_quant_config = wrapped


def install_audit():
    import torch
    from tensorrt_edgellm.quantization import quantize as _q
    original = _q._calibrate_multimodal

    def audited(model, batches):
        original(model, batches)
        bad = []
        for name, mod in model.named_modules():
            if "projector" not in name:
                continue
            wq = getattr(mod, "weight_quantizer", None)
            if wq is not None and not getattr(wq, "_disabled", True):
                bad.append(name)
        print(f"\n[audit] projector submodules with an ENABLED weight quantizer: {len(bad)}", flush=True)
        for n in bad[:5]:
            print(f"[audit]     {n}", flush=True)
        if bad:
            raise SystemExit(
                "\nFAIL: the projector is still being quantised. The visual encoder export will fail on a "
                "packed-vs-dense shape mismatch. Aborting rather than writing another checkpoint that only "
                "breaks two steps later.")
        print("[audit] OK: projector stays FP16 alongside the visual tower\n", flush=True)

    _q._calibrate_multimodal = audited


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    install_projector_exclusion()
    install_audit()
    from tensorrt_edgellm.scripts.quantize import main
    sys.argv[0] = "tensorrt-edgellm-quantize"
    main()
