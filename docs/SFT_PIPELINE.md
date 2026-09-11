# Runbook — SFT to redeployed engine, Cosmos3-Edge

Scopes the half the training document leaves out: what it takes to get a fine-tuned model
back into the artefact we actually deploy, and what has to be re-measured before any
published number is trusted again.

**Scope includes tool-calling.** The source document recommends against it; that is
overridden. The justification is its own diagnosis — schema (0 of 24) and tool calls
(1 of 10) are *one* defect, "it cannot emit structure". Training structured output while
excluding tools would be training the defect and withholding the cheapest case of it.

**Everything published today is a property of the current checkpoint.** An SFT changes the
weights, so the AWQ calibration is re-fitted, the engines are rebuilt, and L0 accuracy,
`k = 1.2148`, L1's 9–10 of 12 and the two model cards all become *unverified until
re-measured*. Budget the re-measurement, not just the training.

---

## Phase 0 — freeze the baseline

Before touching weights. Without this nothing afterwards is attributable.

```bash
git -C <codefest repo> rev-parse HEAD            # record
sha256sum quantized-int4-awq-v2/model.safetensors
sha256sum onnx-v3/llm/model.onnx.data
```

Record, from the current artefact: L0 `task_mean(d,lr)` and per-task accuracy; the fitted
`k` globally **and per range bucket**; L1 targets @ IoU 0.5 and false triggers; commands
13/15 with stops 5/5; tools 1/10 arguments; schema 0/24. These are the regression gates.

**Hold-out — never train on any of these.** The 24-item grounded set, the 60 frozen
frames, the lw-lab1 judge set, the 15-item command set, the two 20-case directional sets,
and the 1,442 L0 items including the 500 left_right. Generate fresh phrasings for
anything that overlaps; "Turn to starboard" and "Continue onward" are worth keeping honest.

## Phase 1 — data

| class | examples | why this weight |
|---|---|---|
| FORWARD phrasings | ~150 | weakest class, 7/10, and "Go forward" → BACK is an inversion |
| BACK | ~100 | include "withdraw", "pull out", "retreat" |
| LEFT / RIGHT | ~100 each | keep balanced; RIGHT must not be the minority |
| STOP | ~80 | already 5/5; this is the safety class and must not regress |
| SEARCH / REPORT | ~80 each | |
| **UNKNOWN** | **~200** | largest bucket: the model has *no* declining behaviour to reinforce |
| **tool calls** | **~250** | all ten tools, arguments in-enum, arguments exercised not just named |
| schema objects | ~120 | every field present, enums exact, **`task_relevance` on every entity** |

Two notes on the last two rows. Tools failed at *arguments* (1/10) rather than at
intent (3/10 correct command), so weight the data toward argument construction, not tool
selection. And `task_relevance` is the field omitted in 11 of 11 parseable outputs and the
one the tolerant parser cannot infer — it has to guess, and that guess is part of why
precision is 0.529. It is the single highest-value field in the schema bucket.

## Phase 2 — train

LoRA on the **bf16** source, not the quantised checkpoint. `peft` 0.20.0, `accelerate`
1.15.0 and `transformers` 5.16.1 are in the container; `trl`/`datasets` are not and are
not needed at this size.

Gate before proceeding: evaluate the adapter *unquantised* on the held-out command and
tool sets. If it does not clear the targets in Phase 6 at bf16, quantising will not
improve it — stop here rather than spending the pipeline.

## Phase 3 — merge and re-quantise

```bash
# merge the adapter, then AWQ with the SAME calibration set
export EDGELLM_QUANT_DATASET_MMMU=<calib_int4_mmmu_override.jsonl>   # the 198 frames
python scripts/l0_cosmos_int4_sim.py --arm int4 --model <merged> --out <dir> ...
```

Same 198 frames as the published checkpoint — a different calibration set changes the
artefact for a second reason and makes the SFT unattributable. The slice is
segment-disjoint from L1 and must stay so.

**Audit before going further.** Three export defects have bitten this pipeline already:

- quantise skipped nothing because `llm_int8_skip_modules` matches **anchored at the start
  of the full module path** — a bare `"visual"` matched nothing and quantised 333 modules
- the projector was not excluded
- `mlp.fc1/fc2` versus `up_proj/down_proj` naming produced 56 all-zero tensors

Assert, per module class: **dtype INT8, shape `fragmentRows(N,K) = divUp(N,128) × divUp(K,64) × 8`, and non-zero content.** Expect **169** quantised modules — 28 × 6 + `lm_head` — matching the AWQ layer set. Sampling one node is not enough: `n0_3` is an attention projection and was correct in every broken revision, and `onnx.checker` passes a graph with 56 all-zero tensors.

## Phase 4 — export ONNX

Use the audited wrapper (`int4_export_diag.py`, `int4_quant_fix.py`). Verify the export
reproduces the checkpoint byte-for-byte where it should: recomputing the cuteDSL fragment
layout from the packed `uint8` weights must match the ONNX initialisers exactly. Expect
169 `Int4GroupwiseGemmPluginV2` nodes in the LLM graph and none in the visual graph.

## Phase 5 — build engines

Both parts. They are different plans from the same ONNX.

```bash
# SM103 (bench):  ubrobotics-ai/edgellm-on-blackwell-ultra
SRC=<edge-llm> scripts/10_build_cutedsl.sh && scripts/20_build.sh && scripts/30_build_engines.sh
# SM87 (deploy):  llm_build + visual_build on the Orin
```

`-DENABLE_CUTE_DSL="fmha;int4_fp16_gemm"` — without the second group the INT4 plugin
compiles with its kernels disabled and `enqueue()` returns `-1` with no diagnostic.

## Phase 6 — re-measure, and the gates

| measurement | instrument | gate |
|---|---|---|
| commands, 15 items | shim, thinking **off** | ≥ 14/15, **stops 5/5** — hard fail otherwise |
| UNKNOWN | `"Sing me a song"` + fresh distractors | declines rather than picking |
| **tool calls, 10** | prose-described tools, 2 examples | **≥ 8/10 correct arguments** (baseline 1/10) |
| schema, 24 | grounded benchmark | `task_relevance` present ≥ 22/24 |
| **perception, 24 frames** | grounded benchmark | **≥ 9/12 targets, 0/14 false triggers — no regression** |
| **L0, 1,442 items** | `l0_trt_eval.py` + `l0_report.py` | **no significant loss vs frozen baseline (paired McNemar p > 0.05)** |
| distance `k` | `fit_k`, per range bucket | **re-fitted — do not carry 1.2148 across** |
| left/right prior | `l0_swap_probe.py` | record; not trained, so expect ~unchanged |

The two bold regression gates matter most: this SFT trains language behaviour, and the
model's *perception* is the thing that earned it the seat. A command win bought with a
detection loss is not a win.

## Phase 7 — re-publish

Nothing ships until Phase 6 passes. Then, in order: `REPORT_L0.md` regenerated (`k` is
new); both model cards (`Cosmos3-Edge-INT4-AWQ`, `Cosmos3-Edge-NF4-bnb` if the control is
re-cut); the calibration slice manifest if it changed; and the tracker rows. Every number
carrying over from the old checkpoint must be re-stated or removed — not left to age.

## What this does not cover

The left/right prior (order-invariance on 65.6% of items) is deliberately out of scope:
it is a perceptual prior, needs spatial rather than command data, and mixing it in makes
neither result attributable. The 500-item probe exists to measure it separately.

And if the tool gate fails at bf16 in Phase 2, that is the answer — the architecture is
removing tool calls from the loop anyway (`direct_execute`, `SweepPolicy`, Nav2), so a
failed tool SFT costs a training run, not the deployment.
