# Index: L0 spatial tests (model x precision, and the SFT gate)

Snapshot of 2026-09-17. The sources of record are `reports/REPORT_L0.md` (the 7-row
matrix) and `sft/README.md` (the SFT gate and its seeds). This page is an index. If a number
here disagrees with its source, the source is correct.

## What L0 measures

- Closed spatial questions on rendered frames with numbered region outlines.
- Three tasks: `distance` (N = 486), `left_right` (N = 500) and `mcq` (N = 456). For `mcq`,
  answers are region indices, not option letters.
- Every row uses the same items, the same prompt, the same parser and greedy decoding, with
  `enable_thinking=False` unless a row says otherwise.
- Statistics: exact two-sided paired McNemar on discordant pairs. `b` counts items right at
  the reference and wrong at the candidate; `c` counts the reverse.
- `task_mean(d,lr)` is the unweighted mean of distance and left_right, and excludes mcq.
  Report mcq beside it.
- All rows ran on a B300 (SM103). There is **no Orin accuracy claim** until the
  engine-equivalence gate has run.

## Test inventory

| id | what | source |
|---|---|---|
| L0-R1 | Cosmos3-Edge bf16, round 1 | REPORT_L0 "The 7 rows" |
| L0-C | Cosmos3-Edge bf16, L0 control | same |
| L0-AWQsim | Cosmos3-Edge INT4 AWQ, simulated (fake-quant) | same |
| L0-AWQtrt | Cosmos3-Edge INT4 AWQ, real TensorRT Edge-LLM kernels (the deployed path) | same |
| L0-NF4 | Cosmos3-Edge NF4, bitsandbytes real kernels | same |
| L0-G | Gemma-4-E4B-it bf16, round 1 | same |
| L0-GQ | Gemma-4-E4B-it QAT Q4_0, llama.cpp | same |
| L0-k | Distance calibration constant `k`, re-fitted per arm | REPORT_L0 "Distance calibration constant k" |
| L0-think | Thinking-enabled subsample | REPORT_L0 "Thinking-enabled subsample" |
| G-v2 | SFT adapter v2 (placeholder prompt), INT4 | sft/README.md |
| G-v3 | SFT adapter v3 (production prompt, seed 0), INT4 | sft/README.md |
| G-v4 | SFT adapter v4 (production prompt, seed 1), INT4 | sft/README.md, section starting at line 622 |

## L0 matrix (raw, before re-fitting k)

| row | distance | left_right | mcq | task_mean(d,lr) |
|---|---|---|---|---|
| Cosmos3-Edge bf16, round 1 | 29.01% | 65.20% | 17.54% | 47.11% |
| Cosmos3-Edge bf16, L0 control | 29.63% | 64.80% | 17.76% | 47.21% |
| Cosmos3-Edge INT4 AWQ, simulated | 21.81% | 67.20% | 16.67% | 44.51% |
| **Cosmos3-Edge INT4 AWQ, TensorRT (deployed)** | **18.52%** | 64.00% | 16.45% | 41.26% |
| Cosmos3-Edge NF4 | 30.25% | 64.60% | 17.54% | 47.42% |
| Gemma-4-E4B-it bf16 | 16.26% | 51.00% | 12.50% | 33.63% |
| Gemma-4-E4B-it Q4_0 | 14.20% | 49.80% | 11.84% | 32.00% |

## Paired tests, bf16 against 4-bit

| comparison | task | b | c | p | verdict |
|---|---|---|---|---|---|
| bf16 to AWQ simulated | distance | 68 | 30 | 0.00016 | real |
| bf16 to AWQ TensorRT | distance | 73 | 19 | < 0.0001 | real |
| bf16 to AWQ TensorRT | left_right | 38 | 34 | 0.724 | churn |
| bf16 to AWQ TensorRT | mcq | 25 | 19 | 0.451 | churn |
| bf16 to AWQ TensorRT, **k re-fitted per arm** | distance | 68 | 66 | **0.931** | loss removed |
| bf16 to NF4, raw | distance | — | — | 0.815 | no re-fit needed |
| NF4 to AWQ, both re-fitted | distance | — | — | 0.804 | indistinguishable |
| NF4 to AWQ, raw | distance | 21 | 78 | < 0.00001 | the scale differs |

**Finding.** AWQ 4-bit shifts the distance *scale* without destroying the information, and
a single re-fitted constant restores it. NF4 does not shift the scale at all.

Fitted `k`: bf16 1.1546, NF4 1.1628, AWQ 1.2148. The provisional deployment value is 1.21,
**but the far bucket needs 1.34**. `k` is set per engine build and per range bucket, not per
model.

## SFT gate (INT4 engines, paired against INT4 base)

| arm | k | distance | left_right | mcq | far [12 m, +) |
|---|---:|---:|---:|---:|---:|
| INT4 base | 1.2148 | 36.21% | 64.00% | 16.45% | 39.3% |
| G-v2, placeholder prompt | 1.2234 | 30.45% | 64.40% | 17.76% | 23.0% |
| G-v3, production prompt, seed 0 | 1.2154 | 34.57% | 66.00% | 16.67% | 37.7% |
| G-v4, production prompt, seed 1 | 1.2109 | 36.42% | 67.60% | 17.54% | 45.9% |

| arm | distance p | left_right p | mcq p |
|---|---|---|---|
| G-v2 | **0.0018** | 0.824 | 0.362 |
| G-v3 | 0.428 | 0.143 | 1.0 |
| G-v4 | 1.0 | **0.0021** | 0.383 |

- **G-v2's distance regression did not reproduce.** Two production-prompt seeds are flat.
  G-v2 was measured under the placeholder prompt, which by itself moves the base model from
  51.4% to 67.5%.
- G-v4's left_right gain survives Bonferroni correction, but the swap probe says only about
  a third of it is real.
- The command-path results for the same adapters are in `sft/README.md`: seed 0 goes from
  67.5% to 97.0% (bf16) and from 63.3% to 95.8% (INT4); seed 1 goes from 67.7% to 97.5%.

## Operational rules that follow

- Ship INT4-AWQ.
- Re-fit distance `k` per engine build and per range bucket. Never carry it across builds.
- Use thinking on for grounding and operator commands, and off for closed spatial queries.
  This matches the configuration that was measured; it has not been isolated as a cause.
- Any adapter must pass this gate on the **deployed-precision** engine under the
  **production prompt** before release.
