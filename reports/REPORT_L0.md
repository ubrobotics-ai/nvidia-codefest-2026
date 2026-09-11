# L0 — model x precision, 7 rows on one machine

Every accuracy below was produced on the same host, over the same items, with the same prompt, the same
rendered numbered region outlines and the same parser (including the mcq fix: answers are **region
indices**, not option letters). Greedy decode, `enable_thinking=False` unless a row says otherwise.

## Decision

- **INT4-AWQ ships.** It is the deployed quantisation; nothing here argues against it.
- **Distance k is fitted per engine build *and per range bucket*, provisionally 1.21.** Not a model
  constant, and not one number: the far bucket needs 1.34 where the global fit gives 1.21.
- **Thinking on for grounding *and operator commands*, off for closed spatial queries.** L1 measured
  commands at 14/15 **with the conversational prompt and thinking on** — the measured configuration —
  against 3/14 with a bare list and thinking off. Those runs changed prompt *and* thinking mode
  together, so thinking's own contribution on the command path is not isolated; the rule follows the
  configuration that was measured, not a demonstrated cause. Per-request flag; no global choice.
- **No Orin accuracy claim until the engine-equivalence gate has run.** These rows are SM103.

## The finding

**4-bit weights shift the distance *scale*; they do not destroy the information, and one constant
restores it.** On the deployed TensorRT INT4-AWQ engine, distance reads 29.63% at bf16 against
18.52% at INT4 — a 11.1-point loss, paired McNemar b=73 c=19, p < 0.0001. Re-fit the
distance constant per arm and that loss disappears entirely: **b = 68, c = 66, p = 0.931**
(**in-sample** discordant pairs, not accuracies; the same in-sample scoring puts bf16 at 178 and the
TensorRT arm at 176 of 486. Held out and split-half, the k table below reads 174 against 167). left_right
(p = 0.724) and mcq (p = 0.451) never moved.

**NF4 does not shift the scale; AWQ does.** NF4 fits k = 1.1628 against bf16's 1.1546 and needs no
re-fit to match it — raw distance 30.25% vs 29.63%, p = 0.815. AWQ fits 1.2148. So the two 4-bit
schemes are indistinguishable only *after* each is re-fitted (p = 0.804); raw they are not
(b = 21, c = 78, p < 0.00001). The scheme decides whether a constant is needed, not whether 4-bit works.

### What to do with it

**Provisional k = 1.2148**, fitted on the **SM103** engine — not yet the number to bake. The constant
is a property of the quantised artefact (bf16 1.1546, NF4 1.1628, AWQ 1.2148), so it is fitted per
artefact, **including per engine build**. The deployed constant is the one fitted on the Orin engine's
own distance outputs, from the engine-equivalence gate run below; bake that, not this. Measured on
these items (**in-sample**, each arm's own k, scored over all 486 by the same rule the McNemar rows
use): the INT4 answers score
36.21% (176/486) under their own constant and 33.33%
(162/486) under the bf16 one —
**2.88 points (14 items) thrown away for no reason.**

That figure is smaller than the 11.1-point raw gap above, and the difference matters: almost
all of the raw gap is removed by calibrating *at all* (18.52%
uncalibrated to 36.21% calibrated, both on the scorer's basis), and
only the 2.88-point remainder is the cost of using the *wrong* constant. Do not quote
the raw gap as the cost of mis-calibration.

**The Friday SFT decision: L0 contributes no SFT target.** That is the finding, not a gap in it.

This report rules two candidates out and downgrades a third:

- **Precision is not the problem.** 4-bit costs nothing once the arm is calibrated (p = 0.931), so
  there is nothing for training to repair there.
- **Format is not the target.** The tolerant parser recovers the boxes (precision 0.529, recall 0.750)
  without valid JSON ever being emitted, and L1's five contract rewrites each scored 0/12 detections
  while improving format. Under the one-model architecture nothing asks the model for a contract.
- **The left prior is a confidence signal, not a training target.** The order-swap probe below shows
  65.6% order-invariance: on two thirds of items the model returns the same word for a question and
  its mirror. That is the absence of a spatial comparison, and a few hundred supervised examples do
  not create one. Order-consistency is worth having as a *signal* (70.35% vs 60.7%); the 72/28 answer
  split is not worth training against.

So the SFT budget belongs with **L1's candidates**, all on the language side: the **tool-calling
re-run** (0/21, but August, a different serving stack, a bare schema, and unmeasured under the prompt
that moved commands from 3/14 to 14/15), the missing **UNKNOWN path** (`"Sing me a song"` returns
`BACK` rather than declining), and **PT-PT** if it returns to scope. L1 has a measured fallback for
the last two — a text-only Gemma E2B at 14/14 including Portuguese, 2,591 MB — so training is not the
only route to any of them.

**Open-vocabulary precision (0.529) stays on the engineering side, not the training side.** It is the
perception limit to design around — gate approaches on a second confirmation, and treat a box as "go
and look" rather than "found" — not something to fine-tune away on 24 items.

### Provenance of the TensorRT row — read before quoting it

This row was produced on a **DGX B300 (SM103)**, which cannot build the Edge-LLM INT4 path at all on
v0.10.1: it needs the SM allowlist patch in NVIDIA/TensorRT-Edge-LLM#207 plus the NVRTC include-path
fix in #205, **both unmerged at the time of writing**. The row therefore exists under a local patch set.

It shares the **ONNX export** with the Jetson deployment, not the engine. SM103 and SM87 produce
different plans from the same export, so "the same artefact the Jetson deploys" is true of the ONNX
and false of the engine. That makes an engine-equivalence gate runnable — B300 outputs on all **1,442**
scored items (986 distance + left_right, 456 mcq, which is why the criterion below covers all three
tasks) against Orin outputs on a 200-pair subsample, compared per item — and **it has not been run.**
Until it is, this row licenses claims about INT4-AWQ *as a quantisation*, not about the Orin engine.

**What passing means.** The gate cannot pass or fail without a criterion, and the hand-off's
">= 99% per-item agreement" cannot be used as written: this report's own environment floor is 87.9%
on distance between two containers running the *same* model, so 99% is unreachable by construction.
The criterion is therefore stated against the floor:

| check | threshold |
|---|---|
| per-item agreement, distance | not below the floor, **>= ~88%** |
| per-item agreement, left_right and mcq | **>= ~99%** (the floor is ~95-99% on the discrete tasks) |
| paired McNemar, each task, on the 200 pairs | **p > 0.05** — no detectable systematic difference |
| k re-fitted on the Orin distance outputs | fills the blank row above, **per range bucket** |

Agreement below the floor on a discrete task means the two engines differ; agreement above it means
they are indistinguishable *at the resolution this benchmark has*, which is the strongest claim the
instrument supports. A McNemar p > 0.05 is the accompanying check that the differences are unsigned
churn rather than one engine being consistently worse.

## Naming, fixed

Round 1 reported a **"weighted"** figure. It was never N-weighted: it is the *unweighted mean of the two
task accuracies*, `(distance + left_right) / 2`, and it **excludes mcq** — the task both models are worst
at. It is called **`task_mean(distance, left_right)`** from here on, and the mcq column is always shown
beside it. Where a single headline number is wanted, use `task_mean(distance, left_right, mcq)`, which is
given as a separate column.

## The 7 rows

The brief was a 2x2 (two models x two precisions). It has since grown a round-1 control, a
second 4-bit quantiser (NF4), and a row run on the *deployed* TensorRT engine rather than a
fake-quant stand-in. Every row is scored by the dataset's own scorer on the same items.

| model | precision | runtime | quantisation | kernels | distance | left_right | mcq | task_mean(d,lr) | task_mean(d,lr,mcq) |
|---|---|---|---|---|---|---|---|---|---|
| Cosmos3-Edge bf16 (round 1) | bf16 | transformers | none | n/a — full precision | 141/486 = 29.01% | 326/500 = 65.20% | 80/456 = 17.54% | **47.11%** | 37.25% |
| Cosmos3-Edge bf16 (L0 control) | bf16 | transformers | none | n/a — full precision | 144/486 = 29.63% | 324/500 = 64.80% | 81/456 = 17.76% | **47.21%** | 37.40% |
| Cosmos3-Edge INT4 AWQ (simulated) | INT4 AWQ | transformers fake-quant (simulated INT4) | PTQ (modelopt AWQ), SIMULATED -- not engine kernels | simulated | 106/486 = 21.81% | 336/500 = 67.20% | 76/456 = 16.67% | **44.51%** | 35.23% |
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | INT4 AWQ | tensorrt-edgellm | PTQ (modelopt AWQ), REAL INT4 kernels via the Edge-LLM ONNX export -- same ONNX export as the Jetson; engine built for SM103 under the #205 + #207 patches | real (TensorRT Int4GroupwiseGemmPluginV2) | 90/486 = 18.52% | 320/500 = 64.00% | 75/456 = 16.45% | **41.26%** | 32.99% |
| Cosmos3-Edge NF4 (real kernels) | NF4 | transformers + bitsandbytes NF4 (REAL 4-bit kernels) | PTQ (bitsandbytes NF4), real kernels -- a DIFFERENT quantiser from the shipped AWQ checkpoint | real (bitsandbytes) | 147/486 = 30.25% | 323/500 = 64.60% | 80/456 = 17.54% | **47.42%** | 37.46% |
| Gemma-4-E4B-it bf16 (round 1) | bf16 | transformers | none | n/a — full precision | 79/486 = 16.26% | 255/500 = 51.00% | 57/456 = 12.50% | **33.63%** | 26.59% |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | Q4_0 | llama.cpp (llama-server, CUDA) | QAT (upstream Google), packed uniform Q4_0 by Unsloth -- the UD-Q4_K_XL filename is a misnomer: 100% of parameters are stored Q4_0, no K-quants | real (llama.cpp) | 69/486 = 14.20% | 249/500 = 49.80% | 54/456 = 11.84% | **32.00%** | 25.28% |

N per cell is printed in the cell. distance N=486, left_right N=500, mcq N=456.

### Throughput and memory — B300, not Orin

| row | tok/s | peak GB | note |
|---|---|---|---|
| Cosmos3-Edge bf16 (round 1) | ~~43.94~~ withdrawn | 5.1 | B300 |
| Cosmos3-Edge bf16 (L0 control) | 76.17 | 5.1 | B300 -- NOT Orin; deployment figures come from the Jetson side |
| Cosmos3-Edge INT4 AWQ (simulated) | 24.44 | 5.6 | B300 -- NOT Orin; deployment figures come from the Jetson side |
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | — | — | B300 |
| Cosmos3-Edge NF4 (real kernels) | 58.96 | 2.6 | B300 -- NOT Orin; deployment figures come from the Jetson side |
| Gemma-4-E4B-it bf16 (round 1) | 18.76 | 16.0 | B300 |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | 53.22 | — | B300 -- NOT Orin; deployment figures come from the Jetson side |

**These columns describe a B300.** They are not deployment figures and must never be mixed into a column
that reports Orin latency. The simulated-INT4 row in particular runs *slower* than bf16 because fake quant
dequantises on every matmul — it says nothing about what an INT4 engine would do.

**The round-1 43.94 tok/s figure is withdrawn.** The same model at the same precision on the same B300
measured 76.17 tok/s one row below, in a different container — 1.7x apart from Python-side overhead alone.
Neither is a deployment number. The Orin figures are the ones that count: **52.8 tok/s decode**, and for
memory, **~4,830 MB peak for the full VLM**. L1 also reports 3,377 MB for the AWQ/TensorRT arm; the two
figures differ by close to the visual engine plus the embedding, but L1 labels both as system-wide
sampling and is reconciling them, so **quote the scope alongside whichever number you use** rather than
treating the difference as settled.

Blank cells are honest blanks: the TensorRT arm is driven by `llm_inference` as a subprocess and the
harness records wall-clock, not generated-token counts, and the llama.cpp arm reports no peak-memory
figure through its server API. Neither blank is a deployment claim.

## Points lost to 4-bit

Positive = **worse** at 4 bits. Read every row against the McNemar table below: most of these
differences are churn, not loss.

| model | task | bf16 | 4-bit | points lost (+ = worse at 4-bit) |
|---|---|---|---|---|
| Cosmos3-Edge / AWQ simulated | distance | 29.63% | 21.81% | +7.82 |
| Cosmos3-Edge / AWQ simulated | left_right | 64.80% | 67.20% | -2.40 |
| Cosmos3-Edge / AWQ simulated | mcq | 17.76% | 16.67% | +1.09 |
| Cosmos3-Edge / AWQ simulated | **task_mean(d,lr)** | 47.21% | 44.51% | +2.70 |
| Cosmos3-Edge / AWQ REAL (TensorRT) | distance | 29.63% | 18.52% | +11.11 |
| Cosmos3-Edge / AWQ REAL (TensorRT) | left_right | 64.80% | 64.00% | +0.80 |
| Cosmos3-Edge / AWQ REAL (TensorRT) | mcq | 17.76% | 16.45% | +1.31 |
| Cosmos3-Edge / AWQ REAL (TensorRT) | **task_mean(d,lr)** | 47.21% | 41.26% | +5.95 |
| Cosmos3-Edge / NF4 | distance | 29.63% | 30.25% | -0.62 |
| Cosmos3-Edge / NF4 | left_right | 64.80% | 64.60% | +0.20 |
| Cosmos3-Edge / NF4 | mcq | 17.76% | 17.54% | +0.22 |
| Cosmos3-Edge / NF4 | **task_mean(d,lr)** | 47.21% | 47.42% | -0.21 |
| Gemma-4-E4B-it | distance | 16.26% | 14.20% | +2.06 |
| Gemma-4-E4B-it | left_right | 51.00% | 49.80% | +1.20 |
| Gemma-4-E4B-it | mcq | 12.50% | 11.84% | +0.66 |
| Gemma-4-E4B-it | **task_mean(d,lr)** | 33.63% | 32.00% | +1.63 |

## Are the 4-bit deltas real? Paired McNemar

The two precisions disagree on half the items, so an unpaired difference of a few points is not evidence
of anything. Only discordant pairs carry information: **b** = right at bf16 and wrong at 4-bit, **c** =
the reverse. Exact two-sided test.

| comparison | task | b (bf16 right) | c (4-bit right) | p | verdict |
|---|---|---|---|---|---|
| Cosmos3-Edge bf16 -> simulated INT4 AWQ | distance | 68 | 30 | 0.00016 | **real** |
| Cosmos3-Edge bf16 -> simulated INT4 AWQ | left_right | 27 | 39 | 0.17529 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> simulated INT4 AWQ | mcq | 25 | 20 | 0.55148 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> REAL INT4 AWQ (TensorRT) | distance | 73 | 19 | 0.00000 | **real** |
| Cosmos3-Edge bf16 -> REAL INT4 AWQ (TensorRT) | left_right | 38 | 34 | 0.72395 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> REAL INT4 AWQ (TensorRT) | mcq | 25 | 19 | 0.45138 | not distinguishable from churn |
| AWQ simulated -> AWQ real (TensorRT), b = simulated right | distance | 37 | 21 | 0.04794 | **real** |
| AWQ simulated -> AWQ real (TensorRT), b = simulated right | left_right | 29 | 13 | 0.01952 | **real** |
| AWQ simulated -> AWQ real (TensorRT), b = simulated right | mcq | 13 | 12 | 1.00000 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> real NF4 | distance | 35 | 38 | 0.81512 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> real NF4 | left_right | 22 | 21 | 1.00000 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> real NF4 | mcq | 14 | 13 | 1.00000 | not distinguishable from churn |
| head-to-head: AWQ (sim) vs NF4 (real), b = AWQ right | distance | 26 | 67 | 0.00003 | **real** |
| head-to-head: AWQ (sim) vs NF4 (real), b = AWQ right | left_right | 44 | 31 | 0.16543 | not distinguishable from churn |
| head-to-head: AWQ (sim) vs NF4 (real), b = AWQ right | mcq | 22 | 26 | 0.66547 | not distinguishable from churn |
| head-to-head: AWQ (real TRT) vs NF4 (real), b = AWQ right | distance | 21 | 78 | 0.00000 | **real** |
| head-to-head: AWQ (real TRT) vs NF4 (real), b = AWQ right | left_right | 37 | 40 | 0.81989 | not distinguishable from churn |
| head-to-head: AWQ (real TRT) vs NF4 (real), b = AWQ right | mcq | 22 | 27 | 0.56817 | not distinguishable from churn |
| Gemma-4-E4B bf16 -> QAT Q4_0 | distance | 45 | 35 | 0.31431 | not distinguishable from churn |
| Gemma-4-E4B bf16 -> QAT Q4_0 | left_right | 28 | 22 | 0.47989 | not distinguishable from churn |
| Gemma-4-E4B bf16 -> QAT Q4_0 | mcq | 28 | 25 | 0.78385 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> simulated INT4 AWQ | distance, each arm with its own re-fitted k | 66 | 50 | 0.16342 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> REAL INT4 AWQ (TensorRT) | distance, each arm with its own re-fitted k | 68 | 66 | 0.93120 | not distinguishable from churn |
| AWQ simulated -> AWQ real (TensorRT), b = simulated right | distance, each arm with its own re-fitted k | 28 | 42 | 0.11961 | not distinguishable from churn |
| Cosmos3-Edge bf16 -> real NF4 | distance, each arm with its own re-fitted k | 47 | 41 | 0.59429 | not distinguishable from churn |
| head-to-head: AWQ (sim) vs NF4 (real), b = AWQ right | distance, each arm with its own re-fitted k | 53 | 63 | 0.40347 | not distinguishable from churn |
| head-to-head: AWQ (real TRT) vs NF4 (real), b = AWQ right | distance, each arm with its own re-fitted k | 75 | 71 | 0.80402 | not distinguishable from churn |
| Gemma-4-E4B bf16 -> QAT Q4_0 | distance, each arm with its own re-fitted k | 42 | 58 | 0.13321 | not distinguishable from churn |

## How to read these deltas

Chance levels, computed from the item set: **left_right 50.0%** (two-way choice) and
**mcq 14.22%** — mcq is not a four-option question, it offers between 3 and 13 numbered
regions per item, so chance is the mean of 1/n_regions, not 1/4.

**Gemma's bf16 left_right (51.00%) is at chance, and its mcq (12.50%)
is *below* the 14.22% chance level.** Signal that was never there cannot be lost, so its
quantisation delta on those two tasks is uninformative — a flat Gemma left_right delta is **not** evidence
that quantisation is free. The only meaningful Gemma delta is **distance** (16.26% bf16),
which is the one task where it demonstrably has something to lose.

Cosmos3-Edge clears chance on both: left_right 64.80% vs 50.0%, mcq
17.76% vs 14.22% — though the mcq margin is only
+3.54 points, so read that column with its N (456) in mind. All three
tasks are informative for Cosmos3-Edge.

## Distance calibration constant k, re-fitted per arm

Estimator, stated so it is reproducible: **k = median(gt / pred)** on the fitting half; halves are even/odd
position in the id-sorted distance items, so the split is deterministic and identical across arms. Each half
is scored with the *other* half's k, so **the gains in this table are held out.** The re-fitted McNemar
rows and the own-constant figures quoted above are **in-sample** (each arm's full-set k applied to its own
full set): 178 / 162 / 176 / 172 for bf16 / simulated / TensorRT / NF4 against the held-out 174 / 158 /
167 / 174 below, and 54 / 70 for the two Gemma rows. Every re-fitted McNemar row — all seven —
reconciles against the in-sample set: b - c equals the difference of the two counts in each case.
In-sample is the right basis for *comparing arms* — every arm
is favoured equally — and the wrong basis for claiming an absolute gain, which is why both are reported.

| row | k (half A) | k (half B) | k mean | uncalibrated | held-out calibrated | gain (pts) | N usable |
|---|---|---|---|---|---|---|---|
| Cosmos3-Edge bf16 (round 1) | 1.1492 | 1.1591 | 1.1541 | 140 | 172 | +7.13 | 449/486 |
| Cosmos3-Edge bf16 (L0 control) | 1.1484 | 1.1609 | 1.1546 | 143 | 174 | +6.90 | 449/486 |
| Cosmos3-Edge INT4 AWQ (simulated) | 1.2225 | 1.1984 | 1.2105 | 104 | 158 | +12.05 | 448/486 |
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | 1.198 | 1.2315 | 1.2148 | 89 | 167 | +17.37 | 449/486 |
| Cosmos3-Edge NF4 (real kernels) | 1.1561 | 1.1696 | 1.1628 | 146 | 174 | +6.24 | 449/486 |
| Gemma-4-E4B-it bf16 (round 1) | 0.9456 | 0.8981 | 0.9218 | 78 | 60 | -4.07 | 442/486 |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | 1.1522 | 1.124 | 1.1381 | 68 | 68 | +0.00 | 449/486 |
| **Cosmos3-Edge INT4 AWQ — Orin engine (SM87)** | — | — | — | — | — | — | — |

The Orin row is deliberately blank. It is filled by the engine-equivalence gate run, and **its constant is
the one that gets baked** — the SM103 figure above is provisional until then.

**This table counts differently from the headline table, by 1-2 items per arm** (bf16 143 here against
144 there; simulated 104/106; NF4 146/147; TensorRT 89/90). The calibration pipeline can only use items
where both ground truth and prediction parse as positive numbers — it needs a `gt/pred` ratio — so it
drops 37-38 per arm. The dataset's own scorer still scores those items, and one or two of them happen to
land inside the +/-10% band anyway. Quote 486-based figures or 449-based ones, never both in a sentence.

### The left_right prior is not a re-thresholding target

left_right is balanced 250/250 and the model answers `left` 72.4% of the time: **87.2% correct when the
answer is left, 42.4% when it is right** — below chance on half the set, and nearly unchanged by
quantisation. The obvious fix is order-debiasing: the question asks whether A is right of B, so asking the
mirror about the same image must invert the answer. Run over all 500 items on the TensorRT engine:

| | n = 500 |
|---|---|
| same answer to the mirrored question (**the prior**) | 328/500 = **65.6%** |
| answer flipped, as geometry requires | 172/500 = 34.4% |
| accuracy, as run | 320/500 = 64.00% |
| accuracy, order-debiased | 254/500 = 50.80% (**-13.20**) |
| accuracy on the order-consistent subset | 121/172 = **70.35%** |

**Tie rule, without which the debiased figure is not derivable:** where the pair flips, keep the original
answer; where it does not, the prior is showing, so take the answer the model is biased *against* (`right`).
Any tie rule loses on those 328 items — keeping the biased answer merely reproduces the 64% baseline — and
that is the point: the pair carries no information there.

**So order-debiasing cannot help.** Its ceiling is the 121 it gets right when consistent plus a guess on
the rest, near 57%, below the 64% already scored. The 72/28 split is not a mis-set threshold on a working
comparison; it is the absence of the comparison on two thirds of items, partly masked by a bias that pays.
What survives is a **confidence signal** — 70.35% when order-consistent against 60.7% when not — at 2x
inference cost, which is worth having on a rover that can decline. A logit-bias variant cannot be judged
from these outputs at all: the harness saves decoded text, not scores. Reproduce with
`scripts/l0_swap_probe.py`.

### Is one constant really one constant?

Not quite, and it matters most where the rover cares most. Median `gt/pred` by ground-truth range:

| arm | [0,4) m | [4,8) | [8,12) | [12,+) | full-set median | spread |
|---|---|---|---|---|---|---|
| bf16 | 1.1028 | 1.2203 | 1.1228 | 1.1465 | 1.1492 | 0.118 |
| **INT4 AWQ (TensorRT)** | 1.1277 | 1.2486 | 1.1516 | **1.3350** | 1.2083 | **0.207** |
| NF4 | 1.0837 | 1.2317 | 1.1513 | 1.1441 | 1.1574 | 0.148 |

The last column is the **full-set median** of `gt/pred`, not the split-half mean `k` in the table below —
1.1492 here against 1.1546 there for bf16. Same estimator, different sample.

bf16's far bucket (1.1465) sits on its global constant (1.1492); the AWQ engine's far bucket needs
**1.3350 against a global 1.2083**, so a single constant systematically under-corrects exactly the long
readings. Held out, split-half, a four-bucket constant is worth **+3.56 points** on the AWQ arm
(167 -> 183 of the 449 usable, the same basis as the `gain` column below, where the global constant is
worth +17.37), **+0.00** on bf16 and **-1.44** on NF4. So AWQ's under-read is range-dependent and
the other two arms' are not — which is a second way of saying the constant belongs to the artefact.

This qualifies the headline rather than overturning it: one global constant still recovers almost all of
the loss (p = 0.931 against bf16). But whoever bakes a constant for a rover that reads at 12 m and beyond
should fit it per range bucket, and the gate below should report k per bucket, not one number.

**Gemma's constant flips sign under quantisation** — 0.9218 at bf16 to 1.1381 at Q4_0. It changes from a
long-reader to a short-reader, and calibration *hurts* it at bf16 (-4.07 pts) while doing nothing at Q4_0
(+0.00). That is the signature of a model whose distance outputs carry no consistent scale to correct;
contrast Cosmos, whose two half-fits agree to three decimals in every arm. Do not bake a constant for Gemma.

**k was re-fitted on the 4-bit outputs, not carried over.** The round-1 constant on record was ~1.15; this
procedure reproduces ~1.154 on the same round-1 bf16 outputs (halves 1.149 / 1.159, +7.13 pts held out),
close to but not identical with the 1.158 / 1.141 and +6.6 pts recorded earlier — the earlier fit used a
procedure that was not written down, so it cannot be reproduced exactly. Every arm here uses the estimator
above, so the arms are comparable with each other.

### Diagnosis from the histograms

The ground truth is balanced — left_right is **250 left / 250 right**, and the
mcq answer key is near-uniform across region indices. So a bias in the answer column is a bias in the model,
not an artefact of the item set, and any arm answering one label most of the time is paying for it.

| row | task | modal answer | modal share | accuracy | reading |
|---|---|---|---|---|---|
| Cosmos3-Edge bf16 (round 1) | left_right | `left` | 72.4% | 65.20% | varied and above chance |
| Cosmos3-Edge bf16 (round 1) | mcq | `2` | 32.0% | 17.54% | varied and above chance |
| Cosmos3-Edge bf16 (L0 control) | left_right | `left` | 72.4% | 64.80% | varied and above chance |
| Cosmos3-Edge bf16 (L0 control) | mcq | `2` | 31.6% | 17.76% | varied and above chance |
| Cosmos3-Edge INT4 AWQ (simulated) | left_right | `left` | 73.6% | 67.20% | varied and above chance |
| Cosmos3-Edge INT4 AWQ (simulated) | mcq | `2` | 40.1% | 16.67% | varied and above chance |
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | left_right | `left` | 70.0% | 64.00% | varied and above chance |
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | mcq | `2` | 40.6% | 16.45% | varied and above chance |
| Cosmos3-Edge NF4 (real kernels) | left_right | `left` | 75.0% | 64.60% | varied and above chance |
| Cosmos3-Edge NF4 (real kernels) | mcq | `2` | 37.1% | 17.54% | varied and above chance |
| Gemma-4-E4B-it bf16 (round 1) | left_right | `left` | 74.6% | 51.00% | answers vary but carry no signal — spread out, still at chance |
| Gemma-4-E4B-it bf16 (round 1) | mcq | `2` | 31.4% | 12.50% | answers vary but carry no signal — spread out, still at chance |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | left_right | `left` | 81.4% | 49.80% | **collapses to a constant** — answers `left` on 81% of items |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | mcq | `1` | 45.8% | 11.84% | answers vary but carry no signal — spread out, still at chance |

**Both models answer `left` about three times in four on a 50/50 set.** For Gemma that bias is the whole
story — it scores at chance. For Cosmos3-Edge it is not: it clears chance by 15 points *despite* the bias,
which means the bias is costing it accuracy it already has. It *looks* like a prior to correct, the way
`k` corrects the distance under-read — and the order-swap probe above shows it is not. `k` fixes a scale
error on a working measurement; this is the **absence** of the comparison on two thirds of items, and
re-thresholding cannot recover a judgement that was never made.

The histogram shape is still a useful diagnostic, just not a training signal: a model that **collapses to
a constant** is failing to engage the task, while one whose answers are **spread out and still at chance**
has no signal to sharpen. Cosmos is the third case — spread, above chance, and order-invariant — which is
why it reads as fixable and is not.

## Answer histograms

Distribution of *parsed* answers per task per arm. This is what separates "no spatial signal" from
"always answers the same thing".

**Cosmos3-Edge bf16 (round 1)**

- distance (N=486): `[8,12)`: 122, `[3,5)`: 114, `[5,8)`: 82, `[1,2)`: 57, `[12,20)`: 49, `[2,3)`: 35, `[0,1)`: 27
- left_right (N=500): `left`: 362, `right`: 138
- mcq (N=456): `2`: 146, `4`: 61, `3`: 58, `1`: 49, `5`: 35, `6`: 29, `9`: 24, `0`: 23, `8`: 16, `7`: 15

**Cosmos3-Edge bf16 (L0 control)**

- distance (N=486): `[8,12)`: 123, `[3,5)`: 112, `[5,8)`: 81, `[1,2)`: 58, `[12,20)`: 50, `[2,3)`: 35, `[0,1)`: 27
- left_right (N=500): `left`: 362, `right`: 138
- mcq (N=456): `2`: 144, `3`: 60, `4`: 60, `1`: 49, `5`: 37, `6`: 27, `9`: 24, `0`: 23, `8`: 17, `7`: 15

**Cosmos3-Edge INT4 AWQ (simulated)**

- distance (N=486): `[8,12)`: 131, `[3,5)`: 114, `[5,8)`: 92, `[1,2)`: 50, `[2,3)`: 46, `[0,1)`: 30, `[12,20)`: 23
- left_right (N=500): `left`: 368, `right`: 132
- mcq (N=456): `2`: 183, `3`: 70, `1`: 59, `4`: 48, `5`: 46, `0`: 23, `9`: 11, `7`: 8, `6`: 7, `8`: 1

**Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)**

- distance (N=486): `[8,12)`: 140, `[3,5)`: 111, `[5,8)`: 93, `[1,2)`: 54, `[2,3)`: 41, `[0,1)`: 29, `[12,20)`: 18
- left_right (N=500): `left`: 350, `right`: 150
- mcq (N=456): `2`: 185, `3`: 67, `1`: 51, `5`: 50, `4`: 50, `0`: 29, `6`: 10, `7`: 9, `9`: 4, `8`: 1

**Cosmos3-Edge NF4 (real kernels)**

- distance (N=486): `[3,5)`: 119, `[8,12)`: 106, `[5,8)`: 84, `[2,3)`: 50, `[12,20)`: 48, `[1,2)`: 44, `[0,1)`: 35
- left_right (N=500): `left`: 375, `right`: 125
- mcq (N=456): `2`: 169, `1`: 60, `4`: 57, `3`: 45, `9`: 41, `5`: 35, `7`: 18, `0`: 14, `6`: 10, `8`: 7

**Gemma-4-E4B-it bf16 (round 1)**

- distance (N=486): `[8,12)`: 291, `[3,5)`: 83, `[5,8)`: 44, `[12,20)`: 40, `[1,2)`: 18, `[0,1)`: 7, `[2,3)`: 2
- left_right (N=500): `left`: 373, `right`: 127
- mcq (N=456): `2`: 143, `1`: 110, `3`: 72, `4`: 60, `6`: 54, `5`: 10, `7`: 5, `0`: 2

**Gemma-4-E4B-it QAT Q4_0 (Unsloth)**

- distance (N=486): `[3,5)`: 167, `[8,12)`: 157, `[2,3)`: 51, `[1,2)`: 45, `[5,8)`: 39, `[12,20)`: 24, `>=20`: 2, `[0,1)`: 1
- left_right (N=500): `left`: 407, `right`: 93
- mcq (N=456): `1`: 209, `2`: 82, `3`: 79, `10`: 46, `7`: 15, `4`: 15, `6`: 7, `5`: 2, `9`: 1

## Thinking-enabled subsample — the winner

Run on the bake-off winner, **Cosmos3-Edge at bf16**, over the first 200 items of the distance + left_right set (paired by id: the thinking-off column is recomputed on exactly these items, not carried from the full-set headline).

| task | thinking off | thinking on | delta |
|---|---|---|---|
| distance | 25/95 = 26.32% | 23/95 = 24.21% | -2.11 |
| left_right | 66/105 = 62.86% | 64/105 = 60.95% | -1.90 |
| **both, pooled** | 91/200 = 45.50% | 87/200 = 43.50% | -2.00 |

**Cost.** Over the same 200 items: 1150 generated tokens in 19.8s at 57.97 tok/s with thinking off, versus 31889 tokens in 337.3s at 94.54 tok/s with it on — **27.7x the tokens and 17.0x the wall time** on a B300.

**Token budget.** The thinking arm was given `max_new_tokens=512` against 192 for the rest of the bake-off. Raising it is not a thumb on the scale: at 192 a reasoner spends the budget on reasoning and never emits its `ANSWER:` line, and the parser then falls back to the last number or direction word in the reasoning text — which scores the model's thinking aloud, not its answer. Generations that still used the whole budget: 5/200 (2.5%) with thinking on, 0/200 (0.0%) with it off.

B300 figures. An Orin would pay the same token multiple against a much smaller budget.

**Reconciling this with L1, which reached the opposite conclusion.** L1 keeps thinking *on* because it
took detections from 8 to 10 of 12; L0 finds it costs ~2 points and 27.7x the tokens. Both are right,
and they are not in tension: L0 asks a closed question with a one-token answer already in the model,
where extra reasoning only adds places to drift. L1 asks for open-vocabulary grounding plus coordinates,
where the reasoning is doing perceptual work — writing the entity out before committing a box. The rule
that satisfies both: **thinking on for grounding and operator commands, off for closed spatial
queries**, and it is a
per-request flag, so nothing has to be chosen globally.

## Per-item agreement — and why the INT4 disagreement needs a floor under it

The bake-off hand-off asks for a per-item agreement check between precisions. That number is uninterpretable on its own,
because two runs of the *same model at the same precision* do not agree either. Both are measured here.

| comparison | identical predictions | per category |
|---|---|---|
| environment floor — round-1 bf16 vs L0 bf16 control (same model, same precision, different container) | **1374/1442 = 95.28%** | distance 87.9% · left_right 99.2% · mcq 98.9% |
| quantisation — L0 bf16 control vs L0 simulated INT4 AWQ (same container, same seed) | **783/1442 = 54.30%** | distance 13.4% · left_right 86.8% · mcq 62.3% |
| quantisation — L0 bf16 control vs L0 real NF4 | **889/1442 = 61.65%** | distance 21.0% · left_right 91.4% · mcq 72.4% |
| the two quantisers against each other — AWQ (sim) vs NF4 (real) | **742/1442 = 51.46%** | distance 9.9% · left_right 85.0% · mcq 59.0% |

Look at the per-category split rather than the totals. Quantisation leaves **86.8%** of left_right and
**62.3%** of mcq answers untouched but changes **all but 13.4%** of the distance answers — while the
k-corrected distance accuracy is statistically unchanged. That is the signature of a **scale shift, not
lost information**: 4-bit weights move nearly every numeric estimate, and moving them all back by one
constant restores the score. It is also why an agreement threshold applied uniformly across tasks would
fail this model on distance for the wrong reason.

Read the second row against the first, never against 100%. Greedy decode is deterministic within a
container but not across two of them: a different transformers build reorders enough floating-point
work to flip a fraction of answers at identical weights. Any INT4 disagreement below the environment
floor is not evidence of quantisation damage.

## What these numbers are not

- **There are two INT4 rows and they answer different questions.** The *simulated* one is
  `mtq.quantize()` fake-quant inside PyTorch — weights on the INT4 grid, GEMM still in bf16 — so it is the
  weight-precision delta and says nothing about runtime. The *TensorRT* one is real INT4 kernels, but on
  **SM103**, which is not the Orin engine: same ONNX export, different plan. Neither row licenses an Orin
  accuracy claim. The **B300-vs-Orin engine-equivalence gate** is what closes that gap, and it has not run.
- The two 4-bit rows are **not the same kind of object**: Gemma's is QAT (quantisation-aware trained
  upstream by Google, then packed to Q4_0); Cosmos3-Edge's is PTQ (post-training AWQ). A QAT model has
  been trained to survive its own quantisation; a PTQ model has not. Do not read the two deltas as a like-
  for-like comparison of "how well each model quantises".
- **No per-item L0 output is reproducible elsewhere to better than ~5%.** Two runs of the *same* model at
  the same precision in different containers agree on 95.3% of items overall and only 87.9% on distance.
  That is the environment floor, and it bounds every per-item comparison in this document: a disagreement
  smaller than the floor is not evidence. It is also why the paired tests here compare arms run in one
  container, and why an Orin-vs-B300 engine gate has to be read against the same floor.
- **The Gemma pair is the exception to that.** Its bf16 row is the round-1 container and its Q4_0 row is
  llama.cpp — a different runtime, not just a different container — so that comparison carries more than
  the floor's worth of nuisance variation. It changes no conclusion, because every Gemma delta is churn
  already (all three p > 0.3), but the pair should not be quoted as a clean precision contrast.
- The Gemma artefact is **Q4_0, not Q4_K_XL**, despite the `UD-Q4_K_XL` filename: 100% of parameters are
  stored Q4_0, no K-quants. This is the artefact deployed on jetson0, so any page or hand-off calling the
  incumbent "Q4_K_XL" is repeating a filename, not a format.
- Parse-failure rates are in the k table's `N usable` column. Gemma bf16 yields a usable distance number
  on 442/486 against 449/486 for the Cosmos arms (448 for the simulated one) — 44 failures against 37.
  Reported for completeness; at p ~ 0.4 it is not a difference this item set establishes.
- Nothing here has been executed on a Jetson.

## Reproducibility

**Dataset.** NVIDIA *PhysicalAI-Spatial-Intelligence-Warehouse* (`nvidia/PhysicalAI-Spatial-Intelligence-
Warehouse` on the Hugging Face Hub), CC-BY-4.0, synthetic warehouse scenes rendered in Omniverse with
rule-generated Q&A refined by Llama-3.1-70B-Instruct (so the annotations also carry Llama 3.1 Community
License terms). **Gated — request access; cited, never redistributed.** Split: `val.json`, 1,942 items, of
which this report scores 1,442 — distance 486, left_right 500, mcq 456. The `count` category (500) is not
used. The model is shown a rendered frame with the referenced regions drawn as numbered outlines, not the
source image.

**Scoring rules.** distance: correct within **+/-10%** of ground truth after applying the arm's constant
`k`. left_right and mcq: **exact match** (mcq answers are region indices, not option letters). Scored by
the dataset's own `utils/compute_scores.py`; `k` is a split-half median of `gt/pred`, fitted on one half
and applied to the other.

**Code.** Prompt, rendering and parser `scripts/spatial_qa_eval.py` (`build_prompt`, `render_item`,
`parse_answer`); PyTorch arms `scripts/l0_cosmos_int4_sim.py`; TensorRT arm `scripts/l0_trt_eval.py`;
scorer, `k` estimator and this document `scripts/l0_report.py` (`fit_k`, `mcnemar`, `wrong_k_cost`);
order-swap probe `scripts/l0_swap_probe.py`.

**Versions.**

| component | version |
|---|---|
| transformers (bf16, simulated-INT4, NF4 arms) | 5.16.1 |
| transformers (round-1 container, environment-floor pair) | round-1 image, **not identical** to the above |
| torch | 2.13.0+cu130 |
| nvidia-modelopt (AWQ quantise) | 0.33.0 |
| bitsandbytes (NF4) | as packaged in the run container |
| llama.cpp (Gemma Q4_0) | b9602 |
| TensorRT | 10.13.2.6 |
| TensorRT Edge-LLM | `e8b2952` (v0.10.1) **+ 7 local patches** = NVIDIA/TensorRT-Edge-LLM #205 and #207 |
| CUDA | 13.0 |
| hardware | DGX B300, SM103 |

The two transformers builds behind the environment floor are **not the same image** — that is what the
floor measures. The round-1 container was not version-pinned at the time, which is itself a finding: it is
why the floor had to be measured rather than assumed.

**Artefacts.** AWQ checkpoint `quantized-int4-awq-v2/model.safetensors`, 2,408,300,272 bytes, built
2026-09-10. ONNX export `onnx-v3/llm/model.onnx.data`, 866,451,480 bytes, same date — the export the
Jetson also consumes. Engines are per-SM and not shared: the SM103 plan measured here is 843.7 MiB against
the Orin SM87 plan's 838.9 MiB from the same ONNX — **the same size, a different plan.** The point is not
that one is larger; it is that they are not the same artefact.

**Run dates.** Round-1 bf16 rows and the L0 control, simulated-INT4 and NF4 arms: 2026-09-09/10. Gemma
Q4_0: 2026-09-10. TensorRT INT4-AWQ rows and the order-swap probe: **2026-09-11**.
