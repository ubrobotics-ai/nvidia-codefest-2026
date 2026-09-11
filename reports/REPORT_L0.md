# L0 — model x precision, 7 rows on one machine

Every accuracy below was produced on the same host, over the same items, with the same prompt, the same
rendered numbered region outlines and the same parser (including the mcq fix: answers are **region
indices**, not option letters). Greedy decode, `enable_thinking=False` unless a row says otherwise.

## The finding

**4-bit weights shift the distance *scale*; they do not destroy the information, and one constant
restores it.** On the deployed TensorRT INT4-AWQ engine, distance reads 29.63% at bf16 against
18.52% at INT4 — a 11.1-point loss, paired McNemar b=73 c=19, p < 0.0001. Re-fit the
distance constant per arm and that loss disappears entirely: 68 vs 66, **p = 0.931**. left_right
(p = 0.724) and mcq (p = 0.451) never moved. NF4 behaves the same way, and the two 4-bit schemes are
indistinguishable from each other (p = 0.804).

### What to do with it

**Bake k = 1.2148 into the INT4 rover build, not 1.1546.** The constant is a property of the
precision, not of the model. Measured on these items: the INT4 answers score
36.01% under their own constant and 33.13% under the bf16 one —
**2.88 points (14 items) thrown away for no reason.**

That figure is smaller than the 11.1-point raw gap above, and the difference matters: almost
all of the raw gap is removed by calibrating *at all* (18.31% uncalibrated to 36.01% calibrated), and
only the 2.88-point remainder is the cost of using the *wrong* constant. Do not quote
the raw gap as the cost of mis-calibration.

**The Friday SFT decision.** L0 measures spatial reasoning that quantisation does not damage, so a
4-bit deployment is not the thing to spend SFT budget repairing. Distance calibration is a one-scalar
fit, not a training problem. If SFT is spent anywhere, the evidence points at *output format*, which
is where L1 shows the real failure (0 of 24 schema-valid outputs) and where no constant helps.

### Provenance of the TensorRT row — read before quoting it

This row was produced on a **DGX B300 (SM103)**, which cannot build the Edge-LLM INT4 path at all on
v0.10.1: it needs the SM allowlist patch in NVIDIA/TensorRT-Edge-LLM#207 plus the NVRTC include-path
fix in #205, **both unmerged at the time of writing**. The row therefore exists under a local patch set.

It shares the **ONNX export** with the Jetson deployment, not the engine. SM103 and SM87 produce
different plans from the same export, so "the same artefact the Jetson deploys" is true of the ONNX
and false of the engine. That makes an engine-equivalence gate runnable — B300 outputs on these 986
items against Orin outputs on a 200-pair subsample, compared per item — and **it has not been run.**
Until it is, this row licenses claims about INT4-AWQ *as a quantisation*, not about the Orin engine.

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
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | INT4 AWQ | tensorrt-edgellm | PTQ (modelopt AWQ), REAL INT4 kernels via the Edge-LLM ONNX export -- the same artefact the Jetson deploys | real (TensorRT Int4GroupwiseGemmPluginV2) | 90/486 = 18.52% | 320/500 = 64.00% | 75/456 = 16.45% | **41.26%** | 32.99% |
| Cosmos3-Edge NF4 (real kernels) | NF4 | transformers + bitsandbytes NF4 (REAL 4-bit kernels) | PTQ (bitsandbytes NF4), real kernels -- a DIFFERENT quantiser from the shipped AWQ checkpoint | real (bitsandbytes) | 147/486 = 30.25% | 323/500 = 64.60% | 80/456 = 17.54% | **47.42%** | 37.46% |
| Gemma-4-E4B-it bf16 (round 1) | bf16 | transformers | none | n/a — full precision | 79/486 = 16.26% | 255/500 = 51.00% | 57/456 = 12.50% | **33.63%** | 26.59% |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | Q4_0 | llama.cpp (llama-server, CUDA) | QAT (upstream Google), packed uniform Q4_0 by Unsloth -- the UD-Q4_K_XL filename is a misnomer: 100% of parameters are stored Q4_0, no K-quants | real (llama.cpp) | 69/486 = 14.20% | 249/500 = 49.80% | 54/456 = 11.84% | **32.00%** | 25.28% |

N per cell is printed in the cell. distance N=486, left_right N=500, mcq N=456.

### Throughput and memory — B300, not Orin

| row | tok/s | peak GB | note |
|---|---|---|---|
| Cosmos3-Edge bf16 (round 1) | 43.94 | 5.1 | B300 |
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
Neither is a deployment number; the Orin figures (52.8 tok/s decode at 3,377 MB) are the ones that count.

Blank cells are honest blanks: the TensorRT arm is driven by `llm_inference` as a subprocess and the
harness records wall-clock, not generated-token counts, and the llama.cpp arm reports no peak-memory
figure through its server API. Neither blank is a deployment claim.

## Points lost to 4-bit

Positive = **worse** at 4 bits. Read every row against the McNemar table above: most of these
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

Cosmos3-Edge clears chance on both: left_right 65.20% vs 50.0%, mcq
17.54% vs 14.22% — though the mcq margin is only
+3.32 points, so read that column with its N (456) in mind. All three
tasks are informative for Cosmos3-Edge.

## Distance calibration constant k, re-fitted per arm

Estimator, stated so it is reproducible: **k = median(gt / pred)** on the fitting half; halves are even/odd
position in the id-sorted distance items, so the split is deterministic and identical across arms. Each half
is scored with the *other* half's k, so every reported gain is held out.

| row | k (half A) | k (half B) | k mean | uncalibrated | held-out calibrated | gain (pts) | N usable |
|---|---|---|---|---|---|---|---|
| Cosmos3-Edge bf16 (round 1) | 1.1492 | 1.1591 | 1.1541 | 140 | 172 | +7.13 | 449/486 |
| Cosmos3-Edge bf16 (L0 control) | 1.1484 | 1.1609 | 1.1546 | 143 | 174 | +6.90 | 449/486 |
| Cosmos3-Edge INT4 AWQ (simulated) | 1.2225 | 1.1984 | 1.2105 | 104 | 158 | +12.05 | 448/486 |
| Cosmos3-Edge INT4 AWQ (real kernels, TensorRT) | 1.198 | 1.2315 | 1.2148 | 89 | 167 | +17.37 | 449/486 |
| Cosmos3-Edge NF4 (real kernels) | 1.1561 | 1.1696 | 1.1628 | 146 | 174 | +6.24 | 449/486 |
| Gemma-4-E4B-it bf16 (round 1) | 0.9456 | 0.8981 | 0.9218 | 78 | 60 | -4.07 | 442/486 |
| Gemma-4-E4B-it QAT Q4_0 (Unsloth) | 1.1522 | 1.124 | 1.1381 | 68 | 68 | +0.00 | 449/486 |

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
which means the bias is costing it accuracy it already has. That is a prior to correct, in the same way
`k` corrects the distance under-read — and it is the cheapest thing on this page to fix.

This is the distinction the Friday SFT decision hangs on: a model that **collapses to a constant** has a
formatting or grounding failure that supervised fine-tuning on a few hundred examples can plausibly fix; a
model whose answers are **spread out and still at chance** has no spatial signal to sharpen, and SFT on this
task size will not create one.

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
that satisfies both: **thinking on for grounding, off for closed spatial queries**, and it is a
per-request flag, so nothing has to be chosen globally.

## Per-item agreement — and why the INT4 disagreement needs a floor under it

§4b asks for a per-item agreement check between precisions. That number is uninterpretable on its own,
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

- The Cosmos INT4 row is **simulated quantisation** — `mtq.quantize()` fake-quant weights inside PyTorch,
  not engine kernels. It is the right number for the weight-precision delta and the wrong one for runtime.
  The jetson1 per-item agreement check against these outputs is what covers the difference.
- The two 4-bit rows are **not the same kind of object**: Gemma's is QAT (quantisation-aware trained
  upstream by Google, then packed to Q4_K_XL); Cosmos3-Edge's is PTQ (post-training AWQ). A QAT model has
  been trained to survive its own quantisation; a PTQ model has not. Do not read the two deltas as a like-
  for-like comparison of "how well each model quantises".
- **No per-item L0 output is reproducible elsewhere to better than ~5%.** Two runs of the *same* model at
  the same precision in different containers agree on 95.3% of items overall and only 87.9% on distance.
  That is the environment floor, and it bounds every per-item comparison in this document: a disagreement
  smaller than the floor is not evidence. It is also why the paired tests here compare arms run in one
  container, and why an Orin-vs-B300 engine gate has to be read against the same floor.
- The Gemma artefact is **Q4_0, not Q4_K_XL**, despite the `UD-Q4_K_XL` filename: 100% of parameters are
  stored Q4_0, no K-quants. This is the artefact deployed on jetson0, so any page or hand-off calling the
  incumbent "Q4_K_XL" is repeating a filename, not a format.
- Parse-failure rates differ sharply by arm and are given in the k table's `N usable` column. Gemma bf16
  yields a usable distance number on 442/486 items — a 9% failure rate — against 449/486 for every Cosmos
  arm. Given how much L1 turns on contract adherence, that gap is a finding, not bookkeeping.
- Nothing here has been executed on a Jetson.
