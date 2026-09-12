# SFT working directory

Scoped by [`../docs/SFT_PIPELINE.md`](../docs/SFT_PIPELINE.md). Phase 0 and the unblocked
half of Phase 1 are done; the rest needs files that live on the robot.

| file | what |
|---|---|
| `BASELINE.json` | Phase 0. Artefact checksums, repo SHAs, and every measurement this SFT must not regress. Frozen before any weight changed. |
| `gen_command_data.py` | Phase 1. Generates the command-vocabulary and UNKNOWN buckets, refusing to write if the output would contaminate the eval. |
| `data/commands.jsonl` | 890 examples. |
| `train_lora.py` | Phase 2. LoRA r=16 on the **bf16** checkpoint, split seed-disjoint. |
| `eval_lora.py` | Phase 3. Scores base vs adapter on the held-out dev slice. |
| `merge_lora.py` | Phase 4. Folds the adapter into the checkpoint's real shards, and verifies it on disk. |

## What is generated, and why in these proportions

| class | n | why |
|---|---|---|
| FORWARD | 150 | weakest measured class (7/10), and `"Go forward"` → BACK is an inversion |
| BACK | 100 | |
| LEFT / RIGHT | 100 each | balanced deliberately; RIGHT must not become the minority |
| STOP | 80 | already 5/5 — this is the safety class and the gate is *no regression* |
| SEARCH / REPORT | 80 each | |
| **UNKNOWN** | **200** | largest bucket: the model has no declining behaviour at all to reinforce, and which wrong thing it picks is unstable |

## Two decisions worth knowing about

**Nautical vocabulary is deliberately untrained.** The eval contains `"Turn to starboard"`
→ RIGHT, which the model inverts. Any nautical seed is a near-duplicate of that item, so
training one and testing the other is contamination by paraphrase — the first generated
batch produced nine `"to starboard side"` rows and the audit caught them. Fixing that
inversion needs a **refreshed eval with fresh phrasings** first. Until then the class stays
untrained and honestly measured.

**The audit is blocking, not advisory.** `gen_command_data.py` exits non-zero rather than
write data containing an exact hold-out phrase, a near-duplicate of one, or stray
non-latin characters — the first batch also contained a CJK token left in a seed list.

## Blocked: the tool-call and schema buckets

These need `protocol.py` and `prompts.py` from the robot, which are not on this machine:

- **~250 tool-call examples.** Needs the ten tools' exact argument names and enums. The
  measured failure is *arguments* (1/10) rather than intent (3/10 correct command), and one
  named failure is `"dir":"right"` — out of enum. Generating against a guessed schema would
  teach a contract production does not use.
- **~120 schema objects.** Needs the nested entity schema. The highest-value field is
  `task_relevance`, omitted in 11 of 11 parseable outputs and the one the tolerant parser
  cannot infer — it has to guess, and that guess is part of why precision is 0.529.

**What to send:** `protocol.py`, `prompts.py`, and one *valid* example of each — a
well-formed tool call and a well-formed nested entity object. That is enough to generate
both buckets against the real contract.


## Phase 2-3 result — command classification

LoRA r=16 (`q,k,v,o,fc1,fc2`), 24.22 M trainable of 2.46 B (0.984%), 264 steps, batch 8,
lr 1e-4, on the **bf16** checkpoint.

| bucket | n | base | + LoRA |
|---|---:|---:|---:|
| FORWARD | 34 | 41.2% | 97.1% |
| BACK | 11 | 54.5% | 100.0% |
| LEFT | 18 | 83.3% | 100.0% |
| RIGHT | 22 | 27.3% | 95.5% |
| STOP | 19 | 73.7% | 100.0% |
| SEARCH | 23 | 91.3% | 100.0% |
| REPORT | 12 | 0.0% | 100.0% |
| UNKNOWN | 40 | 40.0% | 80.0% |
| **overall** | **179** | **51.4%** | **94.4%** |

Unparseable outputs go from 12 to 0. The two classes the L0 report called out both move:
RIGHT 27.3% → 95.5% (the inversion), and REPORT 0% → 100% (the model had no REPORT
behaviour at all). UNKNOWN is the weakest class after training and the one to watch — 80%
means the adapter still answers 8 of 40 unanswerable commands with a command.

### The split is seed-disjoint, and the first run was not

The first run of this scored **48.3% → 98.9%** and that number is void. Dev rows were
sampled at row level, and because each seed phrase expands into several paraphrases,
**every dev seed also appeared in training — 55/55 seeds, 89/89 rows.** The adapter was
being tested on phrasings of sentences it had memorised.

The split above is by **seed phrase**: 711 rows / 88 seeds train, 179 rows / 21 seeds dev,
no seed on both sides. 94.4% is the honest number and it is 4.5 points below the
contaminated one — the gap is the memorisation the first split was measuring.

### Why bf16 and not the quantised weights

The AWQ checkpoint is GPTQ-packed `uint8` — there are no differentiable weights to attach
adapters to. NF4 is the measurement control and training it would destroy that role. So
the adapter trains on bf16 and the **quantise-and-rebuild happens after**, which is also
what makes Phase 5 (re-export → re-quantise → rebuild the engine) a real gate rather than
a formality: none of the numbers above have survived INT4 yet.

### Not yet done

These are command-classification numbers on generated data, measured on one seed-disjoint
split with no confidence interval and no repeat at a second seed. They are **not** the
spatial-QA numbers in `../reports/REPORT_L0.md` and do not transfer to them. The
`BASELINE.json` regression gates have not been re-run against the adapter.


## Phases 4-6 — does it survive INT4?

Adapter merged into bf16, re-quantised with the **same recipe `BASELINE.json` froze** (only
`--model_dir` changed), re-exported, and rebuilt as an SM103 engine. Census on the new ONNX:
**169 `Int4GroupwiseGemmPluginV2` in the LLM graph, 0 in visual** — the Phase 4 gate.

The 179 seed-disjoint dev items, through the engine:

| bucket | n | INT4 engine | bf16 + LoRA |
|---|---:|---:|---:|
| FORWARD | 34 | 85.3% | 97.1% |
| BACK | 11 | 100.0% | 100.0% |
| LEFT | 18 | 100.0% | 100.0% |
| RIGHT | 22 | 100.0% | 95.5% |
| STOP | 19 | 100.0% | 100.0% |
| SEARCH | 23 | 100.0% | 100.0% |
| REPORT | 12 | 100.0% | 100.0% |
| UNKNOWN | 40 | 80.0% | 80.0% |
| **overall** | **179** | **92.7%** | **94.4%** |

Unparseable outputs: 0.

### The 1.7-point gap is serving format, not quantisation

`llm_inference` builds its own prompt — a **doubled system block** (ours, then an empty one)
plus a trailing `<think></think>` — which is not what the HF chat template produced during
training. Scoring bf16 through the template and INT4 through the engine would confound the
two. So the control replays the engine's exact formatted strings through the bf16 merged
checkpoint:

| arm | score |
|---|---:|
| bf16, HF chat template | 94.4% |
| bf16, **engine's** format | **92.7%** |
| INT4 engine | **92.7%** |

Paired on the same items: **3 discordant each way, exact McNemar p = 1.0000.** Quantisation
costs nothing measurable here, matching L0's finding on spatial QA (p = 0.931). The entire
1.7-point drop is the prompt mismatch — which is worth 3 items, and is a *fixable* loss:
train under the string the runtime actually sends.

### A failure worth recording

The first pass through Phase 4 reported success at every stage and was wrong. `merge_lora.py`
wrote the merged tensors to a side file and re-pointed `model.safetensors.index.json` at it;
`tensorrt-edgellm-quantize` resolves weights through the `transformer/` directory and ignored
the index, so it quantised the **un-finetuned** shards. Quantise exit 0, export exit 0, the
169/169 census passed, the engine built and answered coherently — and every one of the 224
language weight tensors was byte-identical to `quantized-int4-awq-v2`.

What caught it was the measurement, not the pipeline: the engine scored **48.6%**, against
51.4% for the base and 94.4% for the adapter, with REPORT at 0/12 — the base model's exact
signature. `merge_lora.py` now rewrites the real shards and re-opens them to assert every
merged tensor differs from the base on disk.

**None of the green lights in this pipeline distinguish "the adapter is in the engine" from
"the adapter is not in the engine."** Only a checkpoint diff and a held-out score do.

## Phase 6 — the L0 gates against this engine

All 1,442 spatial-QA items through the SFT engine, 0 empty generations, 136 s. `k` re-fitted
per arm rather than carried over, since AWQ scales moved on all 168 changed tensors. It
barely moved: **1.2148 → 1.2234, +0.71%**.

| category | n | baseline | SFT engine | exact McNemar |
|---|---:|---:|---:|---|
| distance (global k) | 486 | 36.21% | **30.45%** | b=52 c=24 **p=0.0018** |
| left_right | 500 | 64.00% | 64.40% | b=9 c=11 p=0.824 |
| mcq | 456 | 16.45% | 17.76% | b=12 c=18 p=0.362 |

**`left_right` and `mcq` did not move.** The interference worth worrying about — the LoRA
trained LEFT/RIGHT hard as command labels, and spatial left/right shares the attention path
— did not materialise.

### Distance moved, and the verdict depends on the calibration

Three defensible scorings disagree, so all three are here:

| scoring | baseline | SFT | verdict |
|---|---:|---:|---|
| **k = 1** (the basis `BASELINE.json`'s 18.52% uses) | 18.52% | **23.87%** | SFT **better**, gate passes |
| **global k**, re-fitted per arm | 36.21% | **30.45%** | SFT worse, **p = 0.0018** |
| **per-bucket k**, fitted split-half, scored held out | 37.24% | **33.95%** | SFT worse, p = 0.097 |

The regression is real but concentrated, and it is a *spread* problem rather than a *scale*
problem: median(SFT pred / baseline pred) = **1.0000**, and the two arms emit an identical
number on 172 of 486 items. Nothing shifted systematically; the far bucket got noisier.

| bucket | n | baseline | SFT |
|---|---:|---:|---:|
| [0,4) | 147 | 19.0% | 21.9% |
| [4,8) | 108 | 36.1% | 29.6% |
| [8,12) | 108 | 55.6% | 50.0% |
| **[12,+)** | 122 | **39.3%** | **23.0%** |

Near distances improved; **the far bucket lost 16.3 points**, and that single bucket is what
drags the global-k number down. That is why k=1 and calibrated k disagree: the SFT's raw
answers need slightly less correction, but they fit one global constant worse.

### What this means

A text-only command LoRA on `q,k,v,o,fc1,fc2` **did** reach numeric distance estimation,
which is the risk that justified running this gate at all. It did not touch left/right or
MCQ. Whether it fails depends on which calibration ships, and `BASELINE.json` does not say
— **the gate should have specified its calibration, and it did not.** That is a defect in
the baseline, not a result.

### The control: neither factor alone accounts for it

The bf16-merged arm has now been run, plus a fresh bf16 base arm — both re-run in this
session's environment, because a `peft` install put torch 2.13 in `~/.local` and shadowed the
container's 2.8, so the frozen bf16 arm was no longer a safe comparator.

| arm | k | distance | left_right | mcq |
|---|---:|---:|---:|---:|
| bf16 base | 1.1546 | 36.63% | 64.80% | 17.76% |
| bf16 merged | 1.1408 | 34.36% | 64.40% | 17.76% |
| INT4 base | 1.2148 | 36.21% | 64.00% | 16.45% |
| INT4 merged | 1.2234 | **30.45%** | 64.40% | 17.76% |

Paired on distance:

| contrast | b | c | p |
|---|---:|---:|---:|
| LoRA at bf16 (bf16 base → bf16 merged) | 31 | 20 | 0.161 |
| **LoRA at INT4** (INT4 base → INT4 merged) | 52 | 24 | **0.0018** |
| quantisation without LoRA (bf16 base → INT4 base) | 68 | 66 | 0.931 |
| quantisation with LoRA (bf16 merged → INT4 merged) | 78 | 59 | 0.124 |

**Neither factor alone is significant; only the combination is.** The far bucket makes it
plain — three cells sit together and the fourth falls off:

| arm | [12,+) accuracy |
|---|---:|
| bf16 base | 41.0% |
| bf16 merged | 37.7% |
| INT4 base | 39.3% |
| **INT4 merged** | **23.0%** |

Two cautions on reading that. The two LoRA contrasts point the same way and differ mainly in
magnitude, and **the interaction itself has not been tested** — "significant here, not there"
is not a test of the difference between them. And left_right is flat in every cell
(p = 0.860 at bf16, p = 0.824 at INT4), so whatever this is, it is specific to numeric
distance.

### The mechanism is not established

The obvious hypothesis — merged weights quantise worse — **does not survive measurement.**
Relative AWQ reconstruction error over 30 modules across 5 layers:

| weights | mean relative error |
|---|---:|
| base | 0.14901 |
| merged | 0.14677 (−1.51%) |

Merged is worse in 22 of 30 modules, but by so little that the mean moves the other way. At
the weight level the merged checkpoint quantises **just as well** as the base. So the
weight-space explanation is out, and the remaining candidates — AWQ scales fitted against
calibration *activations*, or the far bucket simply being the noisiest 122 items in the set —
are untested. Recorded as unexplained rather than guessed at.


## UNKNOWN on held-out nonsense — the hole the code cannot close

The brain session's `command_intent.py` gates two things in code: replies outside the
eight-word vocabulary, and replies naming two distinct actions. Neither can catch **a
confident, well-formed, single, wrong vocabulary word** — nothing downstream tells that from
a right one, and one measured case (`"What is the capital of France"` → SEARCH) sets off a
full 9-look, 8-turn sweep.

Both engines, production prompt from `prompts.py`, thinking off:

| utterance | base | SFT |
|---|---|---|
| Sing me a song | SEARCH | **UNKNOWN** |
| What is the capital of France | REPORT | **UNKNOWN** |
| Make me a sandwich | SEARCH | **UNKNOWN** |
| Tell me a joke | SEARCH | **UNKNOWN** |
| What time is it | REPORT | REPORT |
| Who won the world cup | REPORT | REPORT |
| Translate this to German | REPORT | REPORT |
| Set an alarm for 7am | STOP | **UNKNOWN** |
| How much do you weigh | REPORT | **UNKNOWN** |
| Write me a poem about rubble | SEARCH | **UNKNOWN** |
| What is 17 plus 25 | REPORT | **UNKNOWN** |
| Play some music | STOP | **UNKNOWN** |

**Base 0/12. SFT 9/12.**

Six of those twelve turned out to be in the training data — this probe was written by hand,
not drawn from the held-out split, so it has to be split before it means anything:

| subset | n | base | SFT |
|---|---:|---:|---:|
| **unseen** | 6 | 0 | **5 (83%)** |
| seen in training | 6 | 0 | 4 (67%) |

The held-out half scores *higher* than the trained half, which is the opposite of a
memorisation artefact — at n = 6 per cell that is not a result, but it does mean the effect
is not contamination.

Two of the three residual failures (`"What time is it"`, `"Translate this to German"`) were
**in the training set and still fail**, so this is not a coverage gap that more of the same
data fixes. All three failures share one shape: an information request answered with REPORT.
REPORT is "say the current status", and the model reads any question as a status request.
That is the specific next target, and it needs contrastive pairs — question-shaped nonsense
against question-shaped genuine status requests — not more volume.

No over-refusal: six real commands, phrased unlike anything trained
(`"Kill power to the wheels"`, `"Push on ahead"`, `"Swing to the left"`, `"Where are you
right now"`, `"Sweep the area"`, `"Reverse out of there"`) score **6/6 on both engines**, and
the SFT wrongly refused **0 of 6**. The refusal behaviour did not come at the cost of the
commands.

## Phase 1b — the grounded-schema bucket

Generated by `gen_schema_data.py` against the **real** vocabularies from
`brain/perception/cosmos_vision.py`, supplied by the brain session:

| field | values |
|---|---|
| `scene_type` | room, corridor, doorway, open_area, unknown |
| `task_state` | no_target, candidate, target_visible, uncertain |
| `task_relevance` | target, distractor, uncertain — **enum** |
| `intent` | continue_search, inspect_candidate, approach_candidate, stop |
| `attributes[]` | free-form `{name, value}`, `certainty` ∈ {observed, uncertain} |
| `label` | open, "concise visible category" |
| `bbox_2d` | `[x1,y1,x2,y2]`, integers 0–1000, origin top-left |

`task_relevance` is the one with a downstream numeric consequence — the consumer maps it to
confidence (target 0.75, uncertain 0.45, distractor 0.30) and `perception.cosmos_min_role`
keeps `{target, uncertain}`, so **`distractor` removes the entity from what the robot acts
on**. Omitted, it defaults to `uncertain`. That default is the guess this bucket exists to
train away, and it is part of why precision is 0.529.

Source is our own `isaac-sdg-rescue-target-v3` (Isaac Sim, public, CC-BY-4.0). Not in
`holdout_never_train`: the 24-item grounded benchmark and the 60 frozen frames are real
lw-lab1 footage, disjoint from these synthetic renders.

150 examples, mean 2.22 entities. `task_state`: 67 target_visible, 45 candidate, 38 no_target.

### Three things this bucket assumes, and one it cannot cover

**The task semantics are this script's invention.** COCO carries pose, vest, distance and
visibility; it does not say what counts as a target. The policy — incapacitated posture →
`target`, standing or walking → `uncertain` (a person, possibly a responder, never
`distractor`), scene clutter → `distractor`, anything past 25 m downgraded to `uncertain`
with uncertain attribute certainty — is recorded at the top of the script so it can be
overruled in one edit.

**The distractor skew was capped, on safety grounds.** Taking the 8 largest boxes per image
gave **552 distractors to 67 targets**. Since `distractor` is what removes an entity from the
robot's world, an 8:1 prior toward it is the one imbalance here that can get a casualty
ignored. Capped at 2 clutter boxes per example: now 221 distractor to 112 person entities.

**Clutter is labelled `"small object on the floor"`, deliberately vague.** The renders place
6 props per scene (DexCube, KLT bin, pallet, forklift) at scale 0.1–0.4, but COCO gives every
one `category_id 0`, so the instance really is unknown. A specific name would train the model
to invent an identity it cannot see.

**Coverage gap: `corridor` and `doorway` are never trained.** The Isaac environments are
halls, warehouses, offices and hospitals — 85 room, 65 open_area, nothing else. Two of five
`scene_type` values have no examples, and real frames are needed for them.

### Not yet trainable

`train_lora.py` is text-only. This bucket is image-conditioned, so the trainer needs a
multimodal path — feeding pixels through the processor and masking loss to the JSON — before
any of it can be used. That is the next piece of work, not a finished artefact.

## v3 — retrained under the production prompt, and the gate clears

`sft/prompt.py` now holds the production command prompt verbatim (`brain/prompts.py:74-96`),
imported by both trainer and evaluator, including the `Operator: "…"  Action:` user shape.
Plus contrastive REPORT/UNKNOWN pairs, and all 12 refusal probes moved into `HOLDOUT`.

### The placeholder prompt was costing 16 points, to the base model

| arm | overall | UNKNOWN |
|---|---:|---:|
| base, placeholder prompt | 51.4% | 40.8% |
| base, **production** prompt | **67.5%** | **10.8%** |
| v3 LoRA, production prompt | **97.0%** | **89.2%** |

Every number before this was measured against a strawman prompt. Note the direction of the
UNKNOWN column: the production prompt makes the base **worse** at refusing — 40.8% → 10.8% —
which is not a defect but the direct consequence of eight actions, worked examples, and
deliberately no UNKNOWN example. The better the prompt gets at the seven real actions, the
harder it pushes toward picking one, and the fine-tune is the only thing that recovers
refusal. It corroborates the brain session's independent 0/12.

### Refusal probe, this time genuinely held out

| | base | v2 | v3 |
|---|---:|---:|---:|
| UNKNOWN on nonsense | 0/12 | 9/12 (6 items trained on) | **11/12 (0 trained on)** |
| real commands | 6/6 | 6/6 | 6/6 |
| wrongly refused | 0/6 | 0/6 | **0/6** |

`"Who won the world cup"` and `"Translate this to German"` both flipped to UNKNOWN — the two
the contrastive pairs targeted. The single remaining miss is `"What time is it"` → REPORT,
which is genuinely ambiguous for a robot carrying a clock.

Dev set through the INT4 engine: **95.8%** (97.0% at bf16).

### The L0 gate

| arm | k | distance | left_right | mcq | far [12,+) |
|---|---:|---:|---:|---:|---:|
| INT4 base | 1.2148 | 36.21% | 64.00% | 16.45% | 39.3% |
| INT4 v2 | 1.2234 | 30.45% | 64.40% | 17.76% | **23.0%** |
| **INT4 v3** | 1.2154 | **34.57%** | 66.00% | 16.67% | **37.7%** |

Paired against base, v3 is flat everywhere: distance p = 0.428, left_right p = 0.143,
mcq p = 1.0. `k` returns to the base's value.

**What this does NOT establish.** v2 and v3 differ in several ways at once — production prompt,
contrastive pairs, a different seed-disjoint split, 270 steps against 264 — so the recovery
cannot be attributed to any one of them, and "v2 regressed, v3 did not" is **not a test of the
difference between them**. Two adapters, one draw each. The v2 regression could have been
run-to-run variance in which weights the LoRA moved, and a third seed is what would separate
those. Until then the honest claim is narrow: **this build clears the gate**, not "the problem
is solved".

## Sampling: the bench was greedy, production is not

`llm_inference` **silently ignores** `temperature`, `top_p` and `top_k` in its input JSON.
The two mentions of `temperature` in `examples/llm/llm_inference.cpp` are both comments
telling you to put it there; nothing assigns `request.temperature`. The runtime supports
sampling (`shouldUseNonGreedySampling`, `SamplingParams`) — only the example driver is
missing the parse. Filed as **NVIDIA/TensorRT-Edge-LLM#211**.

Confirmed by measurement, not by reading: the same open-ended prompt at `temperature` 0.0 and
0.3, three runs each, produced **one distinct output across all six**.

So every engine number in this file is a **greedy** number, while the brain runs
`llm.temperature: 0.3`. That gap has teeth — the sim session measured this engine entering a
non-terminating repetition loop at 0.3 on open-ended generation that greedy never enters, and
raising the budget 250 → 512 → 2048 extended the loop (24 → 83 → 424 steps) without moving
the first incoherent step past 12.

`experimental.server.LLM` does honour temperature, but it is the checkpoint-direct builder —
the path that produces numerically wrong INT4 output (#208) — so it cannot give a valid
sampled INT4 measurement either.

### What could be measured: the merged bf16 model under real sampling

| arm | dev (166) | differs from greedy |
|---|---:|---:|
| bf16 merged, greedy | 97.0% | — |
| bf16 merged, temp 0.3, seed 1 | 97.0% | **0** |
| bf16 merged, temp 0.3, seed 2 | 97.0% | **0** |
| bf16 merged, temp 0.3, seed 3 | 97.0% | **0** |

**0 of 166 items unstable across the three sampled runs**, and not one differs from greedy.
The refusal probe is identical at every temperature too (12/12 nonsense at bf16), with 0
wrongly-refused real commands.

That is what a peaked distribution looks like: training loss settles near 1e-4, the answer is
a single token from an eight-word vocabulary with `max_new_tokens` 8, and `temperature` 0.3
is nowhere near enough to move the argmax. The open-ended looping the sim session found needs
hundreds of sampled tokens to develop; this task never generates more than one.

**Limits, stated.** This is the **bf16 merged** model, not the INT4 engine — no local runtime
both honours temperature and builds INT4 correctly, so the INT4-at-0.3 measurement needs the
brain's shim. And one real command is refused at bf16 (5/6) that the INT4 engine gets right
(6/6), which is a bf16-vs-INT4 difference at n=6 and not worth a conclusion either way.

## The refusal ceiling is not hypothetical: Gemma reaches it untrained

Measured by the brain session on the same production prompt, thinking off, ten items through
the wired path:

| model | correct |
|---|---:|
| Gemma 4 E4B (llama.cpp, Q4_K_XL) | **10/10** |
| Cosmos3-Edge INT4-AWQ, base | 7/10 |

Gemma returns UNKNOWN on nonsense **with no fine-tune and no UNKNOWN example in the prompt**,
and separately answers REPORT correctly for `"Report your status"`. So it makes the exact
distinction our adapter still conflates on three items.

Two things follow, and they cut in opposite directions:

**It justifies the bucket.** The open question when we kept the 200 UNKNOWN examples was
whether "cannot decline" was a property of the task framing or of this model. A 4B-class model
reading the identical prompt declines correctly, so it is Cosmos, the behaviour is learnable,
and 0/12 → 11/12 is closing a real gap rather than teaching a trick.

**It sets a real target instead of a guess.** 10/10 is what this command path looks like with
refusal working. Our 11/12 is near it, not at it.

It also argues that the contrastive pairs matter more than rewording the REPORT line: another
model separates the two surface shapes from this prompt unchanged, so the distinction is
reachable without touching the description.

## Why every number here is a thinking-OFF number, verified

The production commitment gate returns UNKNOWN when a reply names two distinct actions. With
thinking ON, reasoning and answer arrive undivided in the text, so a model reasoning aloud
over two candidates parses as two actions and is scored UNKNOWN — which would penalise
whichever arm reasons more and inflate the delta.

Checked directly rather than assumed:

| call | prompt tail | thinking |
|---|---|---|
| `enable_thinking=False` | `<think></think>` | **off** |
| `enable_thinking=True` | `<think>\n` | on |
| flag omitted | `<think>\n` | **on** |

The flag is accepted, so all runs are thinking-off. But both scripts had
`except TypeError: <retry without the flag>`, which would have **silently enabled thinking**
had the template ever rejected it. That fallback now aborts instead.

Also documented rather than changed: the scorer scans `LABELS` order, not text order, so a
reply naming two actions resolves to whichever comes first in `LABELS` — unlike the production
gate, which returns UNKNOWN. Both arms are scored by the identical rule so the comparison is
fair, but a number here does not predict what `command_intent.py` would do with the same text.

## Scored under the production gate's rule: identical

Our scorer resolves a multi-action reply by `LABELS` order; `command_intent.py` returns
UNKNOWN. The brain session pointed out that difference is **not symmetric**: on refusal items
UNKNOWN is the correct answer, so the gate turns a multi-action reply into a hit where we
score a miss (our refusal number would be a *floor*); on real commands the same reply becomes
a miss where we may score a hit (our command number would be a *ceiling*). Refusal could only
rise, commands could only fall — and the command number is the deployment-relevant one.

Re-scored the v3 engine's raw outputs under the gate's rule — distinct = the seven real
actions named, UNKNOWN excluded from the set; exactly one → that action; zero → UNKNOWN;
two or more → UNKNOWN:

| measure | our rule | gate rule |
|---|---:|---:|
| refusal probe (12 nonsense) | 11/12 | **11/12** |
| real commands (6) | 6/6, 0 wrongly refused | **6/6, 0 wrongly refused** |
| dev set (166) | 95.8% | **95.8%** |

**0 disagreements across all 184 outputs. 0 replies naming two or more actions.**

So neither figure was a floor or a ceiling — both are exact. The reason is mechanical rather
than lucky, which is what makes it predictive: across the 166-item dev run there are **8
distinct output strings, the longest 7 characters**, and none contains more than one action
word. The two rules cannot disagree because the adapter never emits the input they disagree
about. On the brain session's own reading, that makes the commitment gate dead weight against
v3 — it still catches base Cosmos, but what protects a v3 deployment is the vocabulary gate
plus the trained refusal.
