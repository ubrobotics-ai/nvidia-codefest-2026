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
