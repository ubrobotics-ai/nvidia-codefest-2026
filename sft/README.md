# SFT working directory

Scoped by [`../docs/SFT_PIPELINE.md`](../docs/SFT_PIPELINE.md). Phase 0 and the unblocked
half of Phase 1 are done; the rest needs files that live on the robot.

| file | what |
|---|---|
| `BASELINE.json` | Phase 0. Artefact checksums, repo SHAs, and every measurement this SFT must not regress. Frozen before any weight changed. |
| `gen_command_data.py` | Phase 1. Generates the command-vocabulary and UNKNOWN buckets, refusing to write if the output would contaminate the eval. |
| `data/commands.jsonl` | 890 examples. |

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
