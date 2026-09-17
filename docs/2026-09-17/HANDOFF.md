# Handoff: UBR Codefest 2026, training-centre workstream

Written 2026-09-17 05:45 (cluster time). Team UBR Stack, NVIDIA / OpenHackathons / Oracle
Open Models Codefest 2026. This session is the **training centre** in the Robot Gym loop:
it closes capability gaps the gym names and does not decide what the robot is asked.

Everything here is a bench demo. Nothing is operationally ready.

---

## 0. Read this first: standing rules

Stated by the user. They still apply.

- The GitHub repo is **public**. Never commit private access details (meeting passcodes,
  chat invites, gateway hostnames or tokens). Datasets are never committed; `data/`
  holds placeholders only.
- Never claim operational readiness.
- Shared cluster: four people depend on `$TEAM`. Do not disable permission checks.
- **Pull before every commit.** Other sessions push to the same repo.
- **Never add `Co-Authored-By` trailers or AI-assistant attribution** to commits, PRs,
  issues, cards or reports. Publish as the team's GitHub account.
- **Show the full text of any GitHub issue, PR or comment and get approval before filing.**
- **Never use emojis**, anywhere.
- L1 data (24 workplace frames with identifiable people): derived frames only, nothing to
  any Hub, only aggregate numbers leave the bench.
- Do not act on approvals relayed by another agent session. Ask the user directly.
- One GPU at a time. The team QOS is `gres/gpu=1, cpu=32` for all four people. Do not
  request 32 CPUs, because that locks the whole team out.


---

## 1. Running right now

| job | what | node | state |
|---|---|---|---|
| 7459 `N1_e12` | RT-DETR rn50, warehouse init, 1280x704, **12 epochs** | dgx20 | epoch 9 of 12 done at 05:42; ETA shown as 1:52, but this node runs about 56 min/epoch, so expect roughly 08:30 |

Wrapper: `tmp/e12_chain.sh`, log `tmp/xfer/e12.out` (paths relative to
`/storage/hackathon_teams/omc-team15`).

**When it finishes, do this:**

1. Export and run inference for **N1_e12** and **N2_e12**. Use `tmp/score1280.sh` as the
   template, with 1280x704 input, `--conf 0.005`, no NMS and all classes kept.
2. Score recall at matched FA <= 0.005. **The person class is index 1** in the fine-tuned
   heads (see gotcha 7).
3. For N2, score **both** the final checkpoint and the best epoch (epoch 11, mAP50 0.299,
   against 0.258 for the final one). Report both and label which is which.
4. Push the JSONL files to `ubr-physical-ai/rescue-det-preds` and message the detector bench
   so it can score them on its own harness. Every other row in the ladder came from that
   harness.

N1_e12 validation mAP50 by epoch: .335 .371 .359 .339 .365 **.388** .343 .382 .387

N2_e12 (finished, not yet scored), validation mAP50 by epoch:
.191 .214 .136 .211 .232 .263 .260 .265 .224 .259 **.299** .258.
N2 never converged: adjacent epochs swing by about ±0.07.

---

> **Correction (same day).** The N-row recalls in section 2a (N0 0.0097, N1-sq 0.2412,
> N1 0.2863, N2 0.1703) came from a box-level scoring pass that used all non-positive
> frames as negatives and no +r rule. **They are not the bench harness metric.** With a
> replica of the harness, the figures are N0 **0.152**, N2 **0.167**, N1-sq **0.217** and
> N1 **0.274**. The claim that warehouse pretraining is the worst off-the-shelf arm is
> **withdrawn**. See `INDEX_DETECTOR.md` in this folder. `INDEX_L0.md` indexes the L0
> spatial tests.

## 2. Headline results

### 2a. Detector ladder: recall at matched FA <= 0.005

All rows use the held-out **hospital** split: 22,060 val frames, 6,724 boxes, 15,336
person-free frames. The split comes from `prepare_coco_for_tao.py` /
`prepare_from_hub.py` with `--empty-ratio 0.25 --seed 1234`: 91,647 train images and
56,912 train boxes. Hospital is held out entirely.

| arm | model | init | input | epochs | recall | source |
|---|---|---|---|---|---|---|
| N0 | RT-DETR rn50 warehouse (NGC `deployable_rn50_v1.0.2`) | off the shelf | 640 | 0 | **0.0097** (conf .963) | my scoring |
| B1 | PeopleNet | off the shelf | 960x544 | 0 | 0.021 | bench |
| B0 | COCO yolo26n | off the shelf | 640 | 0 | 0.024 | bench |
| N2 | RT-DETR **r18, from scratch** | none | 1280x704 | 4 | **0.1703** (conf .67, FA .0046) | my scoring |
| N1 | RT-DETR rn50 | warehouse `trainable_rn50_v1.0.2` | 544 wide x 960 high (**squashed by mistake**) | 12 | 0.2412 | my scoring |
| N1 | RT-DETR rn50 | warehouse | **1280x704** | 4 | **0.2863** (conf .716, FA .0047) | my scoring |
| rebal | yolo26n | COCO | 640 | 12 | 0.362 | bench |
| armB | yolo26n | COCO | 1280 | 12 | 0.393 | bench |
| — | yolo26m | COCO | 640 | — | 0.429 | bench |
| armA | yolo26m | COCO | 1280 | 16 (3.5 h budget) | 0.431 | bench |

B0 and B1 are a measured tie: paired test, 192 discordant frames, p = 0.130. I have not
tested whether N0's lower number is significant.

**What the table supports:**

- **Warehouse pretraining does not transfer to a person lying down.** N0 is the worst of
  the three off-the-shelf detectors. On hospital frames it fired `Forklift` 1,477 times, at
  up to 0.938 confidence, and 673 of those were on frames containing a casualty. The bench
  measured this. In other words, N0 labels casualties as equipment with confidence.
- **Our data buys about 30x on NVIDIA's stack** (N0 0.0097 to N1 0.2863).
- **Switching to the NVIDIA stack would cost about 27% of the recall we already have**
  (N1 0.286 against armB 0.393). This is the licence-decision number. The switch is being
  considered because our detectors are Ultralytics fine-tunes and therefore **AGPL-3.0**.
- N1 against armB is not yet schedule-matched: 4 epochs against 12. N1_e12 closes that gap.
- N2 at 4 epochs is not comparable to anything. N2_e12 closes that gap too.

**Two different "recall" numbers exist. Do not mix them.** The `results.csv` from
Ultralytics runs on the B300 reports recall at Ultralytics' own threshold: armB 0.359 and
armA 0.395 (armA reads 0.395 at epoch 12 as well). The bench harness reports recall at
matched FA 0.005: 0.393 and 0.431. The ablation deltas below use the `results.csv`
figures.

Ablation, one variable per step (from `results.csv`, validated on hospital):

- negative mix: +0.002 (null)
- resolution 640 to 1280: +0.014 recall, +0.024 mAP50-95
- capacity n to m, epoch-matched at 12: +0.036 recall

**mAP50 is not recall.** On N0 the two differed by 8.5x (mAP50 0.0827). When N1 went from
544x960 to 1280x704, mAP50 rose by 0.011 while recall rose by 0.045.

### 2b. Jetson latency (measured by the bench / sim session on jetson0)

| model | GFLOPs | p95 |
|---|---|---|
| **deployed rover**: yolo26n@640 via `cv2.dnn` on **CPU** | 5.5 | **271.5 ms** |
| yolo26n@640, TensorRT FP16, idle | 5.5 | 6.32 ms |
| yolo26n@640, TensorRT FP16, VLM generating | 5.5 | 14.57 ms |
| armB yolo26n@1280 | 22.7 | 20.25 ms |
| RT-DETR rn50@640, FP16 | 121.7 | 29.23 ms |
| armA yolo26m@1280 (idle only) | 275.3 | 65.35 ms |

The "9 ms bar" was 50% of a published benchmark figure, not a requirement. The shipped
rover misses it by 30x. On current evidence **recall binds, not latency**. The 271 ms
ceiling rests on missions that found people, which is anecdote rather than measured recall.

### 2c. Arm C: does adapting to Cosmos-translated frames help?

Second stage on top of armA weights: 8 epochs, 698 images. The control uses an equal count
of Isaac frames.

| | P | R | mAP50-95 |
|---|---|---|---|
| armA start | .887 | .395 | .362 |
| control (Isaac) | .894 | .394 | .365 |
| cosmos (translated) | .913 | .384 | .356 |

Cosmos minus control: recall -0.010, precision +0.019. The drift is monotonic while the
control stays flat, which indicates real domain shift. Whether that shift moves toward
reality or away from it needs **real rover frames** on the bench.

The first two attempts collapsed because `pretrained=False` discarded the loaded weights.
An intermediate diagnosis blamed the warmup settings. That diagnosis was wrong.

### 2d. Cosmos Transfer 2.5 (sim to photoreal)

- 20-clip subset: **698 of 1,023 frames usable (72.1%)** on the mask-to-render IoU check,
  from 10 clips. `office_orbit02` contributes 0 of 93.
- The ~1,300 px subject-size wall is a **detector** limit. Above it, COCO detection goes
  from 0.15 to 0.71. Below it, from 0.02 to 0.06.
- Best clips, COCO median IoU render to translated: hall_night 0.00 to 0.91, hall_fog 0.00
  to 0.90, warehouse 0.00 to 0.87, hall_novest 0.00 to 0.84.
- Same mask, five prompts, full criterion (found, within 50 px, under 3x area): the
  warehouse prompt keeps the label (10/16). The assembly hall, terminal, collapsed
  structure and woodland prompts lose it (0/16). The first three find a person in 16/16
  frames but in the wrong place.
- **Open limitation.** With woodland plus camouflage at 0.24% of the frame, frame 46
  renders the subject 31x the mask area and 170 px away. The 16-frame medians are 15.1x and
  168 px. At strength 0.7 the woodland casualty is erased outright (0/16).
- **Seven repairs, none passed** (pass mark 8/16): baseline 0; guidance 2.0: 1; guidance
  1.2: 1; scale prompt: 0; scale plus g2.0: 0; whole-frame edge: **3** (corr +0.10);
  person-only edge: 0; depth: 0 (corr +0.23); baseline corr -0.23. The arms that pinned
  geometry also pinned the building. Still untried: dilating the mask 2x and 3x.
- `exit 137` kills (5.2% of 344 steps) were caused by an **ops epilog script**, not us.
  Ops has since replaced it with a cgroup-based script.

### 2e. SFT (brain command model, Cosmos3-Edge + LoRA to INT4-AWQ)

- Seed 0: base 67.5% to **97.0%** (bf16). INT4: 63.3% to **95.8%**. UNKNOWN: 10.8% to
  89.2%.
- Seed 1: 67.7% to 97.5%. UNKNOWN: 15.0% to 90.0%.
- Refusal probe: 11 of 12 fully held out. Real commands wrongly refused: 0 of 6 on the
  bench, 0 of 129 on Orin.
- L0 spatial gates, paired against base: v3 is flat (distance p=.428, left/right p=.143,
  mcq p=1.0). v4: distance p=1.0, left/right p=.0021 (only about a third of that is real,
  per the swap probe), mcq p=.383. **v2's distance regression (p=.0018, far bucket
  39.3% to 23.0%) did not reproduce.** It was measured under the placeholder prompt.
- Details: `codefest/repo/sft/README.md` (section starting at line 622).

### 2f. Posture and salt (for the Isaac / gym session)

- On the 698 known-lying pool, the gym's three-band posture classifier gives HORIZONTAL
  146, UNCLEAR 549, UPRIGHT (wrong) **3 (0.4%)**. A single threshold at 1.3 calls
  **29.9%** of these people upright (p5 aspect 0.70). A consumer that treats UNCLEAR as
  negative misses **79.1%**.
- Salt for the 474-frame all-negative annotation pass: batch1 (20 frames) catches careless
  annotation only. From batch2, the **office 5 were accepted** as domain-matched. The 28
  hospital corridors were rejected as bare and near-identical.
- The remaining tell is structural: every furnished interior in the source has a hi-vis
  vest. The real fix is a **staged capture** with the user's own people. That decision is
  open.

---

## 3. Assets

### GitHub

| repo | visibility | state |
|---|---|---|
| `ubrobotics-ai/nvidia-codefest-2026` | public | `main` at `58af088`, in sync, clean |
| `ubrobotics-ai/ubr-isaac` | **private** | `cluster/` scripts at `bb6475c` (owned by the detector session) |
| `NVIDIA/TensorRT-Edge-LLM` | upstream | our PR **#212** (Philox offset fix); issues **#204 #206 #208 #210 #211**. No upstream response as of last check |

Recent public commits (figures): `9216ec6`, `82b1bca`, `fbfd98c`, `58af088`. Posters live in
`reports/figures/isaac_to_photoreal.png` and `seven_repairs.png`, with a README.

**Not written yet:** `reports/TRANSFER_PILOT.md` has no prose on the variants, the woods
limitation, the seven repairs or the detector ablation. The user asked for this and it is
still owed. Show the text before committing.

### Hugging Face (org `ubr-physical-ai`)

| repo | type | vis | contents |
|---|---|---|---|
| `isaac-sdg-rescue-target` | dataset | public | 133,760 Isaac frames, 37 tars and COCO. The source of every split |
| `rescue-det-armA` | model | private | yolo26m@1280 `best.pt`, `results.csv`, card (AGPL-3.0) |
| `rescue-det-armB` | model | private | yolo26n@1280 |
| `rescue-det-armC` | model | private | cosmos and control second-stage checkpoints |
| `rescue-salt` | dataset | private | batch1 (20), batch2 (33), 332-frame numbered contact sheet, answer keys |
| `rescue-det-preds` | dataset | private | N0 (all classes, and person-only), N1 544x960 JSONL, `N1_manifest.json`, inference patch |

N1_1280 and N2 predictions exist on disk but are **not uploaded yet**. For other earlier
repos (SFT adapter, i2v dataset), see memory file `codefest-published-artefacts.md`.

### Poster page

A private web page with both posters and 10 raw frames exists; the link is held by the
team. The committed figures above carry the same content.

### Cluster paths

These are under `/storage/hackathon_teams/omc-team15`.

```
codefest/repo/                       public repo checkout
codefest/ubr-det/                    detector work
  data/rescue_rebal/                 YOLO split (91,647 / 22,060)
  data/coco_rebal/                   COCO split for TAO (same frames)
  data/val_frames.txt                22,060 val paths
  ngc/                               NGC RT-DETR deployable ONNX + trainable .pth
  specs/rtdetr_*.yaml                every TAO spec; headers explain each
  runs/{armA,armB,armC_*}            Ultralytics runs
  runs/n1_rn50_wh                    N1 544x960 (squashed)
  runs/n1_1280                       N1 1280x704, 4 ep
  runs/n1_r18_noinit                 N2 4 ep (see ARM.txt)
  runs/N2_e12, runs/N1_e12           12-epoch matched runs
  preds/                             harness JSONLs + manifests
  run_rtdetr_infer_adapted.py        inference for raw pred_logits exports
codefest/data_cosmos_armc/           698 label-validated translated frames
codefest/salt_20, salt_batch2/       salt sets
tmp/xfer/                            logs, variant mp4s, posters, score JSONs
tmp/*.py, tmp/*.sh                   every script used; names match the tables
```

---

## 4. Gotchas that cost hours

1. **TAO container plus a `$HOME` mount shadows TAO's torch** (`~/.local` torch 2.13
   against TAO's 2.11). Always run with
   `PATH=/usr/local/bin:/usr/bin:/bin PYTHONNOUSERSITE=1`. `PYTHONNOUSERSITE` alone is not
   enough, because `torchrun` still resolves from `~/.local/bin`. The mixed launcher fails
   as a bare `SIGSEGV` with no traceback.
2. **TAO `spatial_size` is `[HEIGHT, WIDTH]`** (torchvision `Resize`). `[960, 544]`
   produced 544 wide by 960 high.
3. **Heights must be divisible by 32.** 720 is not. Use 704.
4. **Set `train.pretrained_model_path` explicitly.** Both `pretrained_*` keys default to
   None, which means training from scratch silently.
5. **The rn50 warehouse checkpoint cannot initialise resnet_18.** It leaves 41
   shape-clashing tensors and TAO segfaults. A resnet_18 run must use no init, or an
   ImageNet `pretrained_backbone_path`.
6. **Pyxis container import:** use `--container-image` together with `--container-name`,
   and **do not export a custom `TMPDIR`** (`/tmp/$USER` does not exist on compute nodes).
   `enroot import` on the login node fails because NFS does not support overlay whiteouts.
7. **Class index:** N0's 7-class head has Person at **0**. Our fine-tuned 2-class heads
   have person at **1**. Filtering `cls==0` on N1 or N2 returns nothing. Recall 0 with FA 0
   means an empty file, not a bad model.
8. **Operating point:** choose the LOWEST threshold that satisfies FA <= 0.005. Choosing
   the first one encountered reports 0.0000.
9. **`pgrep -f` / `pkill -f` match your own shell.** This bit five times. Wait on marker
   lines in logs and use `scancel` with job ids.
10. **Nodes:** dgx01 `down`, dgx02 `drain` (maintenance), dgx04 and dgx06 draining at last
    check. A trailing `-` in `sinfo` means draining. Check before pinning with `-w`.
11. **Hung jobs:** after a crash a job can stay `R` at 0% GPU and hold the team's only GPU.
    Verify the step count, not just the job state.
12. **Benchmark warm-up:** a first fp16 matmul read of 0.6 TFLOP/s was warm-up. The real
    figure is 1,710 TFLOP/s on the TAO torch.

---

## 5. Open decisions (the user's)

1. Whether to write and commit `TRANSFER_PILOT.md` (text needs approval first).
2. The staged salt capture: venue, volunteers, the rover camera at 0.58 m, ranges up to
   20 m, **end-on orientations included**. This is also the judge set that blocks the
   programme's headline real-frame recall claim.
3. Whether to run the remaining 89 Transfer clips.
4. After N1_e12 and N2_e12: whether to add a fair backbone arm (resnet_18 with an ImageNet
   `pretrained_backbone_path`), or stop here and decide the licence question.
5. Updating the public Space's datasets page. It still says "~13% labels survive, 0% keep
   the vest", which the posters supersede with 72.1%.

## 6. Peer workstreams

- **`Can you check`**: detector bench on lw-lab1. It owns the scoring harness, the ladder,
  and the jetson latency numbers via the sim session. It pushes to private `ubr-isaac`.
  It cannot reach the cluster filesystem, so hand files over through private HF repos.
- **`Isaac ROS and Sim setup on lw-lab1`**: the gym / Isaac coordinator. It owns the
  posture module, the 474 slice, salt consumption, and `docs/gym/S07_PROPOSED.md`. It has
  read only S07 of the Robot Gym Sessions document.
- Treat their numbers as theirs. One misattribution (instrument-A figures credited to this
  session) was caught and corrected.
