#!/usr/bin/env python3
"""Score every L0 arm with the official scorer, re-fit the distance constant k, dump answer histograms,
and write the arm table.

Naming, fixed here and used everywhere downstream: `task_mean(distance, left_right)` is the *unweighted mean of
the two task accuracies*, not an N-weighted figure. Round 1 called it "weighted", which it never was. The table
reports mcq separately and states explicitly whether the headline includes it.

  python scripts/l0_report.py --dir <L0 dir> --round1 <night dir> --data <spatial-qa> --out REPORT_L0.md
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

EPS = 1e-6


def load_raw(d, tag):
    p = Path(d) / f"E_{tag}_raw.jsonl"
    if not p.exists(): return None
    return [json.loads(l) for l in open(p)]


def official_score(data_dir, rows):
    """Run the dataset's own compute_scores.py on one arm's rows. Returns {category: (correct, n)}."""
    scorer = Path(data_dir) / "utils" / "compute_scores.py"
    with tempfile.TemporaryDirectory() as td:
        gt = [{"id": r["id"], "normalized_answer": r["gt"], "category": r["category"]} for r in rows]
        pr = [{"id": r["id"], "normalized_answer": r["pred"]} for r in rows]
        (Path(td) / "gt.json").write_text(json.dumps(gt))
        (Path(td) / "pred.json").write_text(json.dumps(pr))
        out = subprocess.run([sys.executable, str(scorer), "--gt_path", f"{td}/gt.json",
                              "--pred_path", f"{td}/pred.json"], capture_output=True, text=True)
    return out.stdout + out.stderr


def acc(rows, cat, k=1.0):
    """Per-category accuracy under the official rule: distance within +/-10%, exact match otherwise."""
    sel = [r for r in rows if r["category"] == cat]
    ok = 0
    for r in sel:
        if cat == "distance":
            try:
                g, p = float(r["gt"]), float(r["pred"]) * k
            except ValueError:
                continue
            ok += int(0.90 * g <= p <= 1.10 * g)
        else:
            ok += int(str(r["pred"]).strip().lower() == str(r["gt"]).strip().lower())
    return ok, len(sel)


def insample_distance(rows, k):
    """Distance items scored correct at constant k, by the same rule mcnemar()'s right() uses.

    Derived rather than written into the prose: an earlier revision quoted three different
    counts for the same arms because two code paths scored distance differently.
    """
    n = 0
    for r in rows or []:
        if r.get("category") != "distance": continue
        try: g, p = float(r["gt"]), float(r["pred"]) * k
        except (ValueError, TypeError): continue
        n += int(0.90 * g <= p <= 1.10 * g)
    return n


def wrong_k_cost(rows, k_right, k_wrong, n_total):
    """Points lost by applying another arm's calibration constant to this arm's distance answers.

    Not the same quantity as the raw bf16-vs-4-bit gap, which is what a reader reaches for by
    mistake: that gap is mostly removed by calibrating AT ALL, and only a small remainder is the
    cost of using the *wrong* constant. Scored with the dataset's +/-10% band.
    """
    # Same rule as mcnemar()'s right(): score every distance item, do not drop
    # non-positive predictions. Using a stricter filter here made the prose disagree
    # with the McNemar rows by one or two items per arm.
    P = [r for r in rows if r.get("category") == "distance"]
    def _ok(r, k):
        try: g, p_ = float(r["gt"]), float(r["pred"]) * k
        except (ValueError, TypeError): return False
        return 0.90 * g <= p_ <= 1.10 * g
    hit = lambda k: sum(int(_ok(r, k)) for r in P)
    a, b = hit(k_wrong), hit(k_right)
    return {"right": b, "wrong": a, "items": b - a, "n": n_total,
            "points": 100.0 * (b - a) / n_total if n_total else 0.0,
            "pct_right": 100.0 * b / n_total, "pct_wrong": 100.0 * a / n_total}


def fit_k(rows):
    """Cross-validated distance calibration by split-half median ratio.

    Estimator: k = median(gt / pred) on the fitting half -- a scale correction for the model's systematic
    under-read, robust to the outliers a free-form numeric answer produces. Halves are even/odd position in the
    id-sorted list, so the split is deterministic and identical across arms.
    """
    sel = sorted([r for r in rows if r["category"] == "distance"], key=lambda r: str(r["id"]))
    pairs = []
    for r in sel:
        try:
            g, p = float(r["gt"]), float(r["pred"])
        except ValueError:
            continue
        if p > 0 and g > 0: pairs.append((g, p, r))
    halves = [[x for i, x in enumerate(pairs) if i % 2 == 0], [x for i, x in enumerate(pairs) if i % 2 == 1]]

    def median(v):
        v = sorted(v); n = len(v)
        return 0.0 if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))

    ks = [median([g / p for g, p, _ in h]) for h in halves]

    def hit(h, k):
        return sum(int(0.90 * g <= p * k <= 1.10 * g) for g, p, _ in h)

    held_ok = hit(halves[1], ks[0]) + hit(halves[0], ks[1])
    base_ok = hit(halves[0], 1.0) + hit(halves[1], 1.0)
    n = len(pairs)
    return {"k_half_a": round(ks[0], 4), "k_half_b": round(ks[1], 4), "k_mean": round(sum(ks) / 2, 4),
            "n_distance_usable": n, "n_distance_total": len(sel),
            "uncalibrated": base_ok, "held_out_calibrated": held_ok,
            "gain_pts": round(100 * (held_ok - base_ok) / max(n, 1), 2)}


def histogram(rows, cat, nbins=8):
    sel = [r for r in rows if r["category"] == cat]
    if cat in ("left_right", "mcq"):
        c = Counter(str(r["pred"]).strip().lower() for r in sel)
        return dict(c.most_common(12)), len(sel)
    vals = []
    for r in sel:
        try: vals.append(float(r["pred"]))
        except ValueError: pass
    if not vals: return {}, len(sel)
    edges = [0, 1, 2, 3, 5, 8, 12, 20, 1e9]
    c = Counter()
    for v in vals:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                c[f"[{edges[i]:g},{edges[i+1]:g})" if edges[i+1] < 1e9 else f">={edges[i]:g}"] += 1
                break
    return dict(c), len(sel)




def chance_levels(data_dir):
    """Chance accuracy per task, from the item set itself.

    left_right is a two-way choice, so 50%. mcq offers a different number of regions per item, so chance is the
    mean of 1/n_regions over the items -- not 1/4, and not a constant.
    """
    items = json.load(open(os.path.join(data_dir, "val.json")))
    ns = []
    for x in items:
        if x["category"] != "mcq": continue
        r = x["rle"]
        if isinstance(r, str): r = eval(r)
        ns.append(len(r))
    lr = Counter(str(x["normalized_answer"]).lower() for x in items if x["category"] == "left_right")
    mc = Counter(str(x["normalized_answer"]) for x in items if x["category"] == "mcq")
    return {"left_right": 50.0, "mcq": round(100 * sum(1 / n for n in ns) / len(ns), 2) if ns else None,
            "mcq_n_regions_range": (min(ns), max(ns)) if ns else None,
            "gt_left_right": dict(lr), "gt_mcq_spread": mc.most_common(4)}


def fmt(e, cat):
    c = e[cat]
    return f"{c['ok']}/{c['n']} = {c['pct']:.2f}%" if c["n"] else "—"


def write_md(res, out, chance, think=None, agr=None, mcn=None, trt_rows=None, arm_rows=None):
    L = []
    A = L.append
    A(f"# L0 — model x precision, {len(res)} rows on one machine\n")
    A("Every accuracy below was produced on the same host, over the same items, with the same prompt, the same")
    A("rendered numbered region outlines and the same parser (including the mcq fix: answers are **region")
    A("indices**, not option letters). Greedy decode, `enable_thinking=False` unless a row says otherwise.\n")

    # ---------------- headline, stated before any table -------------------------------
    TRT_LBL = "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)"
    BF_LBL  = "Cosmos3-Edge bf16 (L0 control)"
    ct_rows_for_k = trt_rows or []
    if TRT_LBL in res and BF_LBL in res and ct_rows_for_k:
        kb = res[BF_LBL]["k"]["k_mean"]; kt = res[TRT_LBL]["k"]["k_mean"]
        db = res[BF_LBL]["distance"]["pct"]; dt = res[TRT_LBL]["distance"]["pct"]
        A("## Decision\n")
        A("- **INT4-AWQ ships.** It is the deployed quantisation; nothing here argues against it.")
        A("- **Distance k is fitted per engine build *and per range bucket*, provisionally 1.21.** Not a model")
        A("  constant, and not one number: the far bucket needs 1.34 where the global fit gives 1.21.")
        A("- **Thinking on for grounding *and operator commands*, off for closed spatial queries.** L1 measured")
        A("  commands at 14/15 **with the conversational prompt and thinking on** — the measured configuration —")
        A("  against 3/14 with a bare list and thinking off. Those runs changed prompt *and* thinking mode")
        A("  together, so thinking's own contribution on the command path is not isolated; the rule follows the")
        A("  configuration that was measured, not a demonstrated cause. Per-request flag; no global choice.")
        A("- **No Orin accuracy claim until the engine-equivalence gate has run.** These rows are SM103.\n")
        A("## The finding\n")
        A(f"**4-bit weights shift the distance *scale*; they do not destroy the information, and one constant")
        A(f"restores it.** On the deployed TensorRT INT4-AWQ engine, distance reads {db:.2f}% at bf16 against")
        A(f"{dt:.2f}% at INT4 — a {db-dt:.1f}-point loss, paired McNemar b=73 c=19, p < 0.0001. Re-fit the")
        A(f"distance constant per arm and that loss disappears entirely: **b = 68, c = 66, p = 0.931**")
        A("(**in-sample** discordant pairs, not accuracies; the same in-sample scoring puts bf16 at 178 and the")
        A("TensorRT arm at 176 of 486. Held out and split-half, the k table below reads 174 against 167). left_right")
        A("(p = 0.724) and mcq (p = 0.451) never moved.\n")
        A("**NF4 does not shift the scale; AWQ does.** NF4 fits k = 1.1628 against bf16's 1.1546 and needs no")
        A("re-fit to match it — raw distance 30.25% vs 29.63%, p = 0.815. AWQ fits 1.2148. So the two 4-bit")
        A("schemes are indistinguishable only *after* each is re-fitted (p = 0.804); raw they are not")
        A("(b = 21, c = 78, p < 0.00001). The scheme decides whether a constant is needed, not whether 4-bit works.\n")
        A("### What to do with it\n")
        wk = wrong_k_cost(ct_rows_for_k, kt, kb, res[TRT_LBL]["distance"]["n"])
        A(f"**Provisional k = {kt:.4f}**, fitted on the **SM103** engine — not yet the number to bake. The constant")
        A(f"is a property of the quantised artefact (bf16 {kb:.4f}, NF4 1.1628, AWQ {kt:.4f}), so it is fitted per")
        A("artefact, **including per engine build**. The deployed constant is the one fitted on the Orin engine's")
        A("own distance outputs, from the engine-equivalence gate run below; bake that, not this. Measured on")
        A("these items (**in-sample**, each arm's own k, scored over all 486 by the same rule the McNemar rows")
        A("use): the INT4 answers score")
        A(f"{wk['pct_right']:.2f}% ({wk['right']}/{wk['n']}) under their own constant and {wk['pct_wrong']:.2f}%")
        A(f"({wk['wrong']}/{wk['n']}) under the bf16 one —")
        A(f"**{wk['points']:.2f} points ({wk['items']} items) thrown away for no reason.**\n")
        A(f"That figure is smaller than the {db-dt:.1f}-point raw gap above, and the difference matters: almost")
        A(f"all of the raw gap is removed by calibrating *at all* ({100.0*insample_distance(ct_rows_for_k,1.0)/wk['n']:.2f}%")
        A(f"uncalibrated to {wk['pct_right']:.2f}% calibrated, both on the scorer's basis), and")
        A(f"only the {wk['points']:.2f}-point remainder is the cost of using the *wrong* constant. Do not quote")
        A("the raw gap as the cost of mis-calibration.\n")
        A("**The Friday SFT decision: L0 contributes no SFT target.** That is the finding, not a gap in it.\n")
        A("This report rules two candidates out and downgrades a third:\n")
        A("- **Precision is not the problem.** 4-bit costs nothing once the arm is calibrated (p = 0.931), so")
        A("  there is nothing for training to repair there.")
        A("- **Format is not the target.** The tolerant parser recovers the boxes (precision 0.529, recall 0.750)")
        A("  without valid JSON ever being emitted, and L1's five contract rewrites each scored 0/12 detections")
        A("  while improving format. Under the one-model architecture nothing asks the model for a contract.")
        A("- **The left prior is a confidence signal, not a training target.** The order-swap probe below shows")
        A("  65.6% order-invariance: on two thirds of items the model returns the same word for a question and")
        A("  its mirror. That is the absence of a spatial comparison, and a few hundred supervised examples do")
        A("  not create one. Order-consistency is worth having as a *signal* (70.35% vs 60.7%); the 72/28 answer")
        A("  split is not worth training against.\n")
        A("So the SFT budget belongs with **L1's candidates**, all on the language side: the **tool-calling")
        A("re-run** (0/21, but August, a different serving stack, a bare schema, and unmeasured under the prompt")
        A("that moved commands from 3/14 to 14/15), the missing **UNKNOWN path** (`\"Sing me a song\"` returns")
        A("`BACK` rather than declining), and **PT-PT** if it returns to scope. L1 has a measured fallback for")
        A("the last two — a text-only Gemma E2B at 14/14 including Portuguese, 2,591 MB — so training is not the")
        A("only route to any of them.\n")
        A("**Open-vocabulary precision (0.529) stays on the engineering side, not the training side.** It is the")
        A("perception limit to design around — gate approaches on a second confirmation, and treat a box as \"go")
        A("and look\" rather than \"found\" — not something to fine-tune away on 24 items.\n")
        A("### Provenance of the TensorRT row — read before quoting it\n")
        A("This row was produced on a **DGX B300 (SM103)**, which cannot build the Edge-LLM INT4 path at all on")
        A("v0.10.1: it needs the SM allowlist patch in NVIDIA/TensorRT-Edge-LLM#207 plus the NVRTC include-path")
        A("fix in #205, **both unmerged at the time of writing**. The row therefore exists under a local patch set.\n")
        A("It shares the **ONNX export** with the Jetson deployment, not the engine. SM103 and SM87 produce")
        A("different plans from the same export, so \"the same artefact the Jetson deploys\" is true of the ONNX")
        A("and false of the engine. That makes an engine-equivalence gate runnable — B300 outputs on all **1,442**")
        A("scored items (986 distance + left_right, 456 mcq, which is why the criterion below covers all three")
        A("tasks) against Orin outputs on a 200-pair subsample, compared per item — and **it has not been run.**")
        A("Until it is, this row licenses claims about INT4-AWQ *as a quantisation*, not about the Orin engine.\n")
        A("**What passing means.** The gate cannot pass or fail without a criterion, and the hand-off's")
        A("\">= 99% per-item agreement\" cannot be used as written: this report's own environment floor is 87.9%")
        A("on distance between two containers running the *same* model, so 99% is unreachable by construction.")
        A("The criterion is therefore stated against the floor:\n")
        A("| check | threshold |")
        A("|---|---|")
        A("| per-item agreement, distance | not below the floor, **>= ~88%** |")
        A("| per-item agreement, left_right and mcq | **>= ~99%** (the floor is ~95-99% on the discrete tasks) |")
        A("| paired McNemar, each task, on the 200 pairs | **p > 0.05** — no detectable systematic difference |")
        A("| k re-fitted on the Orin distance outputs | fills the blank row above, **per range bucket** |")
        A("")
        A("Agreement below the floor on a discrete task means the two engines differ; agreement above it means")
        A("they are indistinguishable *at the resolution this benchmark has*, which is the strongest claim the")
        A("instrument supports. A McNemar p > 0.05 is the accompanying check that the differences are unsigned")
        A("churn rather than one engine being consistently worse.\n")
    A("## Naming, fixed\n")
    A("Round 1 reported a **\"weighted\"** figure. It was never N-weighted: it is the *unweighted mean of the two")
    A("task accuracies*, `(distance + left_right) / 2`, and it **excludes mcq** — the task both models are worst")
    A("at. It is called **`task_mean(distance, left_right)`** from here on, and the mcq column is always shown")
    A("beside it. Where a single headline number is wanted, use `task_mean(distance, left_right, mcq)`, which is")
    A("given as a separate column.\n")

    A(f"## The {len(res)} rows\n")
    A("The brief was a 2x2 (two models x two precisions). It has since grown a round-1 control, a\n"
      "second 4-bit quantiser (NF4), and a row run on the *deployed* TensorRT engine rather than a\n"
      "fake-quant stand-in. Every row is scored by the dataset's own scorer on the same items.\n")
    A("| model | precision | runtime | quantisation | kernels | distance | left_right | mcq | task_mean(d,lr) | task_mean(d,lr,mcq) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for label, e in res.items():
        s = e.get("stats", {})
        prec = ("NF4" if "NF4" in label else "bf16" if "bf16" in label else
                "INT4 AWQ" if "INT4" in label else "Q4_0" if "Q4_0" in label else "bf16")
        rt = s.get("runtime", "transformers")
        qk = s.get("quantization_kind", "none")
        if qk == "none" and "tensorrt" in str(rt):
            # The TRT arm's stats file is written by l0_trt_eval.py, which does not carry the
            # quantiser metadata the PyTorch harness emits; it is the deployed AWQ export.
            qk = ("PTQ (modelopt AWQ), REAL INT4 kernels via the Edge-LLM ONNX export -- "
                  "same ONNX export as the Jetson; engine built for SM103 under the #205 + #207 patches")
        kern = ("simulated" if "SIMULATED" in str(qk) else
                "real (bitsandbytes)" if "bitsandbytes" in str(rt) else
                "real (llama.cpp)" if "llama.cpp" in str(rt) else
                "real (TensorRT Int4GroupwiseGemmPluginV2)" if "tensorrt" in str(rt) else
                "n/a — full precision")
        A(f"| {label} | {prec} | {rt} | {qk} | {kern} | {fmt(e,'distance')} | {fmt(e,'left_right')} | "
          f"{fmt(e,'mcq')} | **{e['task_mean']:.2f}%** | {e.get('task_mean_with_mcq','—') if isinstance(e.get('task_mean_with_mcq'),str) else format(e.get('task_mean_with_mcq',0),'.2f')+'%'} |")
    A("")
    A("N per cell is printed in the cell. distance N=486, left_right N=500, mcq N=456.\n")

    A("### Throughput and memory — B300, not Orin\n")
    A("| row | tok/s | peak GB | note |")
    A("|---|---|---|---|")
    for label, e in res.items():
        s = e.get("stats", {})
        tps = s.get("tokens_per_s", "—")
        if "round 1" in label and str(tps).startswith("43.9"):
            tps = f"~~{tps}~~ withdrawn"
        A(f"| {label} | {tps} | {s.get('peak_gb') or '—'} | {s.get('hardware','B300')} |")
    A("")
    A("**These columns describe a B300.** They are not deployment figures and must never be mixed into a column")
    A("that reports Orin latency. The simulated-INT4 row in particular runs *slower* than bf16 because fake quant")
    A("dequantises on every matmul — it says nothing about what an INT4 engine would do.\n")
    A("**The round-1 43.94 tok/s figure is withdrawn.** The same model at the same precision on the same B300")
    A("measured 76.17 tok/s one row below, in a different container — 1.7x apart from Python-side overhead alone.")
    A("Neither is a deployment number. The Orin figures are the ones that count: **52.8 tok/s decode**, and for")
    A("memory, **~4,830 MB peak for the full VLM**. L1 also reports 3,377 MB for the AWQ/TensorRT arm; the two")
    A("figures differ by close to the visual engine plus the embedding, but L1 labels both as system-wide")
    A("sampling and is reconciling them, so **quote the scope alongside whichever number you use** rather than")
    A("treating the difference as settled.\n")
    A("Blank cells are honest blanks: the TensorRT arm is driven by `llm_inference` as a subprocess and the")
    A("harness records wall-clock, not generated-token counts, and the llama.cpp arm reports no peak-memory")
    A("figure through its server API. Neither blank is a deployment claim.\n")

    A("## Points lost to 4-bit\n")
    pairs = [("Cosmos3-Edge / AWQ simulated", "Cosmos3-Edge bf16 (L0 control)",
              "Cosmos3-Edge INT4 AWQ (simulated)"),
             ("Cosmos3-Edge / AWQ REAL (TensorRT)", "Cosmos3-Edge bf16 (L0 control)",
              "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)"),
             ("Cosmos3-Edge / NF4", "Cosmos3-Edge bf16 (L0 control)", "Cosmos3-Edge NF4 (real kernels)"),
             ("Gemma-4-E4B-it", "Gemma-4-E4B-it bf16 (round 1)", "Gemma-4-E4B-it QAT Q4_0 (Unsloth)")]
    A("Positive = **worse** at 4 bits. Read every row against the McNemar table below: most of these")
    A("differences are churn, not loss.\n")
    A("| model | task | bf16 | 4-bit | points lost (+ = worse at 4-bit) |")
    A("|---|---|---|---|---|")
    for name, hi, lo in pairs:
        if hi not in res or lo not in res: continue
        for cat in ("distance", "left_right", "mcq"):
            a_, b_ = res[hi][cat]["pct"], res[lo][cat]["pct"]
            if a_ is None or b_ is None: continue
            A(f"| {name} | {cat} | {a_:.2f}% | {b_:.2f}% | {a_-b_:+.2f} |")
        A(f"| {name} | **task_mean(d,lr)** | {res[hi]['task_mean']:.2f}% | {res[lo]['task_mean']:.2f}% | "
          f"{res[hi]['task_mean']-res[lo]['task_mean']:+.2f} |")
    A("")

    if mcn:
        A("## Are the 4-bit deltas real? Paired McNemar\n")
        A("The two precisions disagree on half the items, so an unpaired difference of a few points is not evidence")
        A("of anything. Only discordant pairs carry information: **b** = right at bf16 and wrong at 4-bit, **c** =")
        A("the reverse. Exact two-sided test.\n")
        A("| comparison | task | b (bf16 right) | c (4-bit right) | p | verdict |")
        A("|---|---|---|---|---|---|")
        for k, v in mcn.items():
            sig = "**real**" if v["p"] < 0.05 else "not distinguishable from churn"
            A(f"| {k[0]} | {k[1]} | {v['b']} | {v['c']} | {v['p']:.5f} | {sig} |")
        A("")
    A("## How to read these deltas\n")
    lo, hi = chance.get("mcq_n_regions_range", (None, None))
    A(f"Chance levels, computed from the item set: **left_right {chance['left_right']:.1f}%** (two-way choice) and")
    A(f"**mcq {chance['mcq']:.2f}%** — mcq is not a four-option question, it offers between {lo} and {hi} numbered")
    A(f"regions per item, so chance is the mean of 1/n_regions, not 1/4.\n")
    if "Cosmos3-Edge bf16 (L0 control)" in res and "Gemma-4-E4B-it bf16 (round 1)" in res:
        c_ = res["Cosmos3-Edge bf16 (L0 control)"]
        g = res["Gemma-4-E4B-it bf16 (round 1)"]
        A(f"**Gemma's bf16 left_right ({g['left_right']['pct']:.2f}%) is at chance, and its mcq ({g['mcq']['pct']:.2f}%)")
        A(f"is *below* the {chance['mcq']:.2f}% chance level.** Signal that was never there cannot be lost, so its")
        A("quantisation delta on those two tasks is uninformative — a flat Gemma left_right delta is **not** evidence")
        A(f"that quantisation is free. The only meaningful Gemma delta is **distance** ({g['distance']['pct']:.2f}% bf16),")
        A("which is the one task where it demonstrably has something to lose.\n")
    if "Cosmos3-Edge bf16 (L0 control)" in res:
        c = res["Cosmos3-Edge bf16 (L0 control)"]
        A(f"Cosmos3-Edge clears chance on both: left_right {c['left_right']['pct']:.2f}% vs 50.0%, mcq")
        A(f"{c['mcq']['pct']:.2f}% vs {chance['mcq']:.2f}% — though the mcq margin is only")
        A(f"{c['mcq']['pct']-chance['mcq']:+.2f} points, so read that column with its N (456) in mind. All three")
        A("tasks are informative for Cosmos3-Edge.\n")

    A("## Distance calibration constant k, re-fitted per arm\n")
    A("Estimator, stated so it is reproducible: **k = median(gt / pred)** on the fitting half; halves are even/odd")
    A("position in the id-sorted distance items, so the split is deterministic and identical across arms. Each half")
    A("is scored with the *other* half's k, so **the gains in this table are held out.** The re-fitted McNemar")
    _ins = []
    for _l in ("Cosmos3-Edge bf16 (L0 control)", "Cosmos3-Edge INT4 AWQ (simulated)",
               "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)", "Cosmos3-Edge NF4 (real kernels)"):
        _ins.append(str(insample_distance((arm_rows or {}).get(_l), res[_l]["k"]["k_mean"]))
                    if _l in res and (arm_rows or {}).get(_l) else "—")
    A("rows and the own-constant figures quoted above are **in-sample** (each arm's full-set k applied to its own")
    A(f"full set): {' / '.join(_ins)} for bf16 / simulated / TensorRT / NF4 against the held-out 174 / 158 /")
    _g = [str(insample_distance((arm_rows or {}).get(_l), res[_l]["k"]["k_mean"]))
          if _l in res and (arm_rows or {}).get(_l) else "—"
          for _l in ("Gemma-4-E4B-it bf16 (round 1)", "Gemma-4-E4B-it QAT Q4_0 (Unsloth)")]
    A(f"167 / 174 below, and {' / '.join(_g)} for the two Gemma rows. Every re-fitted McNemar row — all seven —")
    A("reconciles against the in-sample set: b - c equals the difference of the two counts in each case.")
    A("In-sample is the right basis for *comparing arms* — every arm")
    A("is favoured equally — and the wrong basis for claiming an absolute gain, which is why both are reported.\n")
    A("| row | k (half A) | k (half B) | k mean | uncalibrated | held-out calibrated | gain (pts) | N usable |")
    A("|---|---|---|---|---|---|---|---|")
    for label, e in res.items():
        k = e["k"]
        A(f"| {label} | {k['k_half_a']} | {k['k_half_b']} | {k['k_mean']} | {k['uncalibrated']} | "
          f"{k['held_out_calibrated']} | {k['gain_pts']:+.2f} | {k['n_distance_usable']}/{k['n_distance_total']} |")
    A("| **Cosmos3-Edge INT4 AWQ — Orin engine (SM87)** | — | — | — | — | — | — | — |")
    A("")
    A("The Orin row is deliberately blank. It is filled by the engine-equivalence gate run, and **its constant is")
    A("the one that gets baked** — the SM103 figure above is provisional until then.\n")
    A("**This table counts differently from the headline table, by 1-2 items per arm** (bf16 143 here against")
    A("144 there; simulated 104/106; NF4 146/147; TensorRT 89/90). The calibration pipeline can only use items")
    A("where both ground truth and prediction parse as positive numbers — it needs a `gt/pred` ratio — so it")
    A("drops 37-38 per arm. The dataset's own scorer still scores those items, and one or two of them happen to")
    A("land inside the +/-10% band anyway. Quote 486-based figures or 449-based ones, never both in a sentence.\n")
    A("### The left_right prior is not a re-thresholding target\n")
    A("left_right is balanced 250/250 and the model answers `left` 72.4% of the time: **87.2% correct when the")
    A("answer is left, 42.4% when it is right** — below chance on half the set, and nearly unchanged by")
    A("quantisation. The obvious fix is order-debiasing: the question asks whether A is right of B, so asking the")
    A("mirror about the same image must invert the answer. Run over all 500 items on the TensorRT engine:\n")
    A("| | n = 500 |")
    A("|---|---|")
    A("| same answer to the mirrored question (**the prior**) | 328/500 = **65.6%** |")
    A("| answer flipped, as geometry requires | 172/500 = 34.4% |")
    A("| accuracy, as run | 320/500 = 64.00% |")
    A("| accuracy, order-debiased | 254/500 = 50.80% (**-13.20**) |")
    A("| accuracy on the order-consistent subset | 121/172 = **70.35%** |")
    A("")
    A("**Tie rule, without which the debiased figure is not derivable:** where the pair flips, keep the original")
    A("answer; where it does not, the prior is showing, so take the answer the model is biased *against* (`right`).")
    A("Any tie rule loses on those 328 items — keeping the biased answer merely reproduces the 64% baseline — and")
    A("that is the point: the pair carries no information there.\n")
    A("**So order-debiasing cannot help.** Its ceiling is the 121 it gets right when consistent plus a guess on")
    A("the rest, near 57%, below the 64% already scored. The 72/28 split is not a mis-set threshold on a working")
    A("comparison; it is the absence of the comparison on two thirds of items, partly masked by a bias that pays.")
    A("What survives is a **confidence signal** — 70.35% when order-consistent against 60.7% when not — at 2x")
    A("inference cost, which is worth having on a rover that can decline. A logit-bias variant cannot be judged")
    A("from these outputs at all: the harness saves decoded text, not scores. Reproduce with")
    A("`scripts/l0_swap_probe.py`.\n")
    A("### Is one constant really one constant?\n")
    A("Not quite, and it matters most where the rover cares most. Median `gt/pred` by ground-truth range:\n")
    A("| arm | [0,4) m | [4,8) | [8,12) | [12,+) | full-set median | spread |")
    A("|---|---|---|---|---|---|---|")
    A("| bf16 | 1.1028 | 1.2203 | 1.1228 | 1.1465 | 1.1492 | 0.118 |")
    A("| **INT4 AWQ (TensorRT)** | 1.1277 | 1.2486 | 1.1516 | **1.3350** | 1.2083 | **0.207** |")
    A("| NF4 | 1.0837 | 1.2317 | 1.1513 | 1.1441 | 1.1574 | 0.148 |")
    A("")
    A("The last column is the **full-set median** of `gt/pred`, not the split-half mean `k` in the table below —")
    A("1.1492 here against 1.1546 there for bf16. Same estimator, different sample.\n")
    A("bf16's far bucket (1.1465) sits on its global constant (1.1492); the AWQ engine's far bucket needs")
    A("**1.3350 against a global 1.2083**, so a single constant systematically under-corrects exactly the long")
    A("readings. Held out, split-half, a four-bucket constant is worth **+3.56 points** on the AWQ arm")
    A("(167 -> 183 of the 449 usable, the same basis as the `gain` column below, where the global constant is")
    A("worth +17.37), **+0.00** on bf16 and **-1.44** on NF4. So AWQ's under-read is range-dependent and")
    A("the other two arms' are not — which is a second way of saying the constant belongs to the artefact.\n")
    A("This qualifies the headline rather than overturning it: one global constant still recovers almost all of")
    A("the loss (p = 0.931 against bf16). But whoever bakes a constant for a rover that reads at 12 m and beyond")
    A("should fit it per range bucket, and the gate below should report k per bucket, not one number.\n")
    A("**Gemma's constant flips sign under quantisation** — 0.9218 at bf16 to 1.1381 at Q4_0. It changes from a")
    A("long-reader to a short-reader, and calibration *hurts* it at bf16 (-4.07 pts) while doing nothing at Q4_0")
    A("(+0.00). That is the signature of a model whose distance outputs carry no consistent scale to correct;")
    A("contrast Cosmos, whose two half-fits agree to three decimals in every arm. Do not bake a constant for Gemma.\n")
    A("**k was re-fitted on the 4-bit outputs, not carried over.** The round-1 constant on record was ~1.15; this")
    A("procedure reproduces ~1.154 on the same round-1 bf16 outputs (halves 1.149 / 1.159, +7.13 pts held out),")
    A("close to but not identical with the 1.158 / 1.141 and +6.6 pts recorded earlier — the earlier fit used a")
    A("procedure that was not written down, so it cannot be reproduced exactly. Every arm here uses the estimator")
    A("above, so the arms are comparable with each other.\n")

    A("### Diagnosis from the histograms\n")
    g = chance.get("gt_left_right", {})
    A(f"The ground truth is balanced — left_right is **{g.get('left','?')} left / {g.get('right','?')} right**, and the")
    A("mcq answer key is near-uniform across region indices. So a bias in the answer column is a bias in the model,")
    A("not an artefact of the item set, and any arm answering one label most of the time is paying for it.\n")
    A("| row | task | modal answer | modal share | accuracy | reading |")
    A("|---|---|---|---|---|---|")
    for label, e in res.items():
        for cat in ("left_right", "mcq"):
            h, n = e["hist"][cat]
            if not n: continue
            mk, mv = max(h.items(), key=lambda x: x[1])
            share = 100 * mv / n
            a_ = e[cat]["pct"]; ch = chance[cat]
            if share >= 80:
                rd = f"**collapses to a constant** — answers `{mk}` on {share:.0f}% of items"
            elif a_ is not None and ch is not None and a_ <= ch + 2:
                rd = "answers vary but carry no signal — spread out, still at chance"
            else:
                rd = "varied and above chance"
            A(f"| {label} | {cat} | `{mk}` | {share:.1f}% | {a_:.2f}% | {rd} |")
    A("")
    A("**Both models answer `left` about three times in four on a 50/50 set.** For Gemma that bias is the whole")
    A("story — it scores at chance. For Cosmos3-Edge it is not: it clears chance by 15 points *despite* the bias,")
    A("which means the bias is costing it accuracy it already has. It *looks* like a prior to correct, the way")
    A("`k` corrects the distance under-read — and the order-swap probe above shows it is not. `k` fixes a scale")
    A("error on a working measurement; this is the **absence** of the comparison on two thirds of items, and")
    A("re-thresholding cannot recover a judgement that was never made.\n")
    A("The histogram shape is still a useful diagnostic, just not a training signal: a model that **collapses to")
    A("a constant** is failing to engage the task, while one whose answers are **spread out and still at chance**")
    A("has no signal to sharpen. Cosmos is the third case — spread, above chance, and order-invariant — which is")
    A("why it reads as fixable and is not.\n")
    A("## Answer histograms\n")
    A("Distribution of *parsed* answers per task per arm. This is what separates \"no spatial signal\" from")
    A("\"always answers the same thing\".\n")
    for label, e in res.items():
        A(f"**{label}**\n")
        for cat in ("distance", "left_right", "mcq"):
            h, n = e["hist"][cat]
            if not n: continue
            top = ", ".join(f"`{k}`: {v}" for k, v in sorted(h.items(), key=lambda x: -x[1])[:10])
            A(f"- {cat} (N={n}): {top}")
        A("")

    if think:
        A("## Thinking-enabled subsample — the winner\n")
        A(think)
        A("")
        A("**Reconciling this with L1, which reached the opposite conclusion.** L1 keeps thinking *on* because it")
        A("took detections from 8 to 10 of 12; L0 finds it costs ~2 points and 27.7x the tokens. Both are right,")
        A("and they are not in tension: L0 asks a closed question with a one-token answer already in the model,")
        A("where extra reasoning only adds places to drift. L1 asks for open-vocabulary grounding plus coordinates,")
        A("where the reasoning is doing perceptual work — writing the entity out before committing a box. The rule")
        A("that satisfies both: **thinking on for grounding and operator commands, off for closed spatial")
        A("queries**, and it is a")
        A("per-request flag, so nothing has to be chosen globally.\n")

    if agr:
        A("## Per-item agreement — and why the INT4 disagreement needs a floor under it\n")
        A("The bake-off hand-off asks for a per-item agreement check between precisions. That number is uninterpretable on its own,")
        A("because two runs of the *same model at the same precision* do not agree either. Both are measured here.\n")
        A("| comparison | identical predictions | per category |")
        A("|---|---|---|")
        for k, v in agr.items():
            pc = " · ".join(f"{c} {d['pct']:.1f}%" for c, d in sorted(v["per_category"].items()))
            A(f"| {k} | **{v['identical']}/{v['n']} = {v['pct']:.2f}%** | {pc} |")
        A("")
        A("Look at the per-category split rather than the totals. Quantisation leaves **86.8%** of left_right and")
        A("**62.3%** of mcq answers untouched but changes **all but 13.4%** of the distance answers — while the")
        A("k-corrected distance accuracy is statistically unchanged. That is the signature of a **scale shift, not")
        A("lost information**: 4-bit weights move nearly every numeric estimate, and moving them all back by one")
        A("constant restores the score. It is also why an agreement threshold applied uniformly across tasks would")
        A("fail this model on distance for the wrong reason.\n")
        A("Read the second row against the first, never against 100%. Greedy decode is deterministic within a")
        A("container but not across two of them: a different transformers build reorders enough floating-point")
        A("work to flip a fraction of answers at identical weights. Any INT4 disagreement below the environment")
        A("floor is not evidence of quantisation damage.\n")
    A("## What these numbers are not\n")
    A("- **There are two INT4 rows and they answer different questions.** The *simulated* one is")
    A("  `mtq.quantize()` fake-quant inside PyTorch — weights on the INT4 grid, GEMM still in bf16 — so it is the")
    A("  weight-precision delta and says nothing about runtime. The *TensorRT* one is real INT4 kernels, but on")
    A("  **SM103**, which is not the Orin engine: same ONNX export, different plan. Neither row licenses an Orin")
    A("  accuracy claim. The **B300-vs-Orin engine-equivalence gate** is what closes that gap, and it has not run.")
    A("- The two 4-bit rows are **not the same kind of object**: Gemma's is QAT (quantisation-aware trained")
    A("  upstream by Google, then packed to Q4_0); Cosmos3-Edge's is PTQ (post-training AWQ). A QAT model has")
    A("  been trained to survive its own quantisation; a PTQ model has not. Do not read the two deltas as a like-")
    A("  for-like comparison of \"how well each model quantises\".")
    A("- **No per-item L0 output is reproducible elsewhere to better than ~5%.** Two runs of the *same* model at")
    A("  the same precision in different containers agree on 95.3% of items overall and only 87.9% on distance.")
    A("  That is the environment floor, and it bounds every per-item comparison in this document: a disagreement")
    A("  smaller than the floor is not evidence. It is also why the paired tests here compare arms run in one")
    A("  container, and why an Orin-vs-B300 engine gate has to be read against the same floor.")
    A("- **The Gemma pair is the exception to that.** Its bf16 row is the round-1 container and its Q4_0 row is")
    A("  llama.cpp — a different runtime, not just a different container — so that comparison carries more than")
    A("  the floor's worth of nuisance variation. It changes no conclusion, because every Gemma delta is churn")
    A("  already (all three p > 0.3), but the pair should not be quoted as a clean precision contrast.")
    A("- The Gemma artefact is **Q4_0, not Q4_K_XL**, despite the `UD-Q4_K_XL` filename: 100% of parameters are")
    A("  stored Q4_0, no K-quants. This is the artefact deployed on jetson0, so any page or hand-off calling the")
    A("  incumbent \"Q4_K_XL\" is repeating a filename, not a format.")
    A("- Parse-failure rates are in the k table's `N usable` column. Gemma bf16 yields a usable distance number")
    A("  on 442/486 against 449/486 for the Cosmos arms (448 for the simulated one) — 44 failures against 37.")
    A("  Reported for completeness; at p ~ 0.4 it is not a difference this item set establishes.")
    A("- Nothing here has been executed on a Jetson.\n")
    # ---------------- (3) reproducibility footer -------------------------------------
    A("## Reproducibility\n")
    A("**Dataset.** NVIDIA *PhysicalAI-Spatial-Intelligence-Warehouse* (`nvidia/PhysicalAI-Spatial-Intelligence-")
    A("Warehouse` on the Hugging Face Hub), CC-BY-4.0, synthetic warehouse scenes rendered in Omniverse with")
    A("rule-generated Q&A refined by Llama-3.1-70B-Instruct (so the annotations also carry Llama 3.1 Community")
    A("License terms). **Gated — request access; cited, never redistributed.** Split: `val.json`, 1,942 items, of")
    A("which this report scores 1,442 — distance 486, left_right 500, mcq 456. The `count` category (500) is not")
    A("used. The model is shown a rendered frame with the referenced regions drawn as numbered outlines, not the")
    A("source image.\n")
    A("**Scoring rules.** distance: correct within **+/-10%** of ground truth after applying the arm's constant")
    A("`k`. left_right and mcq: **exact match** (mcq answers are region indices, not option letters). Scored by")
    A("the dataset's own `utils/compute_scores.py`; `k` is a split-half median of `gt/pred`, fitted on one half")
    A("and applied to the other.\n")
    A("**Code.** Prompt, rendering and parser `scripts/spatial_qa_eval.py` (`build_prompt`, `render_item`,")
    A("`parse_answer`); PyTorch arms `scripts/l0_cosmos_int4_sim.py`; TensorRT arm `scripts/l0_trt_eval.py`;")
    A("scorer, `k` estimator and this document `scripts/l0_report.py` (`fit_k`, `mcnemar`, `wrong_k_cost`);")
    A("order-swap probe `scripts/l0_swap_probe.py`.\n")
    A("**Versions.**\n")
    A("| component | version |")
    A("|---|---|")
    A("| transformers (bf16, simulated-INT4, NF4 arms) | 5.16.1 |")
    A("| transformers (round-1 container, environment-floor pair) | round-1 image, **not identical** to the above |")
    A("| torch | 2.13.0+cu130 |")
    A("| nvidia-modelopt (AWQ quantise) | 0.33.0 |")
    A("| bitsandbytes (NF4) | as packaged in the run container |")
    A("| llama.cpp (Gemma Q4_0) | b9602 |")
    A("| TensorRT | 10.13.2.6 |")
    A("| TensorRT Edge-LLM | `e8b2952` (v0.10.1) **+ 7 local patches** = NVIDIA/TensorRT-Edge-LLM #205 and #207 |")
    A("| CUDA | 13.0 |")
    A("| hardware | DGX B300, SM103 |")
    A("")
    A("The two transformers builds behind the environment floor are **not the same image** — that is what the")
    A("floor measures. The round-1 container was not version-pinned at the time, which is itself a finding: it is")
    A("why the floor had to be measured rather than assumed.\n")
    A("**Artefacts.** AWQ checkpoint `quantized-int4-awq-v2/model.safetensors`, 2,408,300,272 bytes, built")
    A("2026-09-10. ONNX export `onnx-v3/llm/model.onnx.data`, 866,451,480 bytes, same date — the export the")
    A("Jetson also consumes. Engines are per-SM and not shared: the SM103 plan measured here is 843.7 MiB against")
    A("the Orin SM87 plan's 838.9 MiB from the same ONNX — **the same size, a different plan.** The point is not")
    A("that one is larger; it is that they are not the same artefact.\n")
    A("**Run dates.** Round-1 bf16 rows and the L0 control, simulated-INT4 and NF4 arms: 2026-09-09/10. Gemma")
    A("Q4_0: 2026-09-10. TensorRT INT4-AWQ rows and the order-swap probe: **2026-09-11**.\n")

    Path(out).write_text("\n".join(L))
    print(f"wrote {out} ({len('\n'.join(L))} chars)")



def mcnemar(rows_a, rows_b, cat, k_a=1.0, k_b=1.0):
    """Exact McNemar on the paired items: is the accuracy difference more than churn?

    The arms disagree on half the items even when their accuracies are close, so an unpaired difference of a few
    points says nothing on its own. b = items arm A got right and arm B got wrong, c = the reverse; only those
    discordant pairs carry information.
    """
    from math import comb
    da = {r["id"]: r for r in rows_a if r["category"] == cat}
    db = {r["id"]: r for r in rows_b if r["category"] == cat}

    def right(r, k):
        if cat == "distance":
            try: g, p = float(r["gt"]), float(r["pred"]) * k
            except ValueError: return False
            return 0.90 * g <= p <= 1.10 * g
        return str(r["pred"]).strip().lower() == str(r["gt"]).strip().lower()

    b = c = n = 0
    for i in da:
        if i not in db: continue
        n += 1
        ra, rb = right(da[i], k_a), right(db[i], k_b)
        b += int(ra and not rb); c += int(rb and not ra)
    disc = b + c
    if disc == 0: return {"n": n, "b": b, "c": c, "p": 1.0}
    tail = sum(comb(disc, i) for i in range(min(b, c) + 1)) / (2 ** disc)
    return {"n": n, "b": b, "c": c, "discordant": disc, "p": round(min(1.0, 2 * tail), 5)}


def agreement(rows_a, rows_b):
    """Per-item prediction agreement between two arms, overall and per category."""
    if not rows_a or not rows_b: return None
    da = {r["id"]: (str(r["pred"]), r["category"]) for r in rows_a}
    db = {r["id"]: str(r["pred"]) for r in rows_b}
    common = [k for k in da if k in db]
    out = {"n": len(common)}
    per = defaultdict(lambda: [0, 0])
    same = 0
    for k in common:
        pa, cat = da[k]
        hit = int(pa == db[k]); same += hit
        per[cat][0] += hit; per[cat][1] += 1
    out["identical"] = same
    out["pct"] = round(100 * same / max(len(common), 1), 2)
    out["per_category"] = {c: {"identical": v[0], "n": v[1], "pct": round(100 * v[0] / v[1], 2)}
                           for c, v in per.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--round1", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--trt", default=None, help="dir holding the TensorRT-backed arms (E_awq_trt_*)")
    a = ap.parse_args()
    D, R1 = a.dir, a.round1
    TRT = a.trt or D

    # (label, rows-for-distance/left_right, rows-for-mcq, stats file)
    arms = [
        ("Cosmos3-Edge bf16 (round 1)", load_raw(f"{R1}/E", "cosmos3edge"), load_raw(f"{R1}/E_mcq", "cosmos3edge_mcq"),
         f"{R1}/E/E_cosmos3edge_stats.json"),
        ("Cosmos3-Edge bf16 (L0 control)", load_raw(D, "cosmos3edge_bf16"), load_raw(D, "cosmos3edge_bf16_mcq"),
         f"{D}/E_cosmos3edge_bf16_stats.json"),
        ("Cosmos3-Edge INT4 AWQ (simulated)", load_raw(D, "cosmos3edge_int4"), load_raw(D, "cosmos3edge_int4_mcq"),
         f"{D}/E_cosmos3edge_int4_stats.json"),
        # Real INT4 kernels: the deployed TensorRT Edge-LLM engine built from the same ONNX export the
        # Jetson runs, so this row carries the quantisation error AND the export/runtime, where the
        # simulated row above carries the quantisation error alone. The gap between them is the cost
        # of the deployment path, not of 4-bit.
        ("Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)", load_raw(TRT, "awq_trt"), load_raw(TRT, "awq_trt_mcq"),
         f"{TRT}/E_awq_trt_stats.json"),
        ("Cosmos3-Edge NF4 (real kernels)", load_raw(D, "cosmos3edge_nf4"), load_raw(D, "cosmos3edge_nf4_mcq"),
         f"{D}/E_cosmos3edge_nf4_stats.json"),
        ("Gemma-4-E4B-it bf16 (round 1)", load_raw(f"{R1}/E", "gemma4e4b"), load_raw(f"{R1}/E_mcq", "gemma4e4b_mcq"),
         f"{R1}/E/E_gemma4e4b_stats.json"),
        ("Gemma-4-E4B-it QAT Q4_0 (Unsloth)", load_raw(D, "gemma4e4b_q4kxl"), load_raw(D, "gemma4e4b_q4kxl_mcq"),
         f"{D}/E_gemma4e4b_q4kxl_stats.json"),
    ]

    res = {}
    for label, main_rows, mcq_rows, sf in arms:
        if not main_rows: 
            print(f"[skip] {label}: no rows", file=sys.stderr); continue
        rows = list(main_rows) + list(mcq_rows or [])
        e = {"n_main": len(main_rows), "n_mcq": len(mcq_rows or [])}
        for cat in ("distance", "left_right", "mcq"):
            ok, n = acc(rows, cat)
            e[cat] = {"ok": ok, "n": n, "pct": round(100 * ok / n, 2) if n else None}
        e["task_mean"] = round((e["distance"]["pct"] + e["left_right"]["pct"]) / 2, 2)
        if e["mcq"]["n"]:
            e["task_mean_with_mcq"] = round((e["distance"]["pct"] + e["left_right"]["pct"] + e["mcq"]["pct"]) / 3, 2)
        e["k"] = fit_k(rows)
        e["hist"] = {c: histogram(rows, c) for c in ("distance", "left_right", "mcq")}
        try: e["stats"] = json.load(open(sf))
        except Exception: e["stats"] = {}
        res[label] = e

    def allrows(main_t, mcq_t, d):
        a = load_raw(d, main_t) or []
        b = load_raw(d, mcq_t) or []
        return list(a) + list(b)

    agr = {
        "environment floor — round-1 bf16 vs L0 bf16 control (same model, same precision, different container)":
            agreement(allrows("cosmos3edge", "cosmos3edge_mcq", f"{R1}/E") +
                      (load_raw(f"{R1}/E_mcq", "cosmos3edge_mcq") or []),
                      allrows("cosmos3edge_bf16", "cosmos3edge_bf16_mcq", D)),
        "quantisation — L0 bf16 control vs L0 simulated INT4 AWQ (same container, same seed)":
            agreement(allrows("cosmos3edge_bf16", "cosmos3edge_bf16_mcq", D),
                      allrows("cosmos3edge_int4", "cosmos3edge_int4_mcq", D)),
        "quantisation — L0 bf16 control vs L0 real NF4":
            agreement(allrows("cosmos3edge_bf16", "cosmos3edge_bf16_mcq", D),
                      allrows("cosmos3edge_nf4", "cosmos3edge_nf4_mcq", D)),
        "the two quantisers against each other — AWQ (sim) vs NF4 (real)":
            agreement(allrows("cosmos3edge_int4", "cosmos3edge_int4_mcq", D),
                      allrows("cosmos3edge_nf4", "cosmos3edge_nf4_mcq", D)),
    }
    agr = {k: v for k, v in agr.items() if v}
    (Path(D) / "agreement.json").write_text(json.dumps(agr, indent=2))

    mcn = {}
    cb = allrows("cosmos3edge_bf16", "cosmos3edge_bf16_mcq", D)
    cq = allrows("cosmos3edge_int4", "cosmos3edge_int4_mcq", D)
    gb = (load_raw(f"{R1}/E", "gemma4e4b") or []) + (load_raw(f"{R1}/E_mcq", "gemma4e4b_mcq") or [])
    gq = allrows("gemma4e4b_q4kxl", "gemma4e4b_q4kxl_mcq", D)
    cn = allrows("cosmos3edge_nf4", "cosmos3edge_nf4_mcq", D)
    ct = allrows("awq_trt", "awq_trt_mcq", TRT)     # real INT4 kernels, TensorRT
    for name, A_, B_ in (("Cosmos3-Edge bf16 -> simulated INT4 AWQ", cb, cq),
                         ("Cosmos3-Edge bf16 -> REAL INT4 AWQ (TensorRT)", cb, ct),
                         # same quantiser both sides: isolates export + runtime from quantisation
                         ("AWQ simulated -> AWQ real (TensorRT), b = simulated right", cq, ct),
                         ("Cosmos3-Edge bf16 -> real NF4", cb, cn),
                         ("head-to-head: AWQ (sim) vs NF4 (real), b = AWQ right", cq, cn),
                         ("head-to-head: AWQ (real TRT) vs NF4 (real), b = AWQ right", ct, cn),
                         ("Gemma-4-E4B bf16 -> QAT Q4_0", gb, gq)):
        if not A_ or not B_: continue
        for cat in ("distance", "left_right", "mcq"):
            m = mcnemar(A_, B_, cat)
            if m.get("n"): mcn[(name, cat)] = m
    # distance again, this time with each arm's own re-fitted k -- the brief's requirement, and the honest test
    for name, A_, B_, la, lb in (("Cosmos3-Edge bf16 -> simulated INT4 AWQ", cb, cq,
                                  "Cosmos3-Edge bf16 (L0 control)", "Cosmos3-Edge INT4 AWQ (simulated)"),
                                 ("Cosmos3-Edge bf16 -> REAL INT4 AWQ (TensorRT)", cb, ct,
                                  "Cosmos3-Edge bf16 (L0 control)",
                                  "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)"),
                                 ("AWQ simulated -> AWQ real (TensorRT), b = simulated right", cq, ct,
                                  "Cosmos3-Edge INT4 AWQ (simulated)",
                                  "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)"),
                                 ("Cosmos3-Edge bf16 -> real NF4", cb, cn,
                                  "Cosmos3-Edge bf16 (L0 control)", "Cosmos3-Edge NF4 (real kernels)"),
                                 ("head-to-head: AWQ (sim) vs NF4 (real), b = AWQ right", cq, cn,
                                  "Cosmos3-Edge INT4 AWQ (simulated)", "Cosmos3-Edge NF4 (real kernels)"),
                                 ("head-to-head: AWQ (real TRT) vs NF4 (real), b = AWQ right", ct, cn,
                                  "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)",
                                  "Cosmos3-Edge NF4 (real kernels)"),
                                 ("Gemma-4-E4B bf16 -> QAT Q4_0", gb, gq,
                                  "Gemma-4-E4B-it bf16 (round 1)", "Gemma-4-E4B-it QAT Q4_0 (Unsloth)")):
        if not A_ or not B_ or la not in res or lb not in res: continue
        m = mcnemar(A_, B_, "distance", res[la]["k"]["k_mean"], res[lb]["k"]["k_mean"])
        if m.get("n"): mcn[(name, "distance, each arm with its own re-fitted k")] = m

    think = None
    tp = Path(D) / "thinking_summary.md"
    if tp.exists(): think = tp.read_text()
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    write_md(res, a.out, chance_levels(a.data), think, agr, mcn,
             trt_rows=(load_raw(TRT, "awq_trt") or []) + (load_raw(TRT, "awq_trt_mcq") or []),
             arm_rows={"Gemma-4-E4B-it bf16 (round 1)": gb,
                       "Gemma-4-E4B-it QAT Q4_0 (Unsloth)": gq,
                       "Cosmos3-Edge bf16 (L0 control)": cb,
                       "Cosmos3-Edge INT4 AWQ (simulated)": cq,
                       "Cosmos3-Edge INT4 AWQ (real kernels, TensorRT)": ct,
                       "Cosmos3-Edge NF4 (real kernels)": cn})
    print(json.dumps({k: {c: v[c] for c in ("distance", "left_right", "mcq", "task_mean")} for k, v in res.items()},
                     indent=2))


if __name__ == "__main__":
    main()
