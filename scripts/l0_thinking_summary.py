#!/usr/bin/env python3
"""Compare a thinking-enabled subsample against the same items with thinking off.

The bake-off ran `enable_thinking=False` throughout, which is the right call for cost but understates a model sold
as a reasoner -- and that asymmetry is a fairness problem in a comparison, not just a missing number. This scores
~200 items both ways on the winner and reports what reasoning costs in tok/s and buys in accuracy.

Items are matched by id, so the comparison is paired: the thinking-off numbers are recomputed on exactly the
subsample, never taken from the full-set headline.

  python scripts/l0_thinking_summary.py --dir <L0 dir> --think-tag cosmos3edge_bf16_think --base-tag cosmos3edge_bf16
"""
import argparse, json, sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l0_report import load_raw, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--think-tag", required=True)
    ap.add_argument("--base-tag", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = Path(a.dir)
    think = load_raw(D, a.think_tag)
    base = load_raw(D, a.base_tag)
    if not think or not base:
        print("missing rows", file=sys.stderr); return 1
    ids = {r["id"] for r in think}
    base = [r for r in base if r["id"] in ids]
    assert len(base) == len(think), f"paired subsample mismatch: {len(base)} vs {len(think)}"

    ts = json.load(open(D / f"E_{a.think_tag}_stats.json"))
    bs = json.load(open(D / f"E_{a.base_tag}_stats.json"))

    L = ["Run on the bake-off winner, **Cosmos3-Edge at bf16**, over the first "
         f"{len(think)} items of the distance + left_right set (paired by id: the thinking-off column is "
         "recomputed on exactly these items, not carried from the full-set headline).\n",
         "| task | thinking off | thinking on | delta |", "|---|---|---|---|"]
    for cat in ("distance", "left_right"):
        ok_b, n_b = acc(base, cat)
        ok_t, n_t = acc(think, cat)
        if not n_b: continue
        pb, pt = 100 * ok_b / n_b, 100 * ok_t / n_t
        L.append(f"| {cat} | {ok_b}/{n_b} = {pb:.2f}% | {ok_t}/{n_t} = {pt:.2f}% | {pt-pb:+.2f} |")
    okb = sum(acc(base, c)[0] for c in ("distance", "left_right"))
    okt = sum(acc(think, c)[0] for c in ("distance", "left_right"))
    n = len(think)
    L.append(f"| **both, pooled** | {okb}/{n} = {100*okb/n:.2f}% | {okt}/{n} = {100*okt/n:.2f}% | "
             f"{100*(okt-okb)/n:+.2f} |")
    L.append("")
    same_n = bs.get("n") == ts.get("n")
    L.append(f"**Cost.** Over the same {ts['n']} items: {bs['tokens_generated']} generated tokens in "
             f"{bs['generate_seconds']}s at {bs['tokens_per_s']} tok/s with thinking off, versus "
             f"{ts['tokens_generated']} tokens in {ts['generate_seconds']}s at {ts['tokens_per_s']} tok/s with it "
             f"on — **{ts['tokens_generated']/max(bs['tokens_generated'],1):.1f}x the tokens and "
             f"{ts['generate_seconds']/max(bs['generate_seconds'],1e-9):.1f}x the wall time** on a B300."
             + ("" if same_n else f"  **WARNING: item counts differ ({bs.get('n')} vs {ts.get('n')}); the "
                                  f"timing ratio is not comparable.**"))
    L.append("")
    L.append(f"**Token budget.** The thinking arm was given `max_new_tokens={ts['max_new_tokens']}` against "
             f"{bs['max_new_tokens']} for the rest of the bake-off. Raising it is not a thumb on the scale: at 192 a "
             "reasoner spends the budget on reasoning and never emits its `ANSWER:` line, and the parser then falls "
             "back to the last number or direction word in the reasoning text — which scores the model's thinking "
             "aloud, not its answer. Generations that still used the whole budget: "
             f"{ts.get('truncated_at_budget','?')}/{ts['n']} ({ts.get('truncated_pct','?')}%) with thinking on, "
             f"{bs.get('truncated_at_budget','?')}/{bs['n']} ({bs.get('truncated_pct','?')}%) with it off.")
    L.append("")
    L.append("B300 figures. An Orin would pay the same token multiple against a much smaller budget.")
    out = a.out or (D / "thinking_summary.md")
    Path(out).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main() or 0)
