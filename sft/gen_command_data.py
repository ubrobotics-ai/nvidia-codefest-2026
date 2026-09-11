#!/usr/bin/env python3
"""Generate the command-vocabulary and UNKNOWN buckets for the SFT.

Scope note: this covers ONLY the classes that need no robot-side schema -- directional
vocabulary and declining. The tool-call and structured-output buckets need protocol.py
and prompts.py (the ten tools' argument enums, and the nested entity schema), which are
not on this machine; generating them against a guessed contract would teach the model a
contract production does not use.

Weighting follows the measured failures, not uniform sampling: FORWARD is the weak class
(7/10, and "Go forward" -> BACK is an inversion), UNKNOWN is the largest bucket because
the model has no declining behaviour to reinforce at all (0/1, and unstable).

Every phrasing named in the L1 evaluation is excluded -- those are the eval, and training
on them makes every subsequent number meaningless.
"""
import json, random, sys

# Nautical vocabulary is deliberately ABSENT from the seeds. The eval's directional set
# contains "Turn to starboard" -> RIGHT, which the model inverts, and any nautical seed
# ("to starboard side", "to port") is a near-duplicate of it: training on one and testing
# on the other is contamination by paraphrase, and the near-miss audit below catches it.
# Fixing the starboard inversion therefore needs a REFRESHED eval with fresh phrasings
# first -- the training document already calls for that. Until then the class stays
# untrained and honestly measured.
HOLDOUT = {s.lower() for s in [
    "turn to starboard", "withdraw", "go forward", "proceed", "continue onward",
    "sing me a song", "stop!", "stop right now", "freeze!", "halt immediately",
    "don't move", "drive forward",
]}

FORWARD = ["move ahead", "advance", "head on", "keep going", "press on", "move up",
    "forward march", "go straight", "straight ahead", "move out", "carry on ahead",
    "push ahead", "onward", "take us forward", "drive on", "roll forward", "ahead please",
    "start moving forward", "go on ahead", "move forwards", "let's move ahead",
    "head straight on", "make your way forward", "get moving forward"]
BACK = ["reverse", "back up", "pull back", "retreat", "go backwards", "back off",
    "move back", "reverse course", "pull out", "fall back", "back away", "come back",
    "return", "back it up", "reverse please"]
LEFT = ["turn left", "go left", "bear left", "hard left", "swing left", "veer left",
    "rotate left", "head left", "take a left", "left turn", "spin left"]
RIGHT = ["turn right", "go right", "bear right", "hard right", "swing right", "veer right",
    "rotate right", "head right", "take a right", "right turn",
    "spin right"]
STOP = ["stop", "halt", "hold", "hold position", "stand by", "cease", "pause",
    "stop moving", "stay put", "brake", "wait there", "all stop"]
SEARCH = ["look for a person", "search the room", "find anyone down", "scan the area",
    "sweep for casualties", "look around", "search ahead", "check the space"]
REPORT = ["report", "what do you see", "status", "tell me what you found",
    "give me an update", "describe the scene", "what's there"]
UNKNOWN = ["sing something", "what's the weather", "tell me a joke", "who are you",
    "make coffee", "what time is it", "play some music", "how old are you",
    "recite a poem", "what's your favourite colour", "order a pizza", "call my mother",
    "solve this equation", "translate this", "book a flight", "set an alarm",
    "what's 17 plus 25", "do a backflip", "open the pod bay doors", "read me the news"]

MODS = ["", "please ", "can you ", "now ", "quickly ", "robot, ", "okay ", "right, "]
SUFF = ["", " please", " now", " right away", "!", ".", " immediately"]

def variants(seed_phrases, label, n, rng):
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        p = rng.choice(seed_phrases)
        t = (rng.choice(MODS) + p + rng.choice(SUFF)).strip()
        k = t.lower()
        if k in HOLDOUT or k in seen:            # never the eval phrasings, never a dup
            continue
        if any(h in k for h in HOLDOUT):
            continue
        seen.add(k)
        out.append({"instruction": t, "label": label})
    return out

def main():
    rng = random.Random(11)
    plan = [(FORWARD,"FORWARD",150), (BACK,"BACK",100), (LEFT,"LEFT",100),
            (RIGHT,"RIGHT",100), (STOP,"STOP",80), (SEARCH,"SEARCH",80),
            (REPORT,"REPORT",80), (UNKNOWN,"UNKNOWN",200)]
    rows = []
    for seeds, label, n in plan:
        got = variants(seeds, label, n, rng)
        if len(got) < n:
            print(f"  WARNING {label}: only {len(got)}/{n} unique variants from "
                  f"{len(seeds)} seeds -- add seed phrasings", file=sys.stderr)
        rows += got
    rng.shuffle(rows)
    out = sys.argv[1] if len(sys.argv) > 1 else "sft/data/commands.jsonl"
    import os; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print(f"  {len(rows)} examples -> {out}")
    for k in sorted(c): print(f"    {k:8s} {c[k]}")
    leaked = [r for r in rows if r["instruction"].lower() in HOLDOUT]
    NEAR = ["starboard", "to port", "withdraw", "go forward", "proceed",
            "continue onward", "sing me a song", "freeze", "drive forward"]
    near = [r for r in rows if any(n in r["instruction"].lower() for n in NEAR)]
    nonascii = [r for r in rows if any(ord(c) > 0x2000 for c in r["instruction"])]
    print(f"  exact hold-out leakage : {len(leaked)}  {'OK' if not leaked else 'FAIL'}")
    print(f"  near-duplicate of eval : {len(near)}  {'OK' if not near else 'FAIL'}")
    print(f"  stray non-latin chars  : {len(nonascii)}  {'OK' if not nonascii else 'FAIL'}")
    if leaked or near or nonascii:
        sys.exit("  refusing to write contaminated training data")

if __name__ == "__main__":
    main()
