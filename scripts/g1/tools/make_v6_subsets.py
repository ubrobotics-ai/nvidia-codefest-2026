"""v6 subsets (Filipe 2026-10-08, option A): N=64 = 48 warehouse + 16 plain (the full set's ~3:1 mix), 12 + 4 per support
height, one fixed shuffle (seed 1234) per (scene, height); N=all = every train episode. Train split only.
Writes curve_v6/episodes_{64,all}.json and subsets_summary.json. Usage: make_v6_subsets.py DATA_ROOT OUT_DIR"""
import json, random, sys, collections
root, out = sys.argv[1], sys.argv[2]
lo, hi = (int(x) for x in json.load(open(f"{root}/meta/info.json"))["splits"]["train"].split(":"))
groups = collections.defaultdict(list)
for p in map(json.loads, open(f"{root}/meta/ubr_provenance.jsonl")):
    e = int(p["episode_index"])
    if p.get("split", "train") == "train" and lo <= e < hi:
        groups[(p["scene"], p["support_height_m"])].append(e)
take = {"warehouse": 12, "plain": 4}
sub = []
for (scene, h), eps in sorted(groups.items()):
    eps = sorted(eps); random.Random(1234).shuffle(eps); sub += eps[:take[scene]]
    assert len(eps) >= take[scene], (scene, h, len(eps))
alle = sorted(e for v in groups.values() for e in v)
json.dump(sorted(sub), open(f"{out}/episodes_64.json", "w")); json.dump(alle, open(f"{out}/episodes_all.json", "w"))
s = {"64": len(sub), "all": len(alle), "groups_train": {f"{k[0]} {k[1]}": len(v) for k, v in sorted(groups.items())}}
json.dump(s, open(f"{out}/subsets_summary.json", "w"), indent=1); print(json.dumps(s))
