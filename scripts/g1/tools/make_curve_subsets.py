"""Nested demo subsets for the v5 data-efficiency curve (lw-lab1 curves/PROTOCOL.md): train-split episodes only,
stratified equally over the four recording seeds (8001-8004), one fixed shuffle (seed 1234) per recording seed, so
N=64 (16 per seed) c N=256 (64 per seed) c N=all. Writes curve_v5/episodes_{64,256,all}.json and a summary.
Usage: make_curve_subsets.py DATA_ROOT OUT_DIR"""
import json, random, sys, collections
root, out = sys.argv[1], sys.argv[2]
prov = [json.loads(l) for l in open(f"{root}/meta/ubr_provenance.jsonl")]
lo, hi = (int(x) for x in json.load(open(f"{root}/meta/info.json"))["splits"]["train"].split(":"))
by_seed = collections.defaultdict(list)
for p in prov:
    e = int(p["episode_index"])
    if p.get("split", "train") == "train" and lo <= e < hi:
        by_seed[str(p["seed"])].append(e)
assert sorted(by_seed) == ["8001", "8002", "8003", "8004"], sorted(by_seed)
order = {}
for s in sorted(by_seed):
    eps = sorted(by_seed[s]); random.Random(1234).shuffle(eps); order[s] = eps
subsets = {"64": sum((order[s][:16] for s in sorted(order)), []),
           "256": sum((order[s][:64] for s in sorted(order)), []),
           "all": sum((order[s] for s in sorted(order)), [])}
assert set(subsets["64"]) <= set(subsets["256"]) <= set(subsets["all"])
for k, v in subsets.items():
    json.dump(sorted(v), open(f"{out}/episodes_{k}.json", "w"))
summary = {k: {"n": len(v), "per_seed": {s: sum(e in set(v) for e in by_seed[s]) for s in sorted(by_seed)}} for k, v in subsets.items()}
summary["train_range"] = f"{lo}:{hi}"
json.dump(summary, open(f"{out}/subsets_summary.json", "w"), indent=1)
print(json.dumps(summary))
