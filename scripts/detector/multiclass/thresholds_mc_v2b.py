"""(v2b copy: data_v2 valid, preds/mc_v2b) Per-class operating thresholds for the 10-class model: lowest score (0.05 steps) with precision >= 0.80
at IoU >= 0.5 on the COCO valid images (box: on the images with LVIS box labels). Greedy matching."""
import json, glob, collections
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
d = json.load(open(f"{M}/data_v2/valid/_annotations.coco.json"))
cn = {c["id"]: c["name"] for c in d["categories"]}
imname = {i["id"]: i["file_name"] for i in d["images"]}
gt = collections.defaultdict(list)
for a in d["annotations"]: gt[(a["image_id"], a["category_id"])].append(a["bbox"])
res = []
for p in glob.glob(f"{M}/preds/mc_v2b.s*.coco.json"): res += json.load(open(p))
coco = {i for i, n in imname.items() if n.startswith("coco_")}
boxc = [k for k, v in cn.items() if v == "box"][0]
boximgs = {i for (i, c) in gt if c == boxc}
def iou(a, b):
    iw = max(0., min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0., min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw*ih; u = a[2]*a[3] + b[2]*b[3] - i; return i/u if u > 0 else 0.
out = {}
for cid, name in cn.items():
    if name == "person_lying": continue
    imgs = boximgs if name == "box" else coco
    pr = collections.defaultdict(list)
    for r in res:
        if r["category_id"] == cid and r["image_id"] in imgs and r["score"] >= 0.1: pr[r["image_id"]].append(r)
    ngt = sum(len(gt[(i, cid)]) for i in imgs)
    pick = None; line = []
    for t in [round(0.1 + 0.05 * k, 2) for k in range(17)]:
        tp = fp = 0
        for i in imgs:
            g = gt[(i, cid)]; used = set()
            for r in sorted([x for x in pr[i] if x["score"] >= t], key=lambda x: -x["score"]):
                best, bj = 0, None
                for j, gb in enumerate(g):
                    if j not in used:
                        v = iou(r["bbox"], gb)
                        if v > best: best, bj = v, j
                if bj is not None and best >= 0.5: tp += 1; used.add(bj)
                else: fp += 1
        p, rc = tp / max(tp + fp, 1), tp / max(ngt, 1)
        line.append(f"{t}:P{p:.2f}/R{rc:.2f}")
        if pick is None and p >= 0.80: pick = (t, round(p, 3), round(rc, 3))
    out[name] = pick
    print(f"{name:<9}", " ".join(line[::2]), flush=True)
print("CHOSEN", json.dumps(out))
json.dump(out, open(f"{M}/export/thresholds_mc_v2b.json", "w"), indent=1)
