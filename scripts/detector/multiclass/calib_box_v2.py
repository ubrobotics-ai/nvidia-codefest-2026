"""Box threshold for pseudo-labelling: v2's precision/recall on the unseen warehouse renders (greedy IoU >= 0.5)."""
import json, glob, collections
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
d = json.load(open(f"{M}/data_v2/valid/_annotations.coco.json"))
box = [c["id"] for c in d["categories"] if c["name"] == "box"][0]
w = {i["id"] for i in d["images"] if i["file_name"].startswith("wbox_")}
gt = collections.defaultdict(list)
for a in d["annotations"]:
    if a["category_id"] == box and a["image_id"] in w: gt[a["image_id"]].append(a["bbox"])
pr = collections.defaultdict(list)
for p in glob.glob(f"{M}/preds/mc_v2.s*.coco.json"):
    for r in json.load(open(p)):
        if r["category_id"] == box and r["image_id"] in w and r["score"] >= 0.2: pr[r["image_id"]].append(r)
def iou(a, b):
    iw = max(0., min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0., min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw*ih; u = a[2]*a[3] + b[2]*b[3] - i; return i/u if u > 0 else 0.
ngt = sum(len(v) for v in gt.values())
for t in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
    tp = fp = 0
    for i in w:
        g = gt[i]; used = set()
        for r in sorted([x for x in pr[i] if x["score"] >= t], key=lambda x: -x["score"]):
            best, bj = 0, None
            for j, gb in enumerate(g):
                if j not in used:
                    v = iou(r["bbox"], gb)
                    if v > best: best, bj = v, j
            if bj is not None and best >= 0.5: tp += 1; used.add(bj)
            else: fp += 1
    print(f"thr {t}: precision {tp/max(tp+fp,1):.3f} recall {tp/ngt:.3f} boxes kept {tp+fp}", flush=True)
