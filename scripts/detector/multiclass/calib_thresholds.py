"""Per-class OWLv2 thresholds from calib.jsonl (raw boxes, score >= 0.05, COCO valid images).
Ground truth: COCO labels (exhaustive for the 6 COCO classes); for box, only the LVIS-val images
that carry box labels, so box precision is measured where boxes are known to exist (optimistic).
Chosen threshold: the lowest with precision >= 0.80 (greedy IoU >= 0.5 matching)."""
import json, collections
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
d = json.load(open(f"{M}/data/valid/_annotations.coco.json")); cn = {c["id"]: c["name"] for c in d["categories"]}
idof = {i["file_name"]: i["id"] for i in d["images"]}
gt = collections.defaultdict(list)
for a in d["annotations"]: gt[(a["image_id"], cn[a["category_id"]])].append(a["bbox"])
rows = [json.loads(l) for l in open(f"{M}/prelabel/calib.jsonl")]
box_imgs = {iid for (iid, c) in gt if c == "box"}
def iou(a, b):
    iw = max(0., min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0., min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw*ih; u = a[2]*a[3] + b[2]*b[3] - i; return i/u if u > 0 else 0.
out = {}
for c in ("backpack", "handbag", "suitcase", "dog", "car", "truck", "box"):
    imgs = [r for r in rows if c != "box" or idof[r["file_name"]] in box_imgs]
    ngt = sum(len(gt[(idof[r["file_name"]], c)]) for r in imgs)
    line = []; pick = None
    for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        tp = fp = 0
        for r in imgs:
            g = list(gt[(idof[r["file_name"]], c)]); used = set()
            for cc, s, b in sorted([x for x in r["added"] if x[0] == c and x[1] >= t], key=lambda x: -x[1]):
                j = max(range(len(g)), key=lambda k: iou(b, g[k]) if k not in used else -1, default=None)
                if j is not None and j not in used and iou(b, g[j]) >= 0.5: tp += 1; used.add(j)
                else: fp += 1
        p = tp / max(tp + fp, 1); rc = tp / max(ngt, 1)
        line.append(f"{t:.2f}:P{p:.2f}/R{rc:.2f}")
        if pick is None and p >= 0.80: pick = (t, p, rc)
    out[c] = pick
    print(f"{c:<9} gt={ngt:<5} " + " ".join(line))
print("chosen (thr, precision, recall):", out)
json.dump({c: (v[0] if v else None) for c, v in out.items()}, open(f"{M}/prelabel/thresholds.json", "w"))
