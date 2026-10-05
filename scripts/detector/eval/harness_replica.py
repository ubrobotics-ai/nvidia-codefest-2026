"""Replica of the bench harness (ubr-isaac baseline/report.py) on the held-out hospital split.
Frame-level decision; hard negatives = frames holding a non-lying person; the rule
+r means a box only fires if w/h >= 1.3. Threshold: lowest conf (0.005 steps from 0.99)
whose hard-negative FA <= target."""
import json, os, collections
PKG="/storage/hackathon_teams/omc-team15/codefest/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd"
R="/storage/hackathon_teams/omc-team15/codefest/ubr-det"
LY={"person_lying_vest","person_lying"}
val={os.path.basename(i["file_name"]) for i in json.load(open(f"{R}/data/coco_rebal/val.json"))["images"]}
kind={}
for sp in ("train","validation"):
    d=json.load(open(f"{PKG}/annotations/coco_{sp}.json"))
    cats={c["id"]:c["name"] for c in d["categories"]}
    names=collections.defaultdict(list)
    for a in d["annotations"]: names[a["image_id"]].append(cats[a["category_id"]])
    for im in d["images"]:
        b=os.path.basename(im["file_name"])
        if b not in val: continue
        n=[x for x in names.get(im["id"],[]) if x.startswith("person_")]
        kind[b]="pos" if set(n)&LY else ("hard" if n else "empty")
cnt=collections.Counter(kind.values())
print(f"hospital val: {len(kind)} frames = {cnt['pos']} positive + {cnt['hard']} hard + {cnt['empty']} empty")
import sys
# usage: harness_replica6.py "name|path|cls" ...   (path relative to ubr-det/preds or absolute)
ARMS=[tuple(x.split("|")[:2])+(int(x.split("|")[2]),) for x in sys.argv[1:]]
T=[0.002,0.005,0.010,0.020]
for rule in (1.3, 1.0, 0.0):
    print(f"\n--- rule w/h >= {rule}  (harness default is 1.3) ---")
    print(f"{'arm':<38}"+"".join(f"  FA<={t:.3f}     " for t in T))
    for name,fn,cls in ARMS:
        best={}
        for l in open(fn if fn.startswith("/") else f"{R}/preds/{fn}"):
            r=json.loads(l); m=0.0
            for x in r["boxes"]:
                if x["cls"]!=cls or x["bbox"][3]<=0: continue
                if x["bbox"][2]/x["bbox"][3] < rule: continue
                if x["conf"]>m: m=x["conf"]
            best[r["file_name"]]=m
        P=[best.get(b,0) for b,k in kind.items() if k=="pos"]
        H=[best.get(b,0) for b,k in kind.items() if k=="hard"]
        cells=[]
        for t in T:
            c=0.99; pick=None
            while c>=0.005:
                if sum(v>=c for v in H)/len(H)<=t: pick=c
                c=round(c-0.005,3)
            rec=sum(v>=pick for v in P)/len(P) if pick is not None else 0.0
            cells.append(f"{rec:.3f}@{pick if pick is not None else float('nan'):.3f}")
        print(f"{name:<38}"+"".join(f"  {c:<15}" for c in cells), flush=True)
