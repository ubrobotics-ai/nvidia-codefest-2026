"""Cosmos Transfer batch 2: per-frame label check, then stage the frames that pass.

A translated frame keeps its Isaac label only if a COCO person detector (torchvision Faster R-CNN,
score > 0.5) finds a person at IoU >= 0.5 with the source coco.json box. This is the same rule as
perframe_iou.py for batch 1, applied per frame to every clip.

Frames whose subject box is under 1300 px^2 cannot be checked: below that size the COCO detector
finds almost nothing even on clean renders (handoff 2d). They are reported as "too_small" and NOT
staged. This biases the staged set toward near and mid range; the far band is under-represented.

Labels (YOLO): class 0 = lying casualty (category person_lying_vest), from the coco box. Frames of
the warehouse_poses clips (category person_standing, i.e. standing or kneeling) are staged with an
EMPTY label file: photoreal hard negatives. Hospital clips were never translated (held-out domain).
"""
import glob, json, os, collections
import numpy as np, cv2, torch, torchvision

T = "/storage/hackathon_teams/omc-team15"
SNAP = f"{T}/codefest/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd/clips"
SRC = f"{T}/tmp/xfer/b2"
OUT = f"{T}/codefest/data_cosmos_b2"
MIN_AREA = 1300
for s in ("images", "labels"):
    os.makedirs(f"{OUT}/{s}", exist_ok=True)

dev = "cuda"
w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=w).eval().to(dev)


def persons(bgr):
    x = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(dev)
    with torch.no_grad():
        o = det([x])[0]
    k = (o["labels"] == 1) & (o["scores"] > 0.5)
    return o["boxes"][k].cpu().numpy()


def iou(a, g):
    ix = max(0., min(a[2], g[2]) - max(a[0], g[0])); iy = max(0., min(a[3], g[3]) - max(a[1], g[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i
    return i / u if u > 0 else 0.


rows, per_clip = [], {}
for mp4 in sorted(glob.glob(f"{SRC}/*/s0.7/*.mp4")):
    clip = os.path.basename(mp4)[:-4]
    coco = json.load(open(f"{SNAP}/{clip}/coco.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["file_name"].split("/")[-1]: im for im in coco["images"]}
    anns = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    cap = cv2.VideoCapture(mp4); i = 0; c = collections.Counter()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        im = imgs.get(f"frame_{i:04d}.jpg")
        a = max(anns.get(im["id"], []), key=lambda a: a["bbox"][2] * a["bbox"][3], default=None) if im else None
        r = dict(clip=clip, frame=i, environment=im and im["environment"], distance_m=im and im["distance_m"],
                 pose=im and im["pose"], category=None, area=None, iou=None, status=None)
        if a is None:
            r["status"] = "no_label"
        else:
            x, y, bw, bh = a["bbox"]; g = (x, y, x + bw, y + bh)
            r.update(category=cats[a["category_id"]], area=round(bw * bh))
            if bw * bh < MIN_AREA:
                r["status"] = "too_small"
            else:
                d = persons(f)
                v = max((iou(b, g) for b in d), default=0.0)
                r["iou"] = round(float(v), 4)
                r["status"] = "keep" if v >= 0.5 else "lost"
        if r["status"] == "keep":
            name = f"{clip}__{i:04d}"
            cv2.imwrite(f"{OUT}/images/{name}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            H, W = f.shape[:2]
            with open(f"{OUT}/labels/{name}.txt", "w") as lf:
                if "lying" in r["category"]:
                    lf.write(f"0 {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}\n")
        c[r["status"]] += 1; rows.append(r); i += 1
    cap.release()
    per_clip[clip] = dict(c)
    print(f"{clip[9:50]:<42} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())), flush=True)

with open(f"{OUT}/frames.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
tot = collections.Counter(r["status"] for r in rows)
kept = [r for r in rows if r["status"] == "keep"]
summary = dict(
    what="Cosmos Transfer 2.5 batch 2 (seg control, strength 0.7, 36 steps, 93 frames, 1280x720), label-checked per frame",
    clips=len(per_clip), frames=len(rows), status=dict(tot),
    staged=len(kept),
    staged_lying=sum("lying" in r["category"] for r in kept),
    staged_hard_negative=sum("lying" not in r["category"] for r in kept),
    staged_by_distance=dict(sorted(collections.Counter(str(r["distance_m"]) for r in kept).items(), key=lambda kv: float(kv[0]))),
    staged_by_environment=dict(collections.Counter(r["environment"] for r in kept)),
    rule="keep = COCO Faster R-CNN person (score>0.5) at IoU>=0.5 with the source box; boxes under 1300 px^2 cannot be checked and are not staged",
    labels="YOLO class 0 = lying casualty; warehouse_poses frames (standing/kneeling) have empty label files (hard negatives)",
    excluded="hospital clips (held-out evaluation domain); batch-1 clips (already in data_cosmos_armc)",
    warning="Do NOT score detectors fine-tuned on Isaac on these frames as an evaluation set; evaluate on real footage only.",
    per_clip=per_clip)
json.dump(summary, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=1))
