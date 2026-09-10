#!/usr/bin/env python3
"""Task F: turn Replicator COCO output into LLaVA-format spatial QA pairs matching the PhysicalAI val.json templates.

  python scripts/replicator_to_spatial_qa.py --coco <coco.json> --out F_qa.json [--images-root DIR] [--seed 0]
  python scripts/replicator_to_spatial_qa.py --metadata <metadata.jsonl> --out F_qa.json     # v1 fallback, see below

Input needs per-image `distance_m`, `bearing_deg` and `environment` (the v2 package carries these on each COCO image
record). Generates two question types, mirroring the dataset's own wording:

  left_right : "From this viewpoint, does the <a> <mask> appear on the right-hand side of the <b> <mask>?"
               normalized_answer "left" | "right", decided by bearing thirds (left third vs right third; the middle
               third is skipped as ambiguous rather than guessed).
  distance   : "...can you measure the distance between <a> and <b>?"  normalized_answer = metres to one decimal.

v1 `metadata.jsonl` has no distance_m/bearing_deg, so with --metadata the script derives a *proxy* bearing from the
box centre x (image thirds) and emits left_right pairs only, clearly tagged `source: "proxy-v1"`. That path exists to
test the generator end to end before the v2 package lands -- it is not a substitute for rendered geometry.
"""
import argparse, json, math, os, random, sys
from pathlib import Path

def bearing_side(bearing_deg, dead_zone=10.0):
    """Negative bearing = left of the optical axis. Returns 'left'/'right'/None inside the dead zone."""
    if bearing_deg is None: return None
    if abs(bearing_deg) < dead_zone: return None
    return "left" if bearing_deg < 0 else "right"

def make_pairs(records, seed=0):
    rnd = random.Random(seed)
    out = []
    for r in records:
        img, objs = r["image"], r["objects"]
        if len(objs) < 2: continue
        for _ in range(min(2, len(objs) // 2)):
            a, b = rnd.sample(objs, 2)
            sa, sb = a.get("side"), b.get("side")
            if sa and sb and sa != sb:
                ans = "right" if sa == "right" else "left"
                q = (f"From this viewpoint, does the {a['name']} <mask> appear on the "
                     f"right-hand side of the {b['name']} <mask>?")
                free = f"The {a['name']} [Region 0] is to the {ans} of the {b['name']} [Region 1]."
                out.append({"id": f"{Path(img).stem}_lr_{len(out)}", "image": img, "category": "left_right",
                            "conversations": [{"from": "human", "value": "<image>\n" + q},
                                              {"from": "gpt", "value": free}],
                            "normalized_answer": ans, "freeform_answer": free,
                            "rle": [a.get("rle"), b.get("rle")], "source": r.get("source", "v2")})
            da, db = a.get("distance_m"), b.get("distance_m")
            if da is not None and db is not None:
                d = round(abs(da - db), 1)
                q = (f"Considering the {a['name']} <mask> and the {b['name']} <mask>, can you measure the "
                     f"distance between them?")
                free = (f"The spatial distance between the {a['name']} [Region 0] and the {b['name']} "
                        f"[Region 1] is {d} metres.")
                out.append({"id": f"{Path(img).stem}_ds_{len(out)}", "image": img, "category": "distance",
                            "conversations": [{"from": "human", "value": "<image>\n" + q},
                                              {"from": "gpt", "value": free}],
                            "normalized_answer": f"{d}", "freeform_answer": free,
                            "rle": [a.get("rle"), b.get("rle")], "source": r.get("source", "v2")})
    return out

def from_coco(path):
    coco = json.load(open(path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = {}
    for im in coco["images"]:
        by_img[im["id"]] = {"image": os.path.basename(im["file_name"]), "objects": [],
                            "environment": im.get("environment"), "source": "v2"}
    miss = 0
    for an in coco["annotations"]:
        rec = by_img.get(an["image_id"])
        if rec is None: continue
        d, b = an.get("distance_m", an.get("distance")), an.get("bearing_deg", an.get("bearing"))
        if d is None or b is None: miss += 1
        rec["objects"].append({"name": cats.get(an["category_id"], "object"), "distance_m": d,
                               "bearing_deg": b, "side": bearing_side(b), "rle": an.get("segmentation")})
    if miss: print(f"[warn] {miss} annotations lack distance_m/bearing_deg", file=sys.stderr)
    return list(by_img.values())

def from_metadata_v2(path, person_ids=frozenset(range(1, 14))):
    """v2 package. Two measured facts are available and one plan assumption is not:

      distance_m   per IMAGE, camera -> person. Verified real (Spearman -0.871 against apparent box height), so
                   distance questions are camera-relative ("how far is the person from the camera?"), NOT the
                   benchmark's object-to-object distance -- v2 carries no per-distractor range.
      bearing_deg  per image, 0-360, but it does NOT encode where the person appears in frame: mean box-centre-x is
                   0.500 in every 30-degree bin and every left/right convention agrees with frame position at
                   chance (n=12,196). It is a camera-azimuth / world-orientation parameter. left/right is therefore
                   taken from the renderer's own box centres, which are exact.
    """
    recs = []
    for line in open(path):
        r = json.loads(line)
        o = r["objects"]; W = r["width"]
        objs = []
        for name, cat, bbox in zip(o["category_name"], o["category"], o["bbox"]):
            cx = bbox[0] + bbox[2] / 2
            objs.append({"name": name, "cx": cx, "is_person": cat in person_ids,
                         "distance_m": r.get("distance_m") if cat in person_ids else None,
                         "bearing_deg": None, "bbox": [round(v, 1) for v in bbox],
                         "side": "left" if cx < W / 2 else "right", "rle": None})
        recs.append({"image": r["file_name"], "objects": objs, "environment": r.get("environment"),
                     "person_visible": r.get("person_visible"), "width": W, "source": "v2"})
    return recs

def make_pairs_v2(records, seed=0):
    """left_right from box centres (pairwise, matching the benchmark's shape); distance camera-relative."""
    rnd = random.Random(seed); out = []
    for r in records:
        objs = r["objects"]; W = r["width"]
        if len(objs) >= 2:
            for _ in range(min(2, len(objs) // 2)):
                a, b = rnd.sample(objs, 2)
                # only ask when the two are unambiguously separated (>10% of frame width apart)
                if abs(a["cx"] - b["cx"]) < 0.10 * W: continue
                ans = "right" if a["cx"] > b["cx"] else "left"
                q = (f"From this viewpoint, does the {a['name']} <mask> appear on the "
                     f"right-hand side of the {b['name']} <mask>?")
                free = f"The {a['name']} [Region 0] is to the {ans} of the {b['name']} [Region 1]."
                out.append({"id": f"{Path(r['image']).stem}_lr_{len(out)}", "image": r["image"],
                            "category": "left_right",
                            "conversations": [{"from": "human", "value": "<image>\n" + q},
                                              {"from": "gpt", "value": free}],
                            "normalized_answer": ans, "freeform_answer": free, "rle": [None, None],
                            "bbox": [a["bbox"], b["bbox"]],   # v2 ships no masks; boxes are exact, render these as regions
                            "source": "v2", "template": "benchmark-pairwise"})
        p = next((o for o in objs if o["is_person"] and o["distance_m"] is not None), None)
        if p and r.get("person_visible"):
            d = round(float(p["distance_m"]), 1)
            q = f"How far is the {p['name']} <mask> from the camera?"
            free = f"The {p['name']} [Region 0] is {d} metres from the camera."
            out.append({"id": f"{Path(r['image']).stem}_ds_{len(out)}", "image": r["image"],
                        "category": "distance",
                        "conversations": [{"from": "human", "value": "<image>\n" + q},
                                          {"from": "gpt", "value": free}],
                        "normalized_answer": f"{d}", "freeform_answer": free, "rle": [None],
                        "bbox": [p["bbox"]],
                        "source": "v2", "template": "camera-relative"})
    return out

def from_metadata(path):
    recs = []
    for line in open(path):
        r = json.loads(line)
        o = r["objects"]; W = r["width"]
        objs = []
        for name, bbox in zip(o["category_name"], o["bbox"]):
            cx = bbox[0] + bbox[2] / 2
            side = "left" if cx < W / 3 else ("right" if cx > 2 * W / 3 else None)
            objs.append({"name": name, "distance_m": None, "bearing_deg": None, "side": side, "rle": None})
        recs.append({"image": r["file_name"], "objects": objs, "environment": r.get("environment"),
                     "source": "proxy-v1"})
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco"); ap.add_argument("--metadata"); ap.add_argument("--metadata-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not (a.coco or a.metadata or a.metadata_v2): sys.exit("need --coco, --metadata or --metadata-v2")
    if a.metadata_v2:
        recs = from_metadata_v2(a.metadata_v2); pairs = make_pairs_v2(recs, a.seed)
    else:
        recs = from_coco(a.coco) if a.coco else from_metadata(a.metadata)
        pairs = make_pairs(recs, a.seed)
    Path(a.out).write_text(json.dumps(pairs, indent=1))
    from collections import Counter
    c = Counter(p["category"] for p in pairs)
    print(f"{len(recs)} images -> {len(pairs)} QA pairs: {dict(c)}")
    print(f"source: {Counter(p['source'] for p in pairs)}")
    if pairs: print("sample:", json.dumps(pairs[0], indent=1)[:500])

if __name__ == "__main__":
    main()
