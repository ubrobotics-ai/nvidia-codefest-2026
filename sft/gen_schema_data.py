#!/usr/bin/env python3
"""Phase 1b — grounded-perception schema examples, generated against the REAL contract.

Vocabularies come from brain/perception/cosmos_vision.py GROUNDED_PROMPT, supplied by the
brain session, not guessed:
    scene_type      room | corridor | doorway | open_area | unknown
    task_state      no_target | candidate | target_visible | uncertain
    task_relevance  target | distractor | uncertain        <- enum, load-bearing
    intent          continue_search | inspect_candidate | approach_candidate | stop
    attributes[]    free-form {name, value, certainty}, certainty in {observed, uncertain}
    label           open, "concise visible category"
    bbox_2d         [x1,y1,x2,y2], integers 0-1000, origin top-left, axes normalised
                    independently

task_relevance is the field with a downstream numeric consequence: the consumer maps it to
confidence (target 0.75, uncertain 0.45, distractor 0.30) and perception.cosmos_min_role keeps
{target, uncertain} by default, so "distractor" REMOVES the entity from what the robot acts
on. When omitted the parser defaults to "uncertain" -- that default is the guess this bucket
exists to train away.

Source: our own isaac-sdg-rescue-target-v3 (Isaac Sim renders, public, CC-BY-4.0). Not in
holdout_never_train: the 24-item grounded benchmark and the 60 frozen frames are real lw-lab1
footage, disjoint from these synthetic renders.

WHAT IS ASSUMED, AND IT IS NOT NOTHING. The COCO metadata carries pose, vest, distance and
visibility; it does NOT carry task semantics. The target/distractor policy below is this
script's invention, applied consistently and recorded here so it can be overruled by one edit
rather than archaeology:

  - an incapacitated posture (lying / supine / prone / seated / kneeling / crouching /
    bending) is the casualty the task is looking for  -> target
  - a standing or walking person is mobile, so not the casualty, but is a person and may be a
    responder                                          -> uncertain, never distractor
  - the dataset's own `distractor` class is scene clutter -> distractor
  - beyond 25 m a prone body is 10-20 px, past the published detection wall, so anything at
    that range is downgraded to uncertain and every attribute certainty becomes "uncertain"

COVERAGE GAP, stated rather than papered over: the Isaac environments are halls, warehouses,
offices, hospitals and outdoor plots. Nothing in them is a corridor or a doorway, so two of
the five scene_type values are NEVER TRAINED by this bucket. A model fine-tuned only on this
may not emit them at all. Real frames are needed for those two.
"""
import argparse, collections, json, os, pathlib, random, re, sys

SCENE = {"warehouse": "open_area", "hall": "open_area", "office": "room", "hospital": "room",
         "rivermark": "open_area", "rough_plane": "open_area", "slope": "open_area",
         "stairs": "unknown", "void": "unknown"}
DOWN = {"lying": "lying flat", "supine": "lying on their back", "prone": "lying face down",
        "seated": "seated", "kneeling": "kneeling", "crouching": "crouching", "bending": "bending over"}
UP = {"standing": "standing", "walking": "walking"}
FAR_M = 25.0

def pose_of(cat, img):
    p = img.get("pose") or ""
    if p in DOWN or p in UP: return p
    m = re.match(r"person_(\w+?)(_vest)?$", cat or "")
    return m.group(1) if m else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=os.environ.get("TEAM","")+"/data/isaac-sdg-rescue-target-v3/annotations/coco_train.json")
    ap.add_argument("--root", default=os.environ.get("TEAM","")+"/data/isaac-sdg-rescue-target-v3")
    ap.add_argument("--out", default="sft/data/schema.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    c = json.load(open(a.coco))
    cats = {x["id"]: x["name"] for x in c["categories"]}
    by_img = collections.defaultdict(list)
    for an in c["annotations"]: by_img[an["image_id"]].append(an)
    images = {i["id"]: i for i in c["images"]}

    # Only images actually extracted on disk. The COCO index covers all 120,384 renders but
    # the repo ships most of them inside 37 tar shards, so sampling the index directly picks
    # paths that do not exist -- the audit caught 149 of 150 on the first run.
    on_disk = {p.name for p in (pathlib.Path(a.root) / "train").glob("*.jpg")}
    images = {i: im for i, im in images.items()
              if os.path.basename(im["file_name"]) in on_disk}
    print(f"  images in COCO index: {len(c['images'])}, extracted on disk: {len(images)}")

    # Stratify: every environment, and a deliberate share of empty scenes so no_target and
    # continue_search are trained rather than inferred.
    with_p = [i for i in images if by_img.get(i) and images[i].get("person_visible")]
    empty  = [i for i in images if not by_img.get(i)]
    by_env = collections.defaultdict(list)
    for i in with_p: by_env[images[i]["environment"]].append(i)
    picked, per_env = [], max(1, int(a.n * 0.75) // max(len(by_env), 1))
    for env, ids in sorted(by_env.items()):
        random.shuffle(ids); picked += ids[:per_env]
    random.shuffle(empty)
    picked += empty[: a.n - len(picked)]
    random.shuffle(picked); picked = picked[: a.n]

    rows, stats = [], collections.Counter()
    for iid in picked:
        img = images[iid]; W, H = img["width"], img["height"]
        far = (img.get("distance_m") or 0) > FAR_M
        ents = []
        # Persons first, then at most MAX_DISTRACTOR clutter boxes. Taking the 8 largest by
        # area instead gave 552 distractors to 67 targets -- and "distractor" is the value
        # that REMOVES an entity from what the robot acts on, so an 8:1 prior toward it is
        # the one imbalance in this schema that can get a casualty ignored. The prompt asks
        # for the salient entities, not an exhaustive inventory of clutter.
        MAX_DISTRACTOR = 2
        anns = by_img.get(iid, [])
        persons = sorted([x for x in anns if cats[x["category_id"]] != "distractor"],
                         key=lambda x: -x["area"])
        clutter = sorted([x for x in anns if cats[x["category_id"]] == "distractor"],
                         key=lambda x: -x["area"])[:MAX_DISTRACTOR]
        for k, an in enumerate(persons + clutter):
            x, y, w, h = an["bbox"]
            if w < 2 or h < 2: continue
            box = [int(round(1000*x/W)), int(round(1000*y/H)),
                   int(round(1000*(x+w)/W)), int(round(1000*(y+h)/H))]
            box = [max(0, min(1000, v)) for v in box]
            if box[2] <= box[0] or box[3] <= box[1]: continue
            name = cats[an["category_id"]]
            cert = "uncertain" if far else "observed"
            if name == "distractor":
                # "small object on the floor" and not a guessed identity: the renders place
                # 6 props per scene at scale 0.1-0.4 and z 0.15-0.5 (DexCube, KLT bin, pallet,
                # forklift), but COCO gives every one of them category_id 0, so the instance is
                # genuinely unknown. Naming it would train the model to invent an identity it
                # cannot see -- the same failure the tolerant_entities() bug was faking.
                ents.append({"entity_id": f"e{k+1}", "label": "small object on the floor",
                             "bbox_2d": box,
                             "attributes": [{"name": "kind", "value": "not a person", "certainty": cert}],
                             "task_relevance": "distractor"})
                stats["distractor"] += 1
                continue
            p = pose_of(name, img)
            vest = name.endswith("_vest")
            desc = DOWN.get(p) or UP.get(p) or "in an unclear posture"
            label = f"person {desc}" + (" in a hi-vis vest" if vest else "")
            attrs = [{"name": "posture", "value": desc, "certainty": cert}]
            if vest: attrs.append({"name": "clothing", "value": "hi-vis vest", "certainty": cert})
            if far: attrs.append({"name": "range", "value": "far, small in frame", "certainty": "observed"})
            rel = "uncertain" if (far or p in UP) else ("target" if p in DOWN else "uncertain")
            ents.append({"entity_id": f"e{k+1}", "label": label, "bbox_2d": box,
                         "attributes": attrs, "task_relevance": rel})
            stats[rel] += 1

        rels = {e["task_relevance"] for e in ents}
        if not ents:                       state, intent = "no_target", "continue_search"
        elif "target" in rels:             state, intent = "target_visible", "approach_candidate"
        elif "uncertain" in rels and any(e["label"].startswith("person") for e in ents):
                                           state, intent = "candidate", "inspect_candidate"
        else:                              state, intent = "no_target", "continue_search"
        if far and state == "target_visible": state, intent = "uncertain", "inspect_candidate"
        stats[f"state:{state}"] += 1

        rows.append({"image": os.path.join(a.root, img["file_name"]),
                     "target": {"schema_version": "0.1",
                                "scene_type": SCENE.get(img["environment"], "unknown"),
                                "entities": ents, "task_state": state, "intent": intent}})

    # Blocking audit. Same posture as gen_command_data.py: refuse to write bad data.
    SCENES = {"room","corridor","doorway","open_area","unknown"}
    STATES = {"no_target","candidate","target_visible","uncertain"}
    RELS   = {"target","distractor","uncertain"}
    INTENTS= {"continue_search","inspect_candidate","approach_candidate","stop"}
    bad = []
    for r in rows:
        t = r["target"]
        if t["scene_type"] not in SCENES: bad.append(("scene_type", t["scene_type"]))
        if t["task_state"] not in STATES: bad.append(("task_state", t["task_state"]))
        if t["intent"] not in INTENTS:    bad.append(("intent", t["intent"]))
        if not os.path.exists(r["image"]): bad.append(("missing image", r["image"]))
        for e in t["entities"]:
            if e["task_relevance"] not in RELS: bad.append(("task_relevance", e["task_relevance"]))
            b = e["bbox_2d"]
            if len(b) != 4 or any((not isinstance(v, int)) or v < 0 or v > 1000 for v in b):
                bad.append(("bbox range", b))
            if b[2] <= b[0] or b[3] <= b[1]: bad.append(("bbox order", b))
            for at in e["attributes"]:
                if at["certainty"] not in {"observed","uncertain"}: bad.append(("certainty", at["certainty"]))
            if not re.fullmatch(r"[\x20-\x7e]+", e["label"]): bad.append(("label charset", e["label"]))
    print(f"  examples: {len(rows)}")
    for k, v in sorted(stats.items()): print(f"    {k:<24} {v}")
    print(f"  scene_type coverage: {collections.Counter(r['target']['scene_type'] for r in rows)}")
    print(f"  entities per example: mean {sum(len(r['target']['entities']) for r in rows)/max(len(rows),1):.2f}")
    if bad:
        for k, v in bad[:8]: print(f"    BAD {k}: {v}")
        sys.exit(f"  refusing to write {len(bad)} malformed rows")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    print(f"  wrote {a.out}")

if __name__ == "__main__":
    main()
