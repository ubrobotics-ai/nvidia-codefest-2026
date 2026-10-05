"""Can Cosmos3-Edge name an object the detector boxed? Measured on COCO val2017 with known labels.

  prep:  python naming_eval.py prep       -> crops/ and outline/ PNGs + in_{crop,outline}.json
  score: python naming_eval.py score OUT.json [OUT.json ...]

480 objects: 6 per COCO class (all 80), non-crowd, both box sides >= 48 px, seed 1234.
Two presentations of the same object:
  crop:    the box plus 20% margin on each side, long side scaled to at least 336 px
  outline: the full frame with that one box outlined in red (a single mark, not numbered regions,
           which scored near chance in the earlier L0 test)
Prompts contain no digits. Answers are scored leniently: correct if the lower-cased answer contains
the COCO class name or one of its listed synonyms. Misses are written out for manual review.
"""
import json, os, random, sys, collections, re

T = "/storage/hackathon_teams/omc-team15/codefest"
D = f"{T}/naming_eval"
COCO = f"{T}/data_coco"
PROMPT = {
    "crop": "What is the main object shown in this image? Answer with the object's name only, in one to three words.",
    "outline": "What is the object outlined by the red rectangle in this image? Answer with the object's name only, in one to three words.",
}
SYN = {
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid", "people", "player", "skier", "surfer", "human", "lady", "guy"],
    "bicycle": ["bicycle", "bike"], "car": ["car", "vehicle", "sedan", "taxi"], "motorcycle": ["motorcycle", "motorbike", "scooter", "moped"],
    "airplane": ["airplane", "plane", "aircraft", "jet", "airliner"], "bus": ["bus"], "train": ["train", "locomotive", "tram"],
    "truck": ["truck", "lorry", "pickup"], "boat": ["boat", "ship", "kayak", "canoe", "yacht", "sailboat"],
    "traffic light": ["traffic light", "traffic signal", "stoplight"], "fire hydrant": ["hydrant"], "stop sign": ["stop sign"],
    "parking meter": ["parking meter", "meter"], "bench": ["bench"], "bird": ["bird", "seagull", "duck", "pigeon", "gull", "parrot"],
    "cat": ["cat", "kitten"], "dog": ["dog", "puppy"], "horse": ["horse", "pony"], "sheep": ["sheep", "lamb"], "cow": ["cow", "cattle", "bull", "calf"],
    "elephant": ["elephant"], "bear": ["bear"], "zebra": ["zebra"], "giraffe": ["giraffe"], "backpack": ["backpack", "rucksack", "bag"],
    "umbrella": ["umbrella", "parasol"], "handbag": ["handbag", "purse", "bag"], "tie": ["tie", "necktie"], "suitcase": ["suitcase", "luggage", "bag"],
    "frisbee": ["frisbee", "disc"], "skis": ["ski"], "snowboard": ["snowboard"], "sports ball": ["ball"], "kite": ["kite"],
    "baseball bat": ["bat"], "baseball glove": ["glove", "mitt"], "skateboard": ["skateboard"], "surfboard": ["surfboard", "board"],
    "tennis racket": ["racket", "racquet"], "bottle": ["bottle"], "wine glass": ["glass"], "cup": ["cup", "mug"], "fork": ["fork"],
    "knife": ["knife"], "spoon": ["spoon"], "bowl": ["bowl"], "banana": ["banana"], "apple": ["apple"], "sandwich": ["sandwich", "burger"],
    "orange": ["orange", "citrus"], "broccoli": ["broccoli"], "carrot": ["carrot"], "hot dog": ["hot dog", "hotdog", "sausage"],
    "pizza": ["pizza"], "donut": ["donut", "doughnut"], "cake": ["cake", "cupcake"], "chair": ["chair", "seat", "stool"],
    "couch": ["couch", "sofa"], "potted plant": ["plant", "flower", "pot"], "bed": ["bed"], "dining table": ["table"],
    "toilet": ["toilet"], "tv": ["tv", "television", "monitor", "screen"], "laptop": ["laptop", "computer", "notebook"],
    "mouse": ["mouse"], "remote": ["remote", "controller"], "keyboard": ["keyboard"], "cell phone": ["phone", "smartphone", "cellphone", "mobile"],
    "microwave": ["microwave"], "oven": ["oven", "stove"], "toaster": ["toaster"], "sink": ["sink", "basin"], "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"], "clock": ["clock"], "vase": ["vase"], "scissors": ["scissors"], "teddy bear": ["teddy", "bear", "stuffed"],
    "hair drier": ["hair dryer", "hair drier", "hairdryer", "blow dryer", "dryer"], "toothbrush": ["toothbrush"],
}


def prep():
    import cv2
    d = json.load(open(f"{COCO}/annotations/instances_val2017.json"))
    cn = {c["id"]: c["name"] for c in d["categories"]}
    ims = {i["id"]: i for i in d["images"]}
    by = collections.defaultdict(list)
    for a in d["annotations"]:
        if not a["iscrowd"] and a["bbox"][2] >= 48 and a["bbox"][3] >= 48:
            by[cn[a["category_id"]]].append(a)
    rnd = random.Random(1234)
    items = []
    for c in sorted(by):
        for a in rnd.sample(by[c], min(6, len(by[c]))):
            items.append((c, a))
    for v in ("crop", "outline"):
        os.makedirs(f"{D}/{v}", exist_ok=True)
    reqs = {"crop": [], "outline": []}; meta = []
    for k, (c, a) in enumerate(items):
        im = cv2.imread(f"{COCO}/val2017/{ims[a['image_id']]['file_name']}"); H, W = im.shape[:2]
        x, y, w, h = a["bbox"]
        x1, y1 = int(max(0, x - 0.2 * w)), int(max(0, y - 0.2 * h)); x2, y2 = int(min(W, x + 1.2 * w)), int(min(H, y + 1.2 * h))
        crop = im[y1:y2, x1:x2]; s = max(1.0, 336 / max(crop.shape[:2]))
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        name = f"obj_{k:04d}.png"
        cv2.imwrite(f"{D}/crop/{name}", crop)
        o = im.copy(); cv2.rectangle(o, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), max(2, int(0.006 * max(H, W))))
        cv2.imwrite(f"{D}/outline/{name}", o)
        meta.append({"k": k, "name": name, "category": c, "ann_id": a["id"], "image_id": a["image_id"]})
        for v in ("crop", "outline"):
            reqs[v].append({"messages": [{"role": "user", "content": [
                {"type": "image", "image": f"{D}/{v}/{name}"}, {"type": "text", "text": PROMPT[v]}]}]})
    for v in ("crop", "outline"):
        json.dump({"batch_size": 1, "temperature": 0.0, "max_generate_length": 24, "requests": reqs[v]},
                  open(f"{D}/in_{v}.json", "w"))
    json.dump(meta, open(f"{D}/meta.json", "w"))
    print(len(items), "objects,", len(by), "classes")


def score(paths):
    meta = json.load(open(f"{D}/meta.json"))
    for p in paths:
        out = json.load(open(p))
        resp = [r.get("output_text", "") for r in out["responses"]]
        ok, miss = 0, []
        per = collections.Counter(); tot = collections.Counter()
        for m, r in zip(meta, resp):
            a = re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip().lower()
            hit = any(s in a for s in SYN[m["category"]] + [m["category"]])
            ok += hit; tot[m["category"]] += 1; per[m["category"]] += hit
            if not hit:
                miss.append((m["category"], a[:60]))
        print(f"{os.path.basename(p):<28} named correctly {ok}/{len(meta)} = {ok / len(meta):.3f}")
        worst = sorted(tot, key=lambda c: per[c] / tot[c])[:8]
        print("   weakest classes:", ", ".join(f"{c} {per[c]}/{tot[c]}" for c in worst))
        json.dump(miss, open(p.replace(".json", "_misses.json"), "w"), indent=0)
        print("   sample misses:", miss[:8])


if __name__ == "__main__":
    prep() if sys.argv[1] == "prep" else score(sys.argv[2:])
