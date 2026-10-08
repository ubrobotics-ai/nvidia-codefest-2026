"""Pick the warehouse renders to re-texture: N (default 3000) of the 9,000 v2b training renders (full_warehouse stage),
seed 1234, only frames with at least one labelled box. Writes retexture_list.json: [{"name", "rel", "prompt"}].
The prompt varies only the boxes' surface and the scene's lighting/wear; geometry is held by the edge control."""
import json, random, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ann = json.load(open("train_annotations_v2b.coco.json"))
box = [c["id"] for c in ann["categories"] if c["name"] == "box"][0]
has_box = {a["image_id"] for a in ann["annotations"] if a["category_id"] == box}
man = {m["name"]: m["rel"] for m in map(json.loads, open("manifest.jsonl")) if m["split"] == "train" and m["src"] == "wbox"}
cands = sorted(im["file_name"] for im in ann["images"] if im["file_name"] in man and im["id"] in has_box)
rng = random.Random(1234); rng.shuffle(cands)
SURF = ["plain brown corrugated cardboard cartons", "brown cardboard cartons with clear packing tape and white shipping labels",
        "printed retail cartons in mixed colours with logos and barcodes", "worn, dented and stained cardboard boxes",
        "white cardboard boxes with black printed text", "brown cartons with red and yellow fragile stickers and handwritten marker",
        "shrink-wrapped stacks of cardboard boxes on pallets", "dark grey and blue plastic storage crates and cardboard cartons"]
LIGHT = ["bright even fluorescent light", "dim warm light with deep shadows", "cool daylight from high windows", "harsh overhead spotlights"]
out = [{"name": n, "rel": man[n], "prompt": f"A realistic photo inside a busy warehouse at floor level, {rng.choice(SURF)} on shelves and on the floor, "
        f"concrete floor, metal racking, {rng.choice(LIGHT)}. Photorealistic, natural camera noise."} for n in cands[:N]]
json.dump(out, open("retexture_list.json", "w"), indent=0); print(len(out), "renders picked of", len(cands))
