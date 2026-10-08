"""Geometry check of the re-textured renders: run detector v2b (checkpoints/rfdetr_s640_10class_v2b.pth from
ubr-physical-ai/rescue-det-permissive) on each source render and its re-textured copy, and match its 'box' detections
(score >= 0.3) to the source render's box labels at IoU >= 0.5. If the re-texturing moved or melted the boxes, recall on the
re-textured copies collapses; a modest drop is expected (the new surfaces are harder than the stock materials).
Also writes check_sheet.jpg: 12 pairs side by side with the labels drawn, for a visual check.
Usage: check_retexture.py CKPT WBOX_DIR RETEX_DIR [N=300]"""
import json, os, random, sys
import torch
from PIL import Image, ImageDraw
from rfdetr import RFDETR
ck, wbox, retex = sys.argv[1], sys.argv[2], sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
tr = json.load(open("train_annotations_v2b.coco.json")); box = [c["id"] for c in tr["categories"] if c["name"] == "box"][0]
by = {im["file_name"]: im for im in tr["images"]}; gt = {}
for a in tr["annotations"]:
    if a["category_id"] == box: gt.setdefault(a["image_id"], []).append(a["bbox"])
files = sorted(f for f in os.listdir(retex) if ("wbox_" + f) in by); random.Random(0).shuffle(files); files = files[:N]
m = RFDETR.from_checkpoint(ck)
def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)
def recall(img, g):
    with torch.inference_mode(): d = m.predict(img, threshold=0.3)
    det = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), k in zip(d.xyxy, d.class_id) if int(k) == 8]
    return sum(any(iou(b, x) >= 0.5 for x in det) for b in g), len(g)
hs = ht = rs = 0; pairs = []
for f in files:
    g = gt.get(by["wbox_" + f]["id"], [])
    if not g: continue
    a = Image.open(f"{wbox}/train/{f}").convert("RGB"); b = Image.open(f"{retex}/{f}").convert("RGB")
    assert a.size == b.size, (f, a.size, b.size)
    h, t = recall(a, g); r, _ = recall(b, g); hs += h; ht += t; rs += r
    if len(pairs) < 12: pairs.append((a, b, g))
print(f"box recall at IoU 0.5, v2b detector, {len(files)} renders: source {hs / ht:.3f}, re-textured {rs / ht:.3f} ({ht} boxes)")
W, H = 640, 360; sheet = Image.new("RGB", (2 * W, len(pairs) * H))
for i, (a, b, g) in enumerate(pairs):
    for j, im in enumerate((a, b)):
        im = im.copy(); dr = ImageDraw.Draw(im)
        for x, y, w, h in g: dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
        sheet.paste(im.resize((W, H)), (j * W, i * H))
sheet.save("check_sheet.jpg", quality=85); print("wrote check_sheet.jpg")
