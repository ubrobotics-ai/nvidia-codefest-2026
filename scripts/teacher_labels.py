#!/usr/bin/env python3
"""Task D: label real rover frames with the v0 detector and an open teacher (Grounding-DINO), and compare them.

  python scripts/teacher_labels.py --frames $TEAM/data/blackbox_frames --out $WORK/logs/night-20260910 \
      --v0 $WORK/runs/v0/weights/best.pt [--teacher IDEA-Research/grounding-dino-base] [--limit N]

Writes D_v0.json / D_teacher.json (COCO), D_agreement.csv (per-frame), D_disagree.jpg (contact sheet of the worst
disagreements) and prints the summary numbers. Labels only -- nothing is uploaded and no frame is modified.
"""
import argparse, csv, json, glob, os, sys, time
from pathlib import Path

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--v0", required=True); ap.add_argument("--teacher", default="IDEA-Research/grounding-dino-base")
    # The generic "person" prompt scores a PRONE figure ~0.27 while "a person lying on the ground" scores it ~0.76
    # on the same frame. The first pass used the generic prompt and therefore under-counts exactly the posture the
    # rescue detector exists to find. Prompts are OR-ed: a frame keeps the union of what each finds.
    ap.add_argument("--prompts", nargs="+", default=["person"])
    ap.add_argument("--conf-v0", type=float, default=0.25); ap.add_argument("--conf-teacher", type=float, default=0.30)
    ap.add_argument("--box-threshold", type=float, default=0.30); ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.frames, "*.jpg")) + glob.glob(os.path.join(a.frames, "*.png")))
    if a.limit: files = files[: a.limit]
    if not files: sys.exit(f"no frames in {a.frames}")
    print(f"{len(files)} frames", flush=True)

    import torch
    from PIL import Image
    from ultralytics import YOLO
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    # --- v0 detector
    t0 = time.time(); v0 = YOLO(a.v0)
    v0_boxes = {}
    for i in range(0, len(files), 32):
        for f, r in zip(files[i:i+32], v0.predict(files[i:i+32], conf=a.conf_v0, verbose=False, device=0)):
            b = []
            if r.boxes is not None:
                for cls, conf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    if int(cls) != 0: continue          # person_lying_vest only
                    x1, y1, x2, y2 = xyxy; b.append(([x1, y1, x2 - x1, y2 - y1], conf))
            v0_boxes[f] = b
        if i % 320 == 0: print(f"  v0 {i}/{len(files)}", flush=True)
    v0_s = time.time() - t0
    print(f"v0 done in {v0_s:.0f}s", flush=True)

    # --- teacher
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.teacher)
    # fp32 weights + autocast: converting Grounding-DINO's weights to bf16 breaks its text/vision cross-attention
    # (dtype mismatches in both directions); autocast gives the same speed without touching the checkpoint.
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.teacher).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    te_boxes = {}
    for i in range(0, len(files), a.batch):
        chunk = files[i:i+a.batch]
        imgs = [Image.open(f).convert("RGB") for f in chunk]
        inp = proc(images=imgs, text=[list(a.prompts)] * len(imgs), return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inp)
        res = proc.post_process_grounded_object_detection(
            o, inp["input_ids"], threshold=a.box_threshold, text_threshold=a.text_threshold,
            target_sizes=[im.size[::-1] for im in imgs])
        for f, r in zip(chunk, res):
            b = []
            for score, box in zip(r["scores"].tolist(), r["boxes"].tolist()):
                if score < a.conf_teacher: continue
                x1, y1, x2, y2 = box; b.append(([x1, y1, x2 - x1, y2 - y1], score))
            te_boxes[f] = b
        if i % (a.batch * 20) == 0: print(f"  teacher {i}/{len(files)}", flush=True)
    te_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"teacher done in {te_s:.0f}s, peak {peak:.1f} GB", flush=True)

    # --- COCO out
    def to_coco(boxes, name):
        imgs, anns, k = [], [], 1
        for j, f in enumerate(files, 1):
            w, h = Image.open(f).size
            imgs.append({"id": j, "file_name": Path(f).name, "width": w, "height": h})
            for bb, sc in boxes[f]:
                anns.append({"id": k, "image_id": j, "category_id": 1, "bbox": [round(v, 1) for v in bb],
                             "area": round(bb[2] * bb[3], 1), "score": round(sc, 4), "iscrowd": 0}); k += 1
        d = {"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
        (out / name).write_text(json.dumps(d))
        return len(anns)
    n_v0 = to_coco(v0_boxes, "D_v0.json"); n_te = to_coco(te_boxes, "D_teacher.json")

    # --- agreement
    rows, tp, fn_, fp_ = [], 0, 0, 0
    for f in files:
        vb, tb = v0_boxes[f], te_boxes[f]
        matched = set()
        m = 0
        for i_t, (tbb, _) in enumerate(tb):
            best, bi = 0.0, -1
            for i_v, (vbb, _) in enumerate(vb):
                if i_v in matched: continue
                s = iou(tbb, vbb)
                if s > best: best, bi = s, i_v
            if best >= a.iou: matched.add(bi); m += 1
        tp += m; fn_ += len(tb) - m; fp_ += len(vb) - m
        rows.append({"file": Path(f).name, "n_v0": len(vb), "n_teacher": len(tb), "matched": m,
                     "teacher_areas": ";".join(f"{bb[2]*bb[3]:.0f}" for bb, _ in tb)})
    with open(out / "D_agreement.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    frames_person = sum(1 for r in rows if r["n_teacher"] > 0)
    areas = [float(x) for r in rows for x in r["teacher_areas"].split(";") if x]
    agree = 100.0 * tp / n_te if n_te else float("nan")
    print(f"\n== N frames {len(files)}")
    print(f"== frames with >=1 person (teacher): {frames_person} ({100*frames_person/len(files):.1f}%)")
    print(f"== teacher boxes {n_te}, v0 boxes {n_v0}")
    print(f"== agreement IoU>={a.iou}: {tp}/{n_te} teacher boxes matched = {agree:.1f}%  (v0-only boxes {fp_}, teacher-only {fn_})")
    if areas:
        areas.sort()
        import statistics as st
        buckets = [(0,1024),(1024,4096),(4096,16384),(16384,65536),(65536,10**9)]
        print("== teacher box area histogram (px^2):")
        for lo,hi in buckets:
            c=sum(1 for x in areas if lo<=x<hi); print(f"     {lo:>7}-{hi:<9} {c:5d}  {100*c/len(areas):5.1f}%")
        print(f"     median {st.median(areas):.0f}, IQR {st.quantiles(areas,n=4)[0]:.0f}-{st.quantiles(areas,n=4)[2]:.0f}")
    (out / "D_summary.json").write_text(json.dumps({
        "n_frames": len(files), "frames_with_person_teacher": frames_person, "teacher_boxes": n_te,
        "v0_boxes": n_v0, "matched": tp, "agreement_pct": round(agree, 2), "v0_only": fp_, "teacher_only": fn_,
        "v0_seconds": round(v0_s), "teacher_seconds": round(te_s), "teacher_peak_gb": round(peak, 1),
        "conf_v0": a.conf_v0, "conf_teacher": a.conf_teacher, "iou": a.iou, "teacher_model": a.teacher}, indent=2))

    # --- disagreement contact sheet
    worst = sorted(rows, key=lambda r: -(abs(r["n_teacher"] - r["matched"]) + abs(r["n_v0"] - r["matched"])))[:20]
    from PIL import ImageDraw
    tiles = []
    for r in worst:
        f = os.path.join(a.frames, r["file"])
        im = Image.open(f).convert("RGB"); tw = 400
        im = im.resize((tw, int(im.height * tw / im.width)))
        sx = tw / Image.open(f).width
        d = ImageDraw.Draw(im)
        for bb, sc in te_boxes[f]:
            d.rectangle([bb[0]*sx, bb[1]*sx, (bb[0]+bb[2])*sx, (bb[1]+bb[3])*sx], outline=(0,255,0), width=3)
        for bb, sc in v0_boxes[f]:
            d.rectangle([bb[0]*sx, bb[1]*sx, (bb[0]+bb[2])*sx, (bb[1]+bb[3])*sx], outline=(255,0,0), width=2)
        d.text((5,5), f"{r['file'][:28]} T{r['n_teacher']} v0:{r['n_v0']} m{r['matched']}", fill=(255,255,0))
        tiles.append(im)
    if tiles:
        cols=5; tw_,th_=tiles[0].size; rows_=(len(tiles)+cols-1)//cols
        sheet=Image.new("RGB",(cols*tw_,rows_*th_),(20,20,20))
        for i,im in enumerate(tiles): sheet.paste(im,((i%cols)*tw_,(i//cols)*th_))
        sheet.save(out/"D_disagree.jpg", quality=88)
        print(f"== disagreement sheet: {out/'D_disagree.jpg'} (green=teacher, red=v0)")

if __name__ == "__main__":
    main()
