#!/usr/bin/env python3
"""Task C: how close is each synthetic candidate set to the real reference set?

  python scripts/dataset_proxies.py --spec spec.json --out C_proxies.csv --cache <dir>

`spec.json` is a list of sets: {"name":..., "role":"reference"|"candidate", "images":[...paths...],
 "coco": "<coco.json or null>"}. CLIP ViT-L/14 image embeddings are cached per set as <cache>/<name>.npy.

Metrics per candidate, all against the reference:
  MMD_clip   unbiased MMD^2 with an RBF kernel on L2-normalised CLIP embeddings; bandwidth by the median heuristic
             computed once on the reference set so every candidate is scored on the same kernel.
  CCDM       stratified variant (Cross-Category Distribution Matching, Yang et al.): images are binned into strata by
             person-box scale bucket x object count; we report JS divergence between the two strata *compositions*
             plus the mean within-stratum MMD over strata populated by both sets, and combine them as
             CCDM = JS + mean_within_MMD. Lower is closer. Strata come from each set's COCO; a set without COCO gets
             MMD only.
  FD_CLIP    Frechet distance between Gaussians fitted to the CLIP features. This is NOT Inception-FID -- torchmetrics
             is not installed in this container -- so it is comparable across rows here and nowhere else.
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

def clip_embed(paths, model, proc, device, batch=64):
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch]]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(device)
            f = model.get_image_features(**inp)
        # transformers 5.x returns an output object here, not a bare tensor
        if not isinstance(f, torch.Tensor):
            for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
                v = getattr(f, attr, None)
                if v is not None:
                    f = v if v.ndim == 2 else v[:, 0]
                    break
            else:
                f = f[0]
        out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        if i % (batch * 10) == 0: print(f"    {i}/{len(paths)}", flush=True)
    return np.concatenate(out)

def median_bandwidth(X, cap=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(cap, len(X)), replace=False)
    A = X[idx]
    d2 = np.maximum(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1), 0)
    m = np.median(d2[np.triu_indices(len(A), k=1)])
    return float(m) if m > 0 else 1.0

def mmd2_rbf(X, Y, gamma):
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)
    m, n = len(X), len(Y)
    if m < 2 or n < 2: return float("nan")
    Kxx = k(X, X); Kyy = k(Y, Y); Kxy = k(X, Y)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
    return float(Kxx.sum()/(m*(m-1)) + Kyy.sum()/(n*(n-1)) - 2*Kxy.mean())

def frechet(X, Y):
    from scipy import linalg
    mx, my = X.mean(0), Y.mean(0)
    cx, cy = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    covmean, _ = linalg.sqrtm(cx.dot(cy), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(((mx-my)**2).sum() + np.trace(cx) + np.trace(cy) - 2*np.trace(covmean))

SCALE_EDGES = [0, 1024, 4096, 16384, 65536, np.inf]      # person-box area px^2
COUNT_EDGES = [0, 1, 2, 4, np.inf]                        # objects per image

def strata_from_coco(coco_path, names):
    """-> {basename: (scale_bucket, count_bucket)}; scale bucket from the largest person box in the image."""
    if not coco_path or not os.path.exists(coco_path): return {}
    c = json.load(open(coco_path))
    cats = {x["id"]: x["name"] for x in c.get("categories", [])}
    person_ids = {i for i, n in cats.items() if "person" in n.lower()} or set(cats)
    by_img = {im["id"]: os.path.basename(im["file_name"]) for im in c["images"]}
    areas, counts = {}, {}
    for an in c["annotations"]:
        b = by_img.get(an["image_id"])
        if b is None: continue
        counts[b] = counts.get(b, 0) + 1
        if an["category_id"] in person_ids:
            a = an.get("area") or (an["bbox"][2] * an["bbox"][3])
            areas[b] = max(areas.get(b, 0.0), float(a))
    out = {}
    for b in names:
        s = int(np.digitize(areas.get(b, 0.0), SCALE_EDGES) - 1)
        n = int(np.digitize(counts.get(b, 0), COUNT_EDGES) - 1)
        out[b] = (s, n)
    return out

def js_divergence(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--max-per-set", type=int, default=1200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = json.load(open(a.spec)); cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import CLIPModel, CLIPImageProcessor
    dev = "cuda"
    model = CLIPModel.from_pretrained(a.clip).to(dev).eval()
    proc = CLIPImageProcessor.from_pretrained(a.clip)

    rng = np.random.default_rng(a.seed)
    feats, strata, sizes = {}, {}, {}
    for s in spec:
        paths = sorted(s["images"])
        if len(paths) > a.max_per_set:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), a.max_per_set, replace=False))]
        npy = cache / f"{s['name']}.npy"
        if npy.exists():
            F = np.load(npy); print(f"  {s['name']}: cached {F.shape}", flush=True)
        else:
            print(f"  {s['name']}: embedding {len(paths)} images", flush=True)
            F = clip_embed(paths, model, proc, dev); np.save(npy, F)
        feats[s["name"]] = F; sizes[s["name"]] = len(paths)
        strata[s["name"]] = strata_from_coco(s.get("coco"), [os.path.basename(p) for p in paths])

    ref = next(s["name"] for s in spec if s["role"] == "reference")
    gamma = 1.0 / median_bandwidth(feats[ref])
    print(f"reference={ref} n={sizes[ref]} gamma={gamma:.4f}", flush=True)

    rows = []
    for s in spec:
        if s["role"] != "candidate": continue
        n = s["name"]
        mmd = mmd2_rbf(feats[ref], feats[n], gamma)
        fd = frechet(feats[ref], feats[n])
        ccdm, js, within = float("nan"), float("nan"), float("nan")
        if strata[ref] and strata[n]:
            rb = [os.path.basename(p) for p in sorted(s["images"])][: sizes[n]]
            keys = sorted(set(strata[ref].values()) | set(strata[n].values()))
            def comp(name, basenames):
                c = {k: 0 for k in keys}
                for b in basenames: 
                    k = strata[name].get(b)
                    if k in c: c[k] += 1
                return [c[k] for k in keys]
            ref_names = list(strata[ref].keys()); cand_names = list(strata[n].keys())
            js = js_divergence(comp(ref, ref_names), comp(n, cand_names))
            ws = []
            for k in keys:
                ri = [i for i, b in enumerate(ref_names) if strata[ref].get(b) == k]
                ci = [i for i, b in enumerate(cand_names) if strata[n].get(b) == k]
                if len(ri) >= 5 and len(ci) >= 5:
                    ws.append(mmd2_rbf(feats[ref][ri], feats[n][ci], gamma))
            within = float(np.mean(ws)) if ws else float("nan")
            ccdm = js + (within if ws else 0.0)
        rows.append({"candidate": n, "N": sizes[n], "N_ref": sizes[ref], "MMD_clip": round(mmd, 6),
                     "CCDM": round(ccdm, 6) if ccdm == ccdm else "", "CCDM_JS": round(js, 6) if js == js else "",
                     "CCDM_withinMMD": round(within, 6) if within == within else "",
                     "FD_CLIP": round(fd, 4), "strata_used": bool(strata[n])})
        print(f"  {n}: MMD={mmd:.6f} CCDM={ccdm:.6f} FD_CLIP={fd:.3f}", flush=True)

    rows.sort(key=lambda r: r["MMD_clip"])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}; closest to real by MMD: {rows[0]['candidate']} ({rows[0]['MMD_clip']})")

if __name__ == "__main__":
    main()
