"""INT8 (ModelOpt Q/DQ) version of RF-DETR-M epoch 12, plus an FP16 baseline, both run through
TensorRT and written as harness JSONL so the recall cost of INT8 is measured, not assumed.

Usage: int8_pipeline.py SIZE   (square input: 960 or 1280)

Steps: calibration tensor from 256 train frames -> modelopt INT8 Q/DQ ONNX -> TensorRT engines
(FP16 from the FP32 ONNX; INT8+FP16 from the Q/DQ ONNX) -> inference over the 22,060 held-out
hospital val frames -> preds/rfdetr_m_ep12_<SIZE>_trt{fp16,int8}.jsonl.
Calibration uses Isaac train renders only (no L1 footage)."""
import json, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tensorrt as trt

Z = int(sys.argv[1])
R = "/storage/hackathon_teams/omc-team15/codefest/permissive"
DATA = "/storage/hackathon_teams/omc-team15/codefest/ubr-det/data/coco_rebal"
ONNX = f"{R}/export/rfdetr_m_ep12_{Z}/rfdetr-medium.onnx"
OUT = f"{R}/export_int8/{Z}"
os.makedirs(OUT, exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def prep(path):
    """Same as RFDETR.predict: [0,1] tensor, bilinear resize without antialias, ImageNet norm."""
    im = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t[None], size=(Z, Z), mode="bilinear", align_corners=False, antialias=False)[0]
    return ((t - MEAN) / STD).numpy(), im.width, im.height


def log(*a):
    print(f"[{Z} {time.strftime('%H:%M:%S')}]", *a, flush=True)


# 1. calibration tensor
calib = f"{OUT}/calib_{Z}.npy"
if not os.path.exists(calib):
    names = [im["file_name"] for im in json.load(open(f"{DATA}/train.json"))["images"]]
    random.Random(1234).shuffle(names)
    arr = np.stack([prep(f"{DATA}/train/{n}")[0] for n in names[:256]]).astype(np.float32)
    np.save(calib, arr)
    log("calib", arr.shape)

# 2. modelopt INT8 Q/DQ ONNX
qdq = f"{OUT}/rfdetr_m_ep12_{Z}_int8.onnx"
if not os.path.exists(qdq):
    cmd = [sys.executable, "-m", "modelopt.onnx.quantization", "--onnx_path", ONNX,
           "--quantize_mode", "int8", "--calibration_data", calib, "--calibration_method", "entropy",
           "--calibration_eps", "cuda:0", "cpu", "--output_path", qdq]
    log("quantize:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    log("qdq onnx written")

# 3. TensorRT engines
TL = trt.Logger(trt.Logger.WARNING)


def build(onnx_path, engine_path, int8):
    if os.path.exists(engine_path):
        return
    b = trt.Builder(TL)
    net = b.create_network(0)
    p = trt.OnnxParser(net, TL)
    if not p.parse_from_file(onnx_path):
        raise RuntimeError([str(p.get_error(i)) for i in range(p.num_errors)])
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
    t0 = time.time()
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError(f"engine build failed: {engine_path}")
    open(engine_path, "wb").write(ser)
    log("built", os.path.basename(engine_path), f"{os.path.getsize(engine_path)/1e6:.1f} MB", f"{time.time()-t0:.0f}s")


engines = {"trtfp16": (ONNX, f"{OUT}/fp16.engine", False), "trtint8": (qdq, f"{OUT}/int8.engine", True)}
for tag, (o, e, i8) in engines.items():
    build(o, e, i8)


# 4. harness inference
class Ds(torch.utils.data.Dataset):
    def __init__(self, names): self.names = names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        x, w, h = prep(f"{DATA}/val/{self.names[i]}")
        return torch.from_numpy(x), w, h, self.names[i]


val = [im["file_name"] for im in json.load(open(f"{DATA}/val.json"))["images"]]
rt = trt.Runtime(TL)
for tag, (_, eng_path, _) in engines.items():
    out_path = f"{R}/preds/rfdetr_m_ep12_{Z}_{tag}.jsonl"
    eng = rt.deserialize_cuda_engine(open(eng_path, "rb").read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    inp = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    bufs = {n: torch.empty(tuple(eng.get_tensor_shape(n)), dtype=torch.float32, device="cuda") for n in names}
    for n in names:
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    log(tag, "io", {n: tuple(bufs[n].shape) for n in names})
    stream = torch.cuda.Stream()
    dl = torch.utils.data.DataLoader(Ds(val), batch_size=None, num_workers=12)
    with open(out_path, "w") as f:
        for k, (x, w, h, n) in enumerate(dl):
            bufs[inp].copy_(x[None])
            with torch.cuda.stream(stream):
                ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            dets, logits = bufs["dets"][0], bufs["labels"][0]           # [Q,4] cxcywh norm, [Q,C] logits
            prob = logits.sigmoid().flatten()
            order = torch.argsort(prob, descending=True, stable=True)[:300]
            sc, q, c = prob[order], order // logits.shape[1], order % logits.shape[1]
            cx, cy, bw, bh = dets[q].unbind(1)
            x1 = ((cx - bw / 2) * w).clamp(0, w); y1 = ((cy - bh / 2) * h).clamp(0, h)
            x2 = ((cx + bw / 2) * w).clamp(0, w); y2 = ((cy + bh / 2) * h).clamp(0, h)
            keep = sc >= 0.005
            rows = torch.stack([sc, c.float(), x1, y1, x2 - x1, y2 - y1], 1)[keep].cpu().tolist()
            f.write(json.dumps({"file_name": n, "boxes": [
                {"conf": round(r[0], 4), "cls": int(r[1]), "bbox": [round(v, 1) for v in r[2:]]} for r in rows]}) + "\n")
            if k % 5000 == 0:
                log(tag, k, len(val))
    log(tag, "done ->", out_path)
