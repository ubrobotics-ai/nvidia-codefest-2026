# ONNX export of D-FINE-M (EMA weights) at the trained input, 1x3x704x1280 (H x W), batch 1,
# static shapes, postprocessor included (as upstream tools/deployment/export_onnx.py does).
import sys
sys.path.insert(0, "/storage/hackathon_teams/omc-team15/codefest/permissive/D-FINE")
import torch, torch.nn as nn
from src.core import YAMLConfig
cfg_p, ckpt, out = sys.argv[1:4]
cfg = YAMLConfig(cfg_p, resume=ckpt)
c = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg.model.load_state_dict(c["ema"]["module"])

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.model = cfg.model.deploy(); s.post = cfg.postprocessor.deploy()
    def forward(s, images, orig_target_sizes):
        return s.post(s.model(images), orig_target_sizes)

m = M().eval()
x, size = torch.rand(1, 3, 704, 1280), torch.tensor([[1280, 720]])
m(x, size)
torch.onnx.export(m, (x, size), out, input_names=["images", "orig_target_sizes"],
                  output_names=["labels", "boxes", "scores"], opset_version=17, do_constant_folding=True, dynamo=False)
import onnx; onnx.checker.check_model(onnx.load(out)); print("exported", out, flush=True)
