# Single-input ONNX export of D-FINE (EMA weights) for the robot's TensorRT runner, which accepts
# exactly one FP32 input: images [1,3,704,1280] (RGB, [0,1]). No postprocessor in the graph.
# Outputs: logits [1,300,2] (sigmoid -> scores; person is index 1) and boxes [1,300,4]
# (cxcywh, normalised to the input). Decode like RF-DETR: top-k over sigmoid(logits).
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
        super().__init__(); s.model = cfg.model.deploy()
    def forward(s, images):
        o = s.model(images)
        return o["pred_logits"], o["pred_boxes"]

m = M().eval()
x = torch.rand(1, 3, 704, 1280)
m(x)
torch.onnx.export(m, (x,), out, input_names=["images"], output_names=["logits", "boxes"],
                  opset_version=17, do_constant_folding=True, dynamo=False)
import onnx; onnx.checker.check_model(onnx.load(out)); print("exported", out, flush=True)
