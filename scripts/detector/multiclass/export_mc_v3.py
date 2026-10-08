# ONNX export of the 10-class RF-DETR-S@640 v3 (final EMA weights), batch 1, static, opset 17. Same as export_mc.py (v1).
from rfdetr import RFDETR
M = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
m = RFDETR.from_checkpoint(f"{M}/runs/mc_v3/last_ema.pth")
m.export(output_dir=f"{M}/export/mc_v3_640", shape=(640, 640), batch_size=1, opset_version=17)
import onnx
g = onnx.load(f"{M}/export/mc_v3_640/rfdetr-small.onnx").graph
print("IO", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in list(g.input) + list(g.output)], flush=True)
