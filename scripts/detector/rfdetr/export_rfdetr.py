# ONNX export of RF-DETR-M epoch 12 (EMA) at several square inputs, batch 1, for TensorRT on Jetson.
import sys
from rfdetr import RFDETR
R = "/storage/hackathon_teams/omc-team15/codefest/permissive"
for sz in (1280, 960, 640):
    m = RFDETR.from_checkpoint(f"{R}/runs/rfdetr_m/last_ema.pth")
    m.export(output_dir=f"{R}/export/rfdetr_m_ep12_{sz}", shape=(sz, sz), batch_size=1, opset_version=17)
    print("exported", sz, flush=True)
