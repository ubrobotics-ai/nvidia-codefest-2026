# ONNX export of RF-DETR-S final EMA weights at the native training size, batch 1, static.
from rfdetr import RFDETR
R = "/storage/hackathon_teams/omc-team15/codefest/permissive"
for z in (640, 768):
    m = RFDETR.from_checkpoint(f"{R}/runs/rfdetr_s{z}/last_ema.pth")
    m.export(output_dir=f"{R}/export/rfdetr_s{z}_ep12", shape=(z, z), batch_size=1, opset_version=17)
    print("exported", z, flush=True)
