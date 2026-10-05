# RF-DETR-S trained natively at a smaller square input (argv[1]), for Jetson latency.
# Same recipe as RF-DETR-M@1280: 12 epochs, total batch 16, seed 1234, rebal split.
import sys
from rfdetr import RFDETRSmall
Z = int(sys.argv[1])
R = "/storage/hackathon_teams/omc-team15/codefest/permissive"
m = RFDETRSmall(resolution=Z)
m.train(dataset_dir=f"{R}/rfdata", output_dir=f"{R}/runs/rfdetr_s{Z}",
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=2, sync_bn=True)
