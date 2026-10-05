# 10-class RF-DETR-S@640, v2: v1's data plus the Isaac warehouse box renders (full_warehouse stage; boxes >= 12 px),
# minus the old Isaac warehouse frames whose boxes were unlabelled. Same recipe as v1: 12 epochs, seed 1234.
from rfdetr import RFDETRSmall
R = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=f"{R}/data_v2", output_dir=f"{R}/runs/mc_v2",
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
