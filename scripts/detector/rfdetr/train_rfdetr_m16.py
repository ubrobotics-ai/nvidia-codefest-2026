# Extend the 12-epoch RF-DETR-M run to 16. LR is constant (step drop at epoch 100), so resuming
# equals a fresh 16-epoch run. checkpoint_interval=1 keeps every epoch's weights for harness scoring.
from rfdetr import RFDETRMedium
R = "/storage/hackathon_teams/omc-team15/codefest/permissive"
m = RFDETRMedium(resolution=1280)
m.train(dataset_dir=f"{R}/rfdata", output_dir=f"{R}/runs/rfdetr_m16",
        resume=f"{R}/runs/rfdetr_m16/resume_from_ep11.ckpt",
        epochs=16, batch_size=4, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
