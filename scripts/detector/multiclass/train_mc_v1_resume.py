# Resume of train_mc_v1.py after node dgx06 failed (Slurm NODE_FAIL at 11:14, job 9064).
# checkpoint_3.ckpt = end of epoch 4 of 12; same settings, same output dir.
from rfdetr import RFDETRSmall
R = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=f"{R}/data", output_dir=f"{R}/runs/mc_v1", resume=f"{R}/runs/mc_v1/checkpoint_3.ckpt",
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
