# Resume of train_mc_v2.py after the 12 h Slurm limit (about 80 min per epoch with the dense box renders).
import glob, re
from rfdetr import RFDETRSmall
R = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
ck = max(glob.glob(f"{R}/runs/mc_v2/checkpoint_[0-9]*.ckpt"), key=lambda p: int(re.findall(r"checkpoint_(\d+)", p)[0]))
print("resuming from", ck, flush=True)
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=f"{R}/data_v2", output_dir=f"{R}/runs/mc_v2", resume=ck,
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
