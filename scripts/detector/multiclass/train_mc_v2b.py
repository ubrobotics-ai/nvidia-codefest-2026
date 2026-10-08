# 10-class RF-DETR-S@640, v2b: v2's data plus the Isaac warehouse train frames back, with their boxes pseudo-labelled
# by v2 (score >= 0.5). Same recipe as v1/v2: 12 epochs, seed 1234. Resumes from the newest epoch checkpoint if any
# (one run does not fit the 12 h Slurm limit).
import glob, re
from rfdetr import RFDETRSmall
R = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
cks = glob.glob(f"{R}/runs/mc_v2b/checkpoint_[0-9]*.ckpt")
ck = max(cks, key=lambda p: int(re.findall(r"checkpoint_(\d+)", p)[0])) if cks else None
print("resuming from", ck, flush=True)
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=f"{R}/data_v2b", output_dir=f"{R}/runs/mc_v2b", **({"resume": ck} if ck else {}),
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
