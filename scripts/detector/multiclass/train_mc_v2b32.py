# v2b-32: v2b's data (v2 + Isaac warehouse train frames with v2 pseudo-labelled boxes), 32 images per GPU instead of 8
# to cut the per-step overhead that leaves the B300s idle (v2b: ~80 min per epoch). Compensations for the 4x batch:
# lr and lr_encoder x2 (square-root rule), EMA decay 0.993 -> 0.993^4 = 0.972 (same averaging span in images).
# Everything else as v1/v2/v2b: 12 epochs, seed 1234. Resumes from the newest epoch checkpoint in OUT.
# Usage: train_mc_v2b32.py OUT [EPOCHS]
import glob, re, sys
from rfdetr import RFDETRSmall
R = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
OUT = sys.argv[1]; EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
cks = glob.glob(f"{OUT}/checkpoint_[0-9]*.ckpt")
ck = max(cks, key=lambda p: int(re.findall(r"checkpoint_(\d+)", p)[0])) if cks else None
print("resuming from", ck, flush=True)
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=f"{R}/data_v2b", output_dir=OUT, **({"resume": ck} if ck else {}),
        epochs=EPOCHS, batch_size=32, grad_accum_steps=1, num_workers=16, seed=1234,
        lr=2e-4, lr_encoder=3e-4, ema_decay=0.993 ** 4,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
