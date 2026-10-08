# 10-class RF-DETR-S@640, v3: v2b's data plus the Cosmos-Transfer-2.5 re-textured warehouse renders (build_v3.py).
# Recipe identical to v1 / v2 / v2b: 12 epochs, 8 images per GPU on 4 GPUs (global batch 32), default lr, seed 1234.
# Resumes from the newest epoch checkpoint in OUT. Usage: torchrun --nproc_per_node=4 train_mc_v3.py DATA OUT
import glob, re, sys
from rfdetr import RFDETRSmall
data, out = sys.argv[1], sys.argv[2]
cks = glob.glob(f"{out}/checkpoint_[0-9]*.ckpt")
ck = max(cks, key=lambda p: int(re.findall(r"checkpoint_(\d+)", p)[0])) if cks else None
print("resuming from", ck, flush=True)
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=data, output_dir=out, **({"resume": ck} if ck else {}),
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
