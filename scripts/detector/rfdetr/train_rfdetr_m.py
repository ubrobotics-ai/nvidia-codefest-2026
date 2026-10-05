# UBR rescue: RF-DETR-M (Apache-2.0 size), single class, held-out hospital split, 12 epochs.
# 4 GPUs DDP, 4 per GPU x 1 accumulation = total batch 16 (same as the 1-GPU 8 x 2).
# Square 1280x1280 input: RF-DETR only supports square resolutions (stretch-resize), and 1280 is
# divisible by patch_size * num_windows = 32. Starts from Roboflow's COCO-pretrained M weights.
from rfdetr import RFDETRMedium
m = RFDETRMedium(resolution=1280)
m.train(dataset_dir="/storage/hackathon_teams/omc-team15/codefest/permissive/rfdata",
        output_dir="/storage/hackathon_teams/omc-team15/codefest/permissive/runs/rfdetr_m",
        epochs=12, batch_size=4, grad_accum_steps=1, num_workers=12, seed=1234,
        strategy="ddp", devices=4, sync_bn=True)
