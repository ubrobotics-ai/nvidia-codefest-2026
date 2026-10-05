# 10-class RF-DETR-S@640, baseline (v1): labels as built, no pre-labelling of the missing classes.
# Classes: person, person_lying, backpack, handbag, suitcase, dog, car, truck, box, object (catch-all).
# Same recipe as the single-class RF-DETR-S@640: 12 epochs, seed 1234, per-epoch checkpoints.
from rfdetr import RFDETRSmall
R = "/storage/hackathon_teams/omc-team15/codefest/multiclass"
m = RFDETRSmall(resolution=640)
m.train(dataset_dir=f"{R}/data", output_dir=f"{R}/runs/mc_v1",
        epochs=12, batch_size=8, grad_accum_steps=1, num_workers=12, seed=1234,
        checkpoint_interval=1, strategy="ddp", devices=4, sync_bn=True)
