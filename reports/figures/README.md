# Figures

## isaac_to_photoreal.png

Four bands, every panel a real artefact from the Cosmos Transfer 2.5 run on one B300.
Nothing is a mock-up and no frame was selected after seeing a detector score, except where
a caption states the score.

1. The pipeline on one frame: Isaac render, instance mask, translated output.
2. The same warehouse orbit rendered as five environments the simulator does not contain.
3. The four clips whose labels survive translation, with COCO detection on each.
4. The open limitation. At the prompt detail a striking scene needs, the subject is
   re-rendered at whatever scale reads best in frame. In the woodland panel that is 31x the
   mask area and 170 px away from it: photoreal, and mislabelled.

Bands 1 and 3 are usable training data. Bands 2 and 4 are not, yet.

## seven_repairs.png

Every lever the Transfer pipeline exposes, tried against the band-4 failure above. Seven arms,
one variable each, scored the same way on the same 16 sampled frames. The pass mark was fixed
before the runs: a person found in at least 8 of 16, within 50 px of the mask, at under 3x the
mask area.

None passed. The informative part is why the two that did put the person back on its label
still failed: edge control over the whole frame and depth control both hold the geometry by
holding the *building*, so the environment never changes, which is the entire reason to run
Transfer. Pinning geometry pins the scene. On this single-ControlNet pipeline that is a
property, not a tuning problem.

One thing remains untried: a mask dilated 2x and 3x, to test whether the failure is
scale-dependent. Indoors at 2,936 px the same prompt lands at 3 px and 1.0x.
