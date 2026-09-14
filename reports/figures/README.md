# Figures

## isaac_to_photoreal.png

Four bands, every panel a real artefact from the Cosmos Transfer 2.5 run on one B300.
Nothing is a mock-up and no frame was selected after seeing a detector score, except where
a caption states the score.

1. The pipeline on one frame: Isaac render, instance mask, translated output.
2. The same warehouse orbit under five prompts. Only ONE of the five keeps the casualty on
   its label (warehouse, 10/16 frames); the other four lose it.
3. The four clips whose labels survive translation, with COCO detection on each.
4. The open limitation. At the prompt detail a striking scene needs, the subject is
   re-rendered at whatever scale reads best in frame. In the woodland panel that is 31x the
   mask area and 170 px away from it: photoreal, and mislabelled.

Bands 1 and 3 are usable training data. In band 2 only the warehouse is.

Scoring note: a frame counts only if a person is found AND within 50 px of the mask AND
under 3x its area. 'Found anywhere' alone is not enough -- the failing baseline finds a
person in 10 of 16 frames and meets the full criterion in 0.

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

Neither strength saves the woodland prompt: at 0.7 the casualty is erased outright (0 of 16
frames find a person) and at 1.0 it is relocated. The two low-guidance arms managed 1 of 16
each, by deleting the casualty in the other 15.

One thing remains untried: a mask dilated 2x and 3x. That is NOT a test of scale -- the
warehouse prompt keeps the label on this same 2,205 px mask -- but of whether a bigger
control signal can outweigh the prompt.
