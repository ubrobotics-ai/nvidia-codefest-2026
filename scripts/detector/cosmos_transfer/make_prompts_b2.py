"""Prompts for Cosmos Transfer batch 2: every clip not yet translated, hospital excluded.

Hospital is the held-out evaluation domain for every detector, so none of its clips are
translated for training. Same prompt rules as make_prompts.py (name the person, environment
and lighting only, never the vehicle), plus the two non-lying poses of warehouse_poses.
"""
import json, os
T = os.environ["TEAM"]
SNAP = f"{T}/codefest/hf-cache/hub/datasets--ubr-physical-ai--isaac-sdg-rescue-target/snapshots/92bb89405242f204dd8d76be45877808bbdd98dd"
SCENE = {
 "hall":      "a large empty industrial hall with a high steel roof and bare concrete floor",
 "warehouse": "a warehouse interior with tall stacked wooden pallets and a concrete floor",
 "office":    "an open-plan office interior with desks and carpet tiles",
 "rivermark": "an outdoor city street beside low buildings, wet asphalt and a kerb",
}
LIGHT = {"fog": "filled with thick fog, diffuse grey light, low visibility",
         "night": "at night in near darkness, lit only by a single moving lamp, deep shadows",
         "plain": "in flat overcast daylight"}
POSE = {"supine": "lying on their back, motionless on the ground",
        "prone": "lying face down, motionless on the ground",
        "standing": "standing still on the ground",
        "kneeling": "kneeling on the ground"}
VEST = {"yellow": "wearing a yellow high-visibility vest", "orange": "wearing an orange high-visibility vest",
        "none": "wearing ordinary dark clothing"}
BANNED = ("robot", "rover", "tracked", "vehicle", "camera platform", "drone", "ugv", "wheelchair")
cond = lambda cid: next((k for k in ("fog", "night") if k in cid), "plain")

done = {c["clip_id"] for c in json.load(open(f"{T}/tmp/xfer/subset20.json"))}
out = {}
for l in open(f"{SNAP}/clips.jsonl"):
    c = json.loads(l); cid = c["clip_id"]
    if cid in done or c["environment"] == "hospital":
        continue
    p = (f"{SCENE[c['environment']]} {LIGHT[cond(cid)]}, a person {VEST[str(c['vest'])]} "
         f"{POSE[c['pose']]} about {c['distance_m']:.0f} metres away")
    assert not [b for b in BANNED if b in p.lower()], cid
    out[cid] = p
json.dump(out, open(f"{T}/tmp/xfer/prompts_b2.json", "w"), indent=1)
print(len(out), "clips")
for k in list(out)[:2] + [k for k in out if "poses" in k][:2]: print(" ", k, "|", out[k])
