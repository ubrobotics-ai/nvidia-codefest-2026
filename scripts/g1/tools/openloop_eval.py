"""Open-loop score of a G1 hug Cosmos3-Edge policy on held-out val episodes (no simulator).

For each val episode and each chunk start f (every STRIDE frames), ask the policy server for a 16-frame action chunk
from the RECORDED camera frame f, with the episode's own prompt, exactly as the closed-loop client does
(cosmos_policy.py: concat_view image 640x540, image_size 256, viewpoint, view description, domain unitree_g1_hug).
Compare with the recorded actions f..f+15:
  - per-frame errors per block: wrist translation delta (mm), wrist rotation delta (deg), openness (abs);
  - the chained wrist position after EXEC=8 and 16 frames, T_f . D_f..D_{f+k-1}, against the recorded wrist pose at
    f+k, in the BOX frame (README_eval section 4): x (away from the robot) and |y| (distance from the box's mid-plane;
    positive d|y| = wider than the demo).
Errors are reported for all chunks and by phase (v4: walk-up 0-69, settle 70-89, grab 90+).
Usage: openloop_eval.py URL DATA_ROOT OUT_JSON [STRIDE] [PHASE_OFFSET]
"""
import base64, json, math, sys, time, urllib.request
import cv2, numpy as np, torch
from cosmos_framework.data.generator.action.datasets.g1_hug_lerobot_dataset import G1HugLeRobotDataset

URL, ROOT, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 16
OFF = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # 90 for grab-only data (v3): its frame 0 is v4's frame 90
CONCAT_DESCRIPTION = (
    "The top row shows the head-mounted camera view looking down at the box in front of the robot. "
    "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
    "the left hand camera on the left and the right hand camera on the right.")
SIDES = {"right": (9, 18, "observation.states.right_wrist_pose"), "left": (19, 28, "observation.states.left_wrist_pose")}

ds = G1HugLeRobotDataset(root=ROOT, viewpoint="concat_view", action_stats_path=f"{ROOT}/action_stats.json", split="val")
import pyarrow.parquet as pq, glob
cols = ["episode_index", "frame_index", "task_index", "action", "observation.states.right_wrist_pose",
        "observation.states.left_wrist_pose", "observation.states.box_pose", "observation.states.hand_open"]
tab = {c: [] for c in cols}
for p in sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet")):
    t = pq.read_table(p, columns=cols)
    for c in cols:
        tab[c] += t[c].to_pylist()
ep_rows = {}
for i, e in enumerate(tab["episode_index"]):
    ep_rows.setdefault(int(e), []).append(i)
val_eps = [int(e) for e in ds._ep_vals]


def rot6d(r6):
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8); b2 = a2 - b1 @ a2 * b1; b2 /= max(np.linalg.norm(b2), 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], 1)


def quat_m(q):  # x, y, z, w
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_of(pose):
    T = np.eye(4); T[:3, :3] = quat_m(pose[3:7]); T[:3, 3] = pose[:3]; return T


def D_of(a9):
    D = np.eye(4); D[:3, :3] = rot6d(np.asarray(a9[3:9])); D[:3, 3] = a9[:3]; return D


def ang(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


def png_b64(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); return base64.b64encode(buf.tobytes()).decode()


jobs = []
for e in val_eps:
    rows = ep_rows[e]; n = len(rows)
    for f in range(0, n - 16, STRIDE):
        jobs.append((e, f))
print(f"{len(val_eps)} val episodes, {len(jobs)} chunks", flush=True)

recs = []; raw = []; t0 = time.time()
for b in range(0, len(jobs), 8):
    batch = jobs[b:b + 8]; items = []
    for e, f in batch:
        rows = ep_rows[e]; ts = [float(f) / 10.0]
        vid = ds._load_video(ds._episodes[e], ts)                       # (1, 3, 540, 640) in [0, 1]
        img = (vid[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        items.append({"image": png_b64(img), "prompt": ds._tasks[int(tab["task_index"][rows[0]])], "domain_name": "unitree_g1_hug",
                      "image_size": 256, "viewpoint": "concat_view", "additional_view_description": CONCAT_DESCRIPTION})
    req = urllib.request.Request(URL + "/predict_batch", data=json.dumps({"items": items}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        acts = np.asarray(json.loads(r.read())["actions"], dtype=np.float64)[..., :29]
    for (e, f), a in zip(batch, acts):
        rows = ep_rows[e]
        gt = np.asarray([tab["action"][rows[f + k]] for k in range(16)])
        box = T_of(np.asarray(tab["observation.states.box_pose"][rows[f]])); Rb, pb = box[:3, :3], box[:3, 3]
        r = {"ep": e, "f": f}
        for side, (i0, io, key) in SIDES.items():
            r[f"{side}_trans_mm"] = float(np.mean(np.linalg.norm(a[:, i0:i0 + 3] - gt[:, i0:i0 + 3], axis=1)) * 1000)
            r[f"{side}_rot_deg"] = float(np.mean([ang(rot6d(a[k, i0 + 3:i0 + 9]), rot6d(gt[k, i0 + 3:i0 + 9])) for k in range(16)]))
            r[f"{side}_open_abs"] = float(np.mean(np.abs(a[:, io] - gt[:, io])))
            r[f"{side}_open_signed"] = float(np.mean(a[:, io] - gt[:, io]))   # > 0: hand more open than the demo
            T = T_of(np.asarray(tab[key][rows[f]]))
            for k in range(16):
                T = T @ D_of(a[k, i0:i0 + 9])
                if k + 1 in (8, 16):
                    rec = np.asarray(tab[key][rows[f + k + 1]])[:3] if f + k + 1 < len(rows) else None
                    if rec is None:
                        continue
                    pp, pr = Rb.T @ (T[:3, 3] - pb), Rb.T @ (rec - pb)       # box frame
                    r[f"{side}_k{k + 1}_pos_mm"] = float(np.linalg.norm(T[:3, 3] - rec) * 1000)
                    r[f"{side}_k{k + 1}_dx_mm"] = float((pp[0] - pr[0]) * 1000)
                    r[f"{side}_k{k + 1}_dabsy_mm"] = float((abs(pp[1]) - abs(pr[1])) * 1000)
        r["pred_open_mean"] = float(np.mean(0.5 * (a[:, 18] + a[:, 28])))
        r["demo_open_mean"] = float(np.mean(0.5 * (gt[:, 18] + gt[:, 28])))
        raw.append((e, f, a.astype(np.float32), gt.astype(np.float32)))
        recs.append(r)
    print(f"{min(b + 8, len(jobs))}/{len(jobs)} chunks, {time.time() - t0:.0f} s", flush=True)

json.dump(recs, open(OUT, "w"))
np.savez_compressed(OUT.replace(".json", "_raw.npz"), ep=np.array([x[0] for x in raw]), f=np.array([x[1] for x in raw]),
                    pred=np.stack([x[2] for x in raw]), demo=np.stack([x[3] for x in raw]))


def summary(sel, name):
    if not sel:
        return
    keys = [k for k in sel[0] if k not in ("ep", "f")]
    print(f"--- {name}: {len(sel)} chunks")
    for k in keys:
        v = [r[k] for r in sel if k in r]
        if v:
            print(f"  {k:<22} median {np.median(v):8.2f}   mean {np.mean(v):8.2f}")


summary(recs, "all")
summary([r for r in recs if r["f"] + OFF < 70], "walk-up (f < 70)")
summary([r for r in recs if 70 <= r["f"] + OFF < 90], "settle (70 <= f < 90)")
summary([r for r in recs if 90 <= r["f"] + OFF < 160], "grab: reach and clamp (90 <= f < 160)")
summary([r for r in recs if r["f"] + OFF >= 160], "grab: lift and hold (f >= 160)")
print("OPENLOOPDONE")
