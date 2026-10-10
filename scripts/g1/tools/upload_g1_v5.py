"""Publish one v5 checkpoint (weights only) to a ubr-physical-ai model repo. The curve points (n64/n256/n965, 500 iterations)
went public (Filipe, 2026-10-05: "Publish as public storage"; card approved the same day); from 2026-10-07 new repos are
PRIVATE (Filipe: "keep this one private for now"). Usage: upload_g1_v5.py ITER_DIR REPO N ITERS CURVE(0|1) EPISODES_JSON"""
import json, os, shutil, sys, time
from huggingface_hub import HfApi
# Uploads on hold (2026-10-09, Filipe: visibility of v6-n971 and the v5 N = 64 2k run not decided): skip while this file exists.
if os.path.exists("/storage/hackathon_teams/omc-team15/codefest/g1_cosmos/HOLD_UPLOADS"):
    raise SystemExit("uploads on hold (HOLD_UPLOADS present); checkpoint stays on the cluster")

it_dir, rid, n, iters, curve, eps = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1", sys.argv[6]
G = "/storage/hackathon_teams/omc-team15/codefest/g1_cosmos"
it = os.path.basename(it_dir)
stage = f"{G}/upload_stage_{rid.split('/')[-1]}"
os.makedirs(stage, exist_ok=True)
if not os.path.isdir(f"{stage}/{it}/model"):
    # Hard-linked copy, not a symlink: the hub uploader does not follow symlinked directories. Same filesystem.
    shutil.copytree(f"{it_dir}/model", f"{stage}/{it}/model", copy_function=os.link)
shutil.copy(f"{os.path.dirname(os.path.dirname(it_dir))}/config.yaml", f"{stage}/config.yaml")
shutil.copy(f"{G}/g1_data_v5/g1_hug_v5/action_stats.json", f"{stage}/action_stats.json")
shutil.copy(eps, f"{stage}/episodes.json")
assert len(json.load(open(eps))) == int(n), (eps, n)

curve_line = ("  One point of a data-efficiency curve (N = 64, 256, 965; nested subsets, equal per recording seed).\n"
              if curve else "")
readme = f"""---
license: other
license_name: openmdw1.1-license
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Cosmos3-Edge
tags: [robotics, unitree-g1, humanoid, action-policy, cosmos, simulation]
---
# G1 box lift: Cosmos3-Edge action policy, v5, {n} demonstrations

NVIDIA Cosmos3-Edge post-trained as an action policy for a simulated Unitree G1 humanoid that walks up
to a box and lifts it with both arms. Trained on {n} of the 965 training episodes of our simulated
demonstration set ubr-physical-ai/g1_hug_v5 @ c6677869 (private; Isaac Lab 3; four support heights, 0.65 to 0.80 m). In v5 the actions are
the scripted controller's commanded wrist targets and hand closure.

- Input: head camera above both wrist cameras (one 640x540 frame), plus a text prompt with the box pose.
- Output: chunks of 16 actions at 10 Hz, 29 values each: head, right wrist and left wrist pose deltas
  (translation + 6-D rotation) and per-hand openness.
- Training: full fine-tune with cosmos-framework (action_policy_g1_hug_edge recipe), {iters} iterations,
  learning rate 5e-5 with linear decay, global batch 2048, 4 x NVIDIA B300.
{curve_line}
## Files
- `{it}/model/`: weights (PyTorch distributed checkpoint).
- `config.yaml`: the resolved training config; pass it to the policy server with `--config-file`.
- `action_stats.json`: the action normalisation statistics the model was trained with; required at inference.
- `episodes.json`: the training episode indices.

## Limitations
Research checkpoint from a hackathon bench demo. Trained and evaluated in simulation only; never run on
a real robot. Not suitable for operational use.

## License
OpenMDW 1.1, as the base model nvidia/Cosmos3-Edge.
"""
open(f"{stage}/README.md", "w").write(readme)

api = HfApi()
# 2026-10-07: Filipe, "keep this one private for now" (the 2,000-iteration runs). New repos are created PRIVATE; an
# existing public repo is left alone (its visibility is Filipe's call), so the upload stops instead.
if api.repo_exists(rid, repo_type="model") and not api.model_info(rid).private:
    raise SystemExit(f"{rid} exists and is public; not uploading (visibility is Filipe's call)")
api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
assert api.model_info(rid).private
for attempt in range(3):
    # One commit, at most 3 tries an hour apart (the hub's 128-commits-per-hour limit; see upload_g1_concat_v4.py).
    try:
        api.upload_folder(repo_id=rid, repo_type="model", folder_path=stage,
                          allow_patterns=[f"{it}/model/**", "config.yaml", "action_stats.json", "episodes.json", "README.md"],
                          commit_message=f"v5 N={n}: {it} weights, config, action stats, episode list")
        break
    except Exception as e:
        print(f"commit attempt {attempt + 1} failed: {str(e)[:200]}", flush=True)
        if "rate limit" not in str(e).lower() or attempt == 2:
            raise
        time.sleep(65 * 60)
info = api.model_info(rid)
print("UPLOADED", rid, info.sha, "private" if info.private else "public", len(api.list_repo_files(rid)), "files", flush=True)
