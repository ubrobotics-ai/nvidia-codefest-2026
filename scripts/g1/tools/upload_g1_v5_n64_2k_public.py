"""Publish the v5 N = 64, 2,000-iteration checkpoint PUBLICLY (Filipe 2026-10-09: "public"): weights only (iter_*/model), the
training config with local paths replaced by <WORKDIR>, the v5 action stats, the episode list and a model card marked
EVALUATION PENDING. The repo may already exist as private (created by a failed private upload); it is switched to public.
Usage: upload_g1_v5_n64_2k_public.py ITER_DIR REPO N"""
import json, os, re, shutil, sys
from huggingface_hub import HfApi

it_dir, rid, n = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3])
G = "/storage/hackathon_teams/omc-team15/codefest/g1_cosmos"
it = os.path.basename(it_dir)
eps = json.load(open(f"{it_dir}/episodes.json"))
assert len(eps) == n, (len(eps), n)
stage = f"{G}/upload_stage_{rid.split('/')[-1]}"
os.makedirs(stage, exist_ok=True)
if not os.path.isdir(f"{stage}/{it}/model"):
    shutil.copytree(f"{it_dir}/model", f"{stage}/{it}/model", copy_function=os.link)   # hard links: same filesystem
cfg = open(f"{os.path.dirname(os.path.dirname(it_dir))}/config.yaml").read()
cfg = re.sub(r"/storage/hackathon_teams/omc-team15/codefest/g1_cosmos", "<WORKDIR>", cfg)
assert "/storage/" not in cfg and "/home/" not in cfg
assert not re.search(r"hf_[A-Za-z0-9]{20,}|ghp_|github_pat_", cfg)
open(f"{stage}/config.yaml", "w").write(cfg)
shutil.copy(f"{G}/g1_data_v5/g1_hug_v5/action_stats.json", f"{stage}/action_stats.json")
shutil.copy(f"{it_dir}/episodes.json", f"{stage}/episodes.json")

assert n == 64
open(f"{stage}/README.md", "w").write(f"""---
license: other
license_name: openmdw1.1-license
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Cosmos3-Edge
tags: [robotics, unitree-g1, humanoid, action-policy, cosmos, simulation]
---
# G1 box lift: Cosmos3-Edge action policy, v5, 64 demonstrations, 2,000 iterations

EVALUATION PENDING: the closed-loop test (same protocol as the v5 curve) has not been run yet.

NVIDIA Cosmos3-Edge post-trained as a grasp-and-lift policy for a simulated Unitree G1 humanoid: it commands the
head, both wrists and the hands; the legs are driven by a separate locomotion controller. Trained on 64 of the 965
training episodes of our simulated demonstration set g1_hug_v5 (private; Isaac Lab 3; four support heights,
0.65 to 0.80 m), 16 per recording seed. The actions are the scripted controller's commanded wrist targets and hand closure.

- Input: head camera above both wrist cameras (one 640x540 frame), plus a text prompt with the box pose.
- Output: chunks of 16 actions at 10 Hz, 29 values each: head, right wrist and left wrist pose deltas
  (translation + 6-D rotation) and per-hand openness.
- Training: the same recipe as the 965-episode 2,000-iteration run: full fine-tune with cosmos-framework
  (action_policy_g1_hug_edge recipe), 2,000 iterations, learning rate 5e-5 with linear decay over 2,000, global batch
  2048, 4 x NVIDIA B300. The second compute budget of our data-efficiency curve (the first was 500 iterations for
  N = 64, 256, 965).

## Files
- `{it}/model/`: weights (PyTorch distributed checkpoint).
- `config.yaml`: the resolved training config (local paths replaced by <WORKDIR>); pass it with `--config-file`.
- `action_stats.json`: the v5 action normalisation statistics (all 1,016 episodes); required at inference.
- `episodes.json`: the {n} training episode indices.

## Limitations
Research checkpoint from a hackathon bench demo. Trained in simulation only; never run on a real robot.
Not suitable for operational use.

## License
OpenMDW 1.1, as the base model nvidia/Cosmos3-Edge.
""")

api = HfApi()
api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
if api.model_info(rid).private:
    api.update_repo_settings(rid, private=False)
assert not api.model_info(rid).private
api.upload_folder(repo_id=rid, repo_type="model", folder_path=stage,
                  allow_patterns=[f"{it}/model/**", "config.yaml", "action_stats.json", "episodes.json", "README.md"],
                  commit_message=f"v5 N={n}, 2,000 iterations: {it} weights, config, action stats, episode list")
info = api.model_info(rid)
print("UPLOADED", rid, info.sha, "private" if info.private else "public", len(api.list_repo_files(rid)), "files", flush=True)
