"""Publish a v6 checkpoint PUBLICLY (Filipe 2026-10-09: v6-n64 and v6-n971 "public"): weights only (iter_*/model), the
training config with local paths replaced by <WORKDIR>, the v6 action stats, the episode list and a model card marked
NOT YET EVALUATED. The repo may already exist as private (created by a failed private upload); it is switched to public.
Usage: upload_g1_v6_public.py ITER_DIR REPO N"""
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
shutil.copy(f"{G}/g1_data_v6/g1_hug_v6/action_stats.json", f"{stage}/action_stats.json")
shutil.copy(f"{it_dir}/episodes.json", f"{stage}/episodes.json")

if n == 64:
    data = ("Trained on 64 of the\n971 training episodes of our simulated demonstration set g1_hug_v6 (private; Isaac Lab 3): 48 in a\n"
            "search-and-rescue warehouse scene and 16 in a plain scene, 12 + 4 per support height (0.65 to 0.80 m).")
else:
    data = ("Trained on all 971\ntraining episodes of our simulated demonstration set g1_hug_v6 (private; Isaac Lab 3): 724 in a\n"
            "search-and-rescue warehouse scene and 247 in a plain scene, at four support heights (0.65 to 0.80 m).")
open(f"{stage}/README.md", "w").write(f"""---
license: other
license_name: openmdw1.1-license
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Cosmos3-Edge
tags: [robotics, unitree-g1, humanoid, action-policy, cosmos, simulation]
---
# G1 box lift: Cosmos3-Edge action policy, v6 (warehouse), {n} demonstrations

NOT YET EVALUATED: the closed-loop test, with and without the warehouse scene, is pending.

NVIDIA Cosmos3-Edge post-trained as a grasp-and-lift policy for a simulated Unitree G1 humanoid: it commands the
head, both wrists and the hands; the legs are driven by a separate locomotion controller. {data}
The actions are the scripted controller's commanded wrist targets; the hands stay open in v6, so the two openness
values are always 1.0.

- Input: head camera above both wrist cameras (one 640x540 frame), plus a text prompt with the box pose.
- Output: chunks of 16 actions at 10 Hz, 29 values each: head, right wrist and left wrist pose deltas
  (translation + 6-D rotation) and per-hand openness.
- Training: full fine-tune with cosmos-framework (action_policy_g1_hug_edge recipe), 500 iterations,
  learning rate 5e-5 with linear decay, global batch 2048, 4 x NVIDIA B300.

## Files
- `{it}/model/`: weights (PyTorch distributed checkpoint).
- `config.yaml`: the resolved training config (local paths replaced by <WORKDIR>); pass it with `--config-file`.
- `action_stats.json`: the v6 action normalisation statistics; required at inference.
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
                  commit_message=f"v6 N={n}: {it} weights, config, action stats, episode list")
info = api.model_info(rid)
print("UPLOADED", rid, info.sha, "private" if info.private else "public", len(api.list_repo_files(rid)), "files", flush=True)
