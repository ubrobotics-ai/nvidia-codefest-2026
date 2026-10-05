import json, os, glob
from huggingface_hub import snapshot_download
T = os.environ["TEAM"]
REV = "92bb89405242f204dd8d76be45877808bbdd98dd"   # the snapshot the worker reads from
ids = sorted(json.load(open(f"{T}/tmp/xfer/prompts_b2.json")))
d = snapshot_download("ubr-physical-ai/isaac-sdg-rescue-target", repo_type="dataset", revision=REV,
                      cache_dir=f"{T}/codefest/hf-cache/hub", allow_patterns=[f"clips/{c}/*" for c in ids], max_workers=16)
miss = [c for c in ids if len(glob.glob(f"{d}/clips/{c}/frame_*.jpg")) != 93]
print(f"  pulled {len(ids)} clips into {d}")
print(f"  clips without 93 frames: {miss or 'none'}")
