# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unitree G1 two-arm box-lift ("hug") LeRobot dataset.

Simulated G1 29-DoF + Dex3 demos recorded in Isaac Lab and converted to LeRobot v3. The
stored ``action`` column already holds the 29-D head + dual-arm contract used by AgiBot
(``agibotworld``, domain 15):

    [head_pos+rot6d(9), right_pos+rot6d(9), right_open(1), left_pos+rot6d(9), left_open(1)]

where each 9-D block is the backward framewise delta ``T_t^-1 T_{t+1}`` of the head camera
or wrist pose (``pose_abs_to_rel``, rot6d = first two rotation-matrix columns) and the
openness is absolute (1 = open). The loader therefore slices the stored per-frame actions
directly, like ``LIBEROLeRobotDataset``, with a compact column index instead of row dicts.

Views: ``concat_view`` (default) puts the head camera on top and the two wrist cameras,
each resized to half the head size, side by side below it (AgiBot layout). The real G1
has only the head camera, so ``ego_view`` (head only) is the deployable setting.

Split: the converter writes ``info.json["splits"]`` (``"train": "a:b"``, ``"val": "b:c"``
episode-index ranges); ``split="full"`` keeps every episode.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.agibotworld_beta_lerobot_dataset import (
    _compute_idle_frames_agibot,
)
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.utils import log

Viewpoint = Literal["concat_view", "ego_view"]

_HEAD_KEY = "observation.images.head"
_HAND_LEFT_KEY = "observation.images.hand_left"
_HAND_RIGHT_KEY = "observation.images.hand_right"
_ACTION_KEY = "action"

_NORMALIZERS_DIR = Path(__file__).parent.parent / "normalizer_stats"
_DEFAULT_STATS = _NORMALIZERS_DIR / "g1_hug_lerobot_stats.json"


class G1HugLeRobotDataset(ActionBaseDataset):
    """G1 hug demos with the 29-D AgiBot-layout framewise-delta action."""

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 16,
        mode: str = "wam",
        pose_convention: str = "backward_framewise",
        tolerance_s: float = 3e-4,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "quantile",
        action_stats_path: str | None = None,
        split: str = "train",
        sample_stride: int = 1,
    ) -> None:
        if viewpoint not in ("concat_view", "ego_view"):
            raise NotImplementedError("Supported viewpoints are concat_view and ego_view.")
        super().__init__(
            root=root,
            domain_name="unitree_g1_hug",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        info_fps = self._info.get("fps")
        if info_fps and float(info_fps) != self._fps:
            log.info(f"G1 hug: using the dataset's native fps={info_fps} (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps
        self._stats_file = Path(action_stats_path) if action_stats_path else _DEFAULT_STATS
        if action_normalization is not None and not self._stats_file.exists():
            raise FileNotFoundError(f"G1 hug action stats not found at {self._stats_file}.")

        index_parts, episode_parts, task_parts, ts_parts, action_parts = [], [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", _ACTION_KEY])
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            action_parts.append(np.asarray(table[_ACTION_KEY].to_pylist(), dtype=np.float32))
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_action = np.concatenate(action_parts, axis=0).astype(np.float32)[order]
        if self._row_action.shape[1] != self.action_dim:
            raise ValueError(f"Expected a {self.action_dim}-D action column, got {self._row_action.shape[1]}.")

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        keep = self._split_episode_ids(ep_vals.tolist(), split)
        # Data-efficiency runs: G1_HUG_EPISODES names a JSON list of episode indices; the train split keeps only those.
        sub = os.environ.get("G1_HUG_EPISODES", "")
        if sub and split.lower().strip() == "train":
            keep &= set(int(v) for v in json.load(open(sub)))
        kept =np.array([int(v) in keep for v in ep_vals], dtype=bool)
        self._ep_vals = ep_vals.astype(np.int64)[kept]
        self._ep_starts = ep_starts.astype(np.int64)[kept]
        # Windows of chunk_length + 1 frames inside one episode.
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts.astype(np.int64)[kept] - self._chunk_length)).astype(
            np.int64
        )
        log.info(
            f"Loaded G1 hug dataset root={self._root} split={split!r} viewpoint={viewpoint!r} fps={self._fps} "
            f"kept_episodes={len(self._ep_vals)}/{len(ep_vals)} valid_indices={len(self)}"
        )

    @property
    def action_dim(self) -> int:
        return 29

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(
            Pos(prefix="head"),
            Rot("rot6d", prefix="head"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
        )

    @classmethod
    def _stats_path(cls) -> Path:
        return _DEFAULT_STATS

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            raw = json.loads(self._stats_file.read_text())
            self._norm_stats = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in raw.items()
                if k in ("mean", "std", "min", "max", "q01", "q99")
            }
        return self._norm_stats

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        return _compute_idle_frames_agibot(action)

    def _split_episode_ids(self, ep_ids: list[int], split: str) -> set[int]:
        split = split.lower().strip()
        if split == "full":
            return set(int(v) for v in ep_ids)
        ranges = self._info.get("splits", {})
        name = "train" if split == "train" else "val"
        if name not in ranges:
            raise ValueError(f"info.json has no {name!r} split; use split='full'.")
        lo, hi = (int(x) for x in str(ranges[name]).split(":"))
        return set(int(v) for v in ep_ids if lo <= int(v) < hi)

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for ``ActionIterableShuffleDataset``."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in self._valid_cum.tolist():
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self)
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 — skip past undecodable frames
                last_err = e
                log.warning(f"G1 hug: sample idx={idx} failed to load ({type(e).__name__}: {e}); resampling")
                if n > 0:
                    idx = random.randint(0, n - 1)
        raise RuntimeError(f"G1 hug: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        ep = int(np.searchsorted(self._valid_cum, int(idx), side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (int(idx) - prev)
        episode = self._episodes[int(self._ep_vals[ep])]

        timestamps = self._row_timestamp[start : start + self._chunk_length + 1].tolist()
        video = self._load_video(episode, timestamps)
        action = torch.from_numpy(np.ascontiguousarray(self._row_action[start : start + self._chunk_length])).float()

        extras: dict[str, Any] = {}
        if self._viewpoint == "concat_view":
            extras["additional_view_description"] = (
                "The top row shows the head-mounted camera view looking down at the box in front of the robot. "
                "The bottom row contains two horizontally concatenated wrist-mounted camera views: "
                "the left hand camera on the left and the right hand camera on the right."
            )
        task = self._tasks[int(self._row_task[start])]
        return self._build_result(mode=mode, video=video, action=action, ai_caption=task, **extras)

    def _load_video_key(self, episode: dict[str, Any], timestamps: list[float], key: str) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
        return decode_video_frames(self._video_path(episode, key), [from_ts + ts for ts in timestamps], self._tolerance_s)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        top = self._load_video_key(episode, timestamps, _HEAD_KEY)  # [T, C, H, W] in [0, 1]
        if self._viewpoint == "ego_view":
            return top
        _, _, h_top, w_top = top.shape
        size = (h_top // 2, w_top // 2)
        left = F.interpolate(self._load_video_key(episode, timestamps, _HAND_LEFT_KEY), size=size, mode="bilinear")
        right = F.interpolate(self._load_video_key(episode, timestamps, _HAND_RIGHT_KEY), size=size, mode="bilinear")
        return torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
