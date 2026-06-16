# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turn recorded ``.npz`` episodes into a flat ``(observation, action)`` table.

Behavior cloning is plain supervised regression: every frame of every demo is one
independent ``(obs, action)`` training pair. We stack them all into two arrays and
hand them to the trainer — there is no sequence structure at rung 1 (that arrives
with action chunking in rung 2).

Observations are rebuilt frame-by-frame with :func:`build_observation` from the
logged ``state`` (joint positions) and ``qpos`` (current cube pose), so the table
is constructed by the *same* code path the live rollout uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pick_and_place.il.observations import (
    ACT_DIM,
    OBS_DIM,
    build_observation,
    cube_pose_from_qpos,
)


def episode_to_pairs(record) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(obs[T,15], act[T,6])`` from one loaded episode ``record``."""
    state = record["state"]
    qpos = record["qpos"]
    qvel = record["qvel"]
    action = record["action"]
    target = record["cube_target"]
    obs = np.stack(
        [
            build_observation(state[t], qvel[t][:6], cube_pose_from_qpos(qpos[t]), target)
            for t in range(len(state))
        ]
    )
    return obs.astype(np.float32), action.astype(np.float32)


def load_dataset(
    episodes_dir: Path, *, successful_only: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Load every ``episode_*.npz`` under ``episodes_dir`` into one flat table.

    ``successful_only`` (the default) drops episodes whose recorded run missed the
    target — a clean expert is the whole premise of this setup, so a stray miss is
    noise, not signal.
    """
    paths = sorted(episodes_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz under {episodes_dir}")

    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    used = 0
    for path in paths:
        record = np.load(path, allow_pickle=True)
        if successful_only and not bool(record["success"]):
            continue
        obs, act = episode_to_pairs(record)
        obs_chunks.append(obs)
        act_chunks.append(act)
        used += 1

    if not obs_chunks:
        raise ValueError(f"no usable episodes in {episodes_dir} (successful_only={successful_only})")

    obs = np.concatenate(obs_chunks, axis=0)
    act = np.concatenate(act_chunks, axis=0)
    assert obs.shape[1] == OBS_DIM and act.shape[1] == ACT_DIM
    print(f"Loaded {used}/{len(paths)} episodes -> {len(obs)} (obs, action) pairs")
    return obs, act
