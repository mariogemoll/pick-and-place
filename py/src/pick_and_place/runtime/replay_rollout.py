# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The ``PPRL`` episode format the browser episode viewer replays.

One file is one episode's per-frame ``qpos``: the 6 arm and gripper joint
angles in radians, followed by the cube's free-joint pose (position, then
MuJoCo's w,x,y,z quaternion). That is the whole state a viewer needs to redraw
the scene, which is why an episode costs a few kilobytes here and megabytes as
video -- and why it can be scrubbed, re-aimed and lit at replay time rather
than baked.

Binary layout, little-endian::

    magic    4 bytes   b"PPRL"
    version  u32       1
    fps      u32       the rate the frames were sampled at
    nframes  u32       number of frames
    nq       u32       floats per frame (6 joints + 7 cube pose = 13)
    target_x f32       drop target position on the floor (meters)
    target_y f32
    qpos     f32[nframes * nq]

``ts/src/visualizations/episode-replay/rollout.ts`` parses it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"PPRL"
VERSION = 1


def write_rollout(
    path: Path,
    qpos: np.ndarray,
    fps: float,
    target_xy: tuple[float, float],
) -> None:
    """Write one episode's per-frame ``qpos`` as a ``PPRL`` file."""
    frames = np.asarray(qpos, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"qpos must be (nframes, nq), got shape {frames.shape}")
    nframes, nq = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<IIII", VERSION, round(fps), nframes, nq))
        file.write(struct.pack("<ff", float(target_xy[0]), float(target_xy[1])))
        file.write(frames.tobytes())
