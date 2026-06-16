# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The observation/action layout shared by training and rollout.

The single source of truth for what a policy sees and emits, so the dataset built
offline (:mod:`pick_and_place.il.dataset`) and the live rollout
(:mod:`pick_and_place.il.rollout`) can never drift apart — a mismatch there is the
quietest, nastiest IL bug.

Observation (21-dim, all in the sim frame):

* ``[0:6]``   measured joint positions, ``JOINT_NAMES`` order (radians)
* ``[6:12]``  measured joint velocities, same order (rad/s)
* ``[12:17]`` current cube pose ``(x, y, z, cos yaw, sin yaw)`` — privileged state
* ``[17:21]`` target cube pose ``(x, y, cos yaw, sin yaw)`` — z is fixed (floor)

Yaw enters as ``(cos, sin)`` so the wrap-around at ±π is invisible to the network.
The cube pose is the *current* per-frame pose, not the episode's start pose: that
is what lets a closed-loop policy tell "cube on the floor" from "cube in the jaws"
from "cube placed", and is the gotcha-#3 fix (joint angles alone cannot see the
cube).

Joint *velocities* are in the observation because position alone is ambiguous: the
arm passes the same configuration on the way out (moving fast), on the retreat,
and at the random start/end (stationary), each demanding a different set point.
Without velocity an MSE-regressed policy blends those into a mush and the arm
decelerates and stalls before the grasp — the failure that sinks position-only BC
here.

Actions are the 6 joint set points the analytic planner commands
(``JOINT_NAMES`` order, radians), identical to the dataset ``action`` field and to
what a real follower is sent at 50 Hz.
"""

from __future__ import annotations

import numpy as np

# Cube free joint within qpos: pos is qpos[6:9], quat (w,x,y,z) qpos[9:13].
CUBE_QPOS_SLICE = slice(6, 13)

OBS_DIM = 21
ACT_DIM = 6


def _yaw_cos_sin(yaw: float) -> tuple[float, float]:
    return float(np.cos(yaw)), float(np.sin(yaw))


def build_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    cube_pose: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray:
    """Assemble one 21-d observation.

    ``joint_pos``/``joint_vel`` are the 6 measured joint positions and velocities
    (``JOINT_NAMES`` order). ``cube_pose`` is ``(x, y, z, yaw)`` of the cube *now*;
    ``target_pose`` is ``(x, y, z, yaw)`` of the goal (z ignored). Yaws are
    expanded to cos/sin.
    """
    cx, cy, cz, cyaw = (float(v) for v in cube_pose[:4])
    tx, ty, _tz, tyaw = (float(v) for v in target_pose[:4])
    cyaw_c, cyaw_s = _yaw_cos_sin(cyaw)
    tyaw_c, tyaw_s = _yaw_cos_sin(tyaw)
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0:6] = np.asarray(joint_pos, dtype=np.float32)
    obs[6:12] = np.asarray(joint_vel, dtype=np.float32)
    obs[12:17] = (cx, cy, cz, cyaw_c, cyaw_s)
    obs[17:21] = (tx, ty, tyaw_c, tyaw_s)
    return obs


def cube_pose_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """Extract ``(x, y, z, yaw)`` of the free-joint cube from a full ``qpos`` row."""
    from pick_and_place.episodes import quat_yaw

    pos = qpos[6:9]
    yaw = quat_yaw(qpos[9:13])
    return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float64)
