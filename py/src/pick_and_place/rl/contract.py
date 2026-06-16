# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The RL observation/action contract — the single source of truth.

The curriculum (position -> orientation -> regrasp -> robustness) is one
continuous finetune chain, and an observation *dimension* change mid-chain forces
a from-scratch retrain (the SB3 ``MlpPolicy`` input layer and
``VecNormalize.obs_rms`` are sized to ``OBS_DIM``). So the layout below is
*frozen*: every chapter is a reward/weight change on this same schema, never a new
dimension. See ``docs/rl-curriculum-roadmap.md`` for the full rationale.

Observation — frozen 31-dim (sim frame)
---------------------------------------
A **pose** is a uniform 9-vector: position ``(x, y, z)`` + a **6D rotation** (the
first two columns of the 3x3 rotation matrix; recover R by Gram-Schmidt, see
:func:`rotation_from_6d`). 6D is *continuous* — no wrap / double-cover
discontinuity — so it is a safe, downside-free choice for a rotation *input*.

* ``[0:6]``   measured joint positions, ``JOINT_NAMES`` order (rad) — proprioception
* ``[6:12]``  measured joint velocities, same order (rad/s)
* ``[12:21]`` **current** cube pose (position + 6D rotation) — a fused state estimate
* ``[21:30]`` **target** cube pose (position + 6D rotation) — the full goal pose
* ``[30]``    current-cube-pose confidence in ``[0, 1]`` — constant ``1.0`` in
  pure-sim chapters; driven by the estimator on real hardware

The full 6-DOF pose is carried (not just ground-cube ``x, y, yaw``) as deliberate
headroom for non-flat futures (bucket, shelf, pick-from-hand). For the current
ground-cube task the dormant DOFs are constant (cube ``z`` = half-height, rotation
= pure yaw) and unused by the reward.

Action — absolute joint set points, internally clamped
------------------------------------------------------
The policy emits **absolute** 6-vector joint set points (``JOINT_NAMES`` order,
rad). The env clamps the per-step change to :data:`MAX_DELTA` via
:func:`clamp_setpoint`. One absolute interface is shared by RL / analytic / any
future IL, the observation stays Markov (no ``prev_action`` needed), and per-step
motion is bounded for real-safe deployment.

This module is RL-owned and *not* driven by the experimental ``il/observations.py``
(21-dim, planar poses), which coincides with a prefix of this layout but is **not**
the source of truth.
"""

from __future__ import annotations

import numpy as np

from pick_and_place import transforms as tf
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import CubePose

# Number of arm+gripper joints the policy sees and commands.
N_JOINTS = len(JOINT_NAMES)  # 6

# A pose is a uniform 9-vector: position (3) + 6D rotation (6).
ROT6D_DIM = 6
POSE_DIM = 3 + ROT6D_DIM  # 9

OBS_DIM = 31
ACT_DIM = N_JOINTS  # 6

# Observation slices / index (sim frame). Defined here so every consumer reads
# the same layout — a mismatch here is the quietest, nastiest schema bug.
JOINT_POS = slice(0, 6)
JOINT_VEL = slice(6, 12)
CUBE_POSE = slice(12, 21)
TARGET_POSE = slice(21, 30)
CONFIDENCE = 30

# Confidence for pure-sim chapters: the privileged cube pose is exact.
DEFAULT_CONFIDENCE = 1.0

# Max absolute per-joint change per control step (rad). The action is an absolute
# set point; :func:`clamp_setpoint` bounds the per-step move to this so motion is
# real-safe regardless of how far the policy's commanded set point jumps.
MAX_DELTA = 0.05


# ----------------------------------------------------------------------------
# 6D rotation representation
# ----------------------------------------------------------------------------


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    """Encode a 3x3 rotation as its first two columns, flattened (col 0 then col 1)."""
    r = np.asarray(rotation, dtype=np.float64)
    return np.concatenate((r[:, 0], r[:, 1]))


def rotation_from_6d(rot6d: np.ndarray) -> np.ndarray:
    """Recover a 3x3 rotation matrix from a 6D representation via Gram-Schmidt.

    Robust to *arbitrary* 6-vectors (e.g. a network output): the two columns are
    orthonormalised and the basis is completed with their cross product, so the
    result is always a valid right-handed rotation (orthonormal, det +1).
    """
    d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = d[0:3], d[3:6]
    b1 = a1 / np.linalg.norm(a1)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / np.linalg.norm(a2_orth)
    b3 = np.cross(b1, b2)
    return np.column_stack((b1, b2, b3))


# ----------------------------------------------------------------------------
# Pose 9-vectors
# ----------------------------------------------------------------------------


def pose_to_vec(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Pack a position (3) and a 3x3 rotation into the 9-vector pose layout."""
    out = np.empty(POSE_DIM, dtype=np.float64)
    out[0:3] = np.asarray(position, dtype=np.float64)
    out[3:POSE_DIM] = rotation_to_6d(rotation)
    return out


def pose_vec_from_xyz_yaw(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """9-vector pose for a ground object: position + a pure-yaw rotation."""
    return pose_to_vec((x, y, z), tf.rot_z(yaw)[:3, :3])


def pose_vec_from_cube_pose(pose: CubePose) -> np.ndarray:
    """9-vector pose for a :class:`CubePose` (full roll/pitch/yaw rotation)."""
    rotation = tf.rotation_zyx(pose.roll, pose.pitch, pose.yaw)[:3, :3]
    return pose_to_vec((pose.x, pose.y, pose.z), rotation)


# ----------------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------------


def build_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    cube_pose: np.ndarray,
    target_pose: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
) -> np.ndarray:
    """Assemble one 31-d observation in the sim frame.

    ``joint_pos`` / ``joint_vel`` are the 6 measured joint positions / velocities
    (``JOINT_NAMES`` order). ``cube_pose`` / ``target_pose`` are 9-vector poses
    (position + 6D rotation; see :func:`pose_to_vec`). ``confidence`` is the
    current-cube-pose confidence in ``[0, 1]`` (constant ``1.0`` in pure-sim
    chapters).
    """
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[JOINT_POS] = np.asarray(joint_pos, dtype=np.float32)
    obs[JOINT_VEL] = np.asarray(joint_vel, dtype=np.float32)
    obs[CUBE_POSE] = np.asarray(cube_pose, dtype=np.float32)
    obs[TARGET_POSE] = np.asarray(target_pose, dtype=np.float32)
    obs[CONFIDENCE] = float(confidence)
    return obs


# ----------------------------------------------------------------------------
# Action
# ----------------------------------------------------------------------------


def clamp_setpoint(
    prev_setpoint: np.ndarray,
    target_setpoint: np.ndarray,
    max_delta: float = MAX_DELTA,
) -> np.ndarray:
    """Clamp an absolute joint set point so no joint moves more than ``max_delta``.

    Both arguments are absolute 6-vectors (``JOINT_NAMES`` order, rad).
    ``target_setpoint`` is what the policy emits; this bounds its per-step change
    relative to ``prev_setpoint`` (the last commanded set point), per joint
    independently, for real-safe motion. Joint-*limit* clipping is the env's job
    (it owns the model), kept out of this model-free contract.
    """
    prev = np.asarray(prev_setpoint, dtype=np.float64)
    target = np.asarray(target_setpoint, dtype=np.float64)
    delta = np.clip(target - prev, -max_delta, max_delta)
    return prev + delta
