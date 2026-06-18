# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared task logic for the move-to-random-pose demo.

Sample a near-neutral arm pose and ease the arm there in joint space. Used by
both ``scripts/move_to_random_pose/sim.py`` and ``real.py`` — this is the
task's own setup/motion code, not part of the episode toolbox
(``episode_loop``/``EpisodeRecorder``/``recover_on`` in
``pick_and_place.episode_loop``/``recorder``/``safety``), which is why it
lives here rather than there.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pick_and_place.follower import ARM_JOINT_NAMES, JOINT_NAMES
from pick_and_place.trajectory import GRIPPER_OPEN, NEUTRAL_ARM_JOINTS

# ±radians of random perturbation from neutral used to sample a reachable pose.
# Tighter on the joints that tilt the gripper toward the floor, so a sampled
# pose rarely dips low enough to scrape it — mirrors the envelope
# pick-and-place uses for its own near-neutral poses, written fresh here since
# this task owns its own setup rather than importing pick-and-place's.
JOINT_PERTURBATION = 0.4
JOINT_PERTURBATION_OVERRIDES: dict[str, float] = {
    "shoulder_lift": 0.2,
    "elbow_flex": 0.2,
    "wrist_flex": 0.2,
}


def sample_reachable_pose(rng: np.random.Generator) -> tuple[dict[str, float], float]:
    """A random pose near neutral, safe enough to move to without IK or a
    collision check."""
    joints = {
        name: value + rng.uniform(
            -JOINT_PERTURBATION_OVERRIDES.get(name, JOINT_PERTURBATION),
            JOINT_PERTURBATION_OVERRIDES.get(name, JOINT_PERTURBATION),
        )
        for name, value in NEUTRAL_ARM_JOINTS.items()
    }
    gripper = float(rng.uniform(0.0, GRIPPER_OPEN))
    return joints, gripper


def smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def lerp_joints(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    return {name: a[name] + (b[name] - a[name]) * alpha for name in ARM_JOINT_NAMES}


def joint_qpos_adr(model: mujoco.MjModel) -> list[int]:
    """``qpos`` address of each of ``JOINT_NAMES`` (arm joints then gripper)."""
    return [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    ]


def current_pose(data: mujoco.MjData, joint_qpos_adr_: list[int]) -> tuple[dict[str, float], float]:
    """Read the sim's current arm joints and gripper from ``qpos``."""
    values = data.qpos[joint_qpos_adr_]
    return {name: float(v) for name, v in zip(ARM_JOINT_NAMES, values[:-1])}, float(values[-1])
