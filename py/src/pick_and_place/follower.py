# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Sim-free helpers for driving a physical SO-101 follower through lerobot.

Lets a hardware run stream the planned trajectory to a real arm without
importing MuJoCo or the simulation. The
trajectory speaks the *sim frame* (arm joints in radians, gripper a joint angle
in radians); the follower speaks the *real frame* (arm joints in degrees,
gripper a 0-100 position). The conversion lives here:

- arm joints: ``real_deg = sim_deg + offset`` per joint (offsets default to
  zero, so by default the real frame is just the sim frame in degrees);
- gripper: a nonlinear angle->position map calibrated on the hardware.

``make_so101_follower`` imports lerobot lazily, so importing this module never
requires lerobot or any hardware to be present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Follower joint order as lerobot's SO101Follower reports and accepts it.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# The five arm joints (everything but the gripper), in the same order.
ARM_JOINT_NAMES = JOINT_NAMES[:5]
GRIPPER_INDEX = 5

# Observed follower gripper encoder endpoints, calibrated on the hardware.
# The physical open stop reads a bit below 100, and the jaw shape matches the
# sim's command curve better than a linear map across the hinge's authored range.
GRIPPER_READBACK_CLOSED = 2.3
GRIPPER_READBACK_OPEN = 98.5
GRIPPER_RENDER_CLOSED_DEG = -10.0
GRIPPER_RENDER_OPEN_DEG = 120.0


def joints_to_action(joints: np.ndarray) -> dict[str, float]:
    """Pack a 6-vector of real-frame joints into a lerobot ``send_action`` dict."""
    return {f"{name}.pos": float(joints[i]) for i, name in enumerate(JOINT_NAMES)}


def action_to_joints(action: dict[str, float], previous: np.ndarray) -> np.ndarray:
    """Read a lerobot observation/action dict back into a 6-vector.

    Missing joints keep their value from ``previous`` so a partial observation
    never silently zeroes a joint.
    """
    joints = np.asarray(previous, dtype=float).copy()
    for i, name in enumerate(JOINT_NAMES):
        if f"{name}.pos" in action:
            joints[i] = float(action[f"{name}.pos"])
        elif name in action:
            joints[i] = float(action[name])
    return joints


def clamp_joints(joints: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clip a real-frame 6-vector to ``[low, high]`` (degrees for the arm, 0-100
    position for the gripper).

    The bounds are not hardcoded here: a caller derives the arm limits from the
    same model the trajectory was planned against, so a valid command is never
    altered while an out-of-range one is still caught. (Hardcoded limits would
    silently pin ``wrist_roll`` to the wrong range.)
    """
    return np.clip(joints, low, high)


def load_follower_joint_offsets(path: str | Path | None) -> np.ndarray:
    """Load per-joint ``sim_deg -> real_deg`` offsets (degrees), defaulting to zero.

    The stored convention is ``real_deg = sim_deg + offset``. A missing path or
    file resolves to zero offsets so a hardware run has a stable default before
    any calibration exists (Phase 1 measures these offsets empirically).
    """
    offsets = np.zeros(len(JOINT_NAMES), dtype=float)
    if path is None:
        return offsets
    p = Path(path)
    if not p.is_file():
        return offsets
    data = json.loads(p.read_text())
    for i, name in enumerate(JOINT_NAMES):
        if name in data:
            offsets[i] = float(data[name])
    return offsets


def gripper_angle_to_position(angle_rad: float) -> float:
    """Map a sim gripper joint angle (radians) to a follower 0-100 position.

    Inverts the calibrated jaw range (encoder ``[2.3, 98.5]`` <-> render
    angle ``[-10deg, 120deg]``) so the physical jaw opens to match the sim.
    Treated as approximate for now.
    """
    angle_deg = math.degrees(angle_rad)
    span_deg = GRIPPER_RENDER_OPEN_DEG - GRIPPER_RENDER_CLOSED_DEG
    t = float(np.clip((angle_deg - GRIPPER_RENDER_CLOSED_DEG) / span_deg, 0.0, 1.0))
    span_enc = GRIPPER_READBACK_OPEN - GRIPPER_READBACK_CLOSED
    return GRIPPER_READBACK_CLOSED + t * span_enc


def sim_frame_to_real(
    arm_joints_rad: dict[str, float], gripper_rad: float, offsets: np.ndarray
) -> np.ndarray:
    """Convert a trajectory set point (sim frame) into a real-frame 6-vector.

    Arm joints: ``real_deg = sim_deg + offset``. Gripper: nonlinear angle->position
    map (the gripper carries no additive offset).
    """
    out = np.zeros(len(JOINT_NAMES), dtype=float)
    for i, name in enumerate(ARM_JOINT_NAMES):
        out[i] = math.degrees(arm_joints_rad[name]) + offsets[i]
    out[GRIPPER_INDEX] = gripper_angle_to_position(gripper_rad)
    return out


def make_so101_follower(
    port: str,
    robot_id: str,
    *,
    calibration_dir: str | None = None,
    max_relative_target: float | None = None,
    disable_torque_on_disconnect: bool = True,
) -> Any:
    """Construct a lerobot ``SO101Follower`` (imported lazily).

    ``use_degrees=True`` makes the follower report and accept the arm joints in
    degrees (gripper as a 0-100 position), which is the real frame this module
    converts to. lerobot is imported here, not at module top, so importing this
    module never requires lerobot or any hardware.
    """
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    return make_robot_from_config(
        SO101FollowerConfig(
            port=port,
            id=robot_id,
            calibration_dir=calibration_dir,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=disable_torque_on_disconnect,
            use_degrees=True,
        )
    )
