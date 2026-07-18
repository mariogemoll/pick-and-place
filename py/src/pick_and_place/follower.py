# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Sim-free helpers for driving a physical SO-101 follower through lerobot.

Lets a hardware run stream the planned trajectory to a real arm without
importing MuJoCo or the simulation. The
trajectory speaks the *sim frame* (arm joints in radians, gripper a joint angle
in radians); the follower speaks the *real frame* (arm joints in degrees,
gripper a 0-100 position). The conversion lives here:

- arm joints: a plain radians<->degrees conversion (the follower's own lerobot
  calibration already aligns each servo's zero with the sim frame);
- gripper: a nonlinear angle->position map calibrated on the hardware.

``make_so101_follower`` imports lerobot lazily, so importing this module never
requires lerobot or any hardware to be present.
"""

from __future__ import annotations

import math
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


def gripper_position_to_angle(position: float) -> float:
    """Map a follower 0-100 position to a sim gripper joint angle (radians)."""
    span_enc = GRIPPER_READBACK_OPEN - GRIPPER_READBACK_CLOSED
    t = float(np.clip((position - GRIPPER_READBACK_CLOSED) / span_enc, 0.0, 1.0))
    span_deg = GRIPPER_RENDER_OPEN_DEG - GRIPPER_RENDER_CLOSED_DEG
    angle_deg = GRIPPER_RENDER_CLOSED_DEG + t * span_deg
    return math.radians(angle_deg)

def sim_frame_to_real(
    arm_joints_rad: dict[str, float],
    gripper_rad: float,
    offsets_deg: dict[str, float] | None = None,
) -> np.ndarray:
    """Convert a trajectory set point (sim frame) into a real-frame 6-vector.

    Arm joints: radians->degrees. Gripper: nonlinear angle->position map.

    ``offsets_deg`` are per-joint zero errors measured by the session
    calibration, in the "add to the sim joints" (exporter) sense. A servo whose
    command reads ``theta`` sits physically at model angle ``theta + offset``, so
    to place a joint at the planned model angle the command is
    ``degrees(model) - offset``. Passing ``offsets_deg`` applies that
    feed-forward correction; omitting it (the default) is the raw mapping the
    calibration and pair export rely on.
    """
    out = np.zeros(len(JOINT_NAMES), dtype=float)
    for i, name in enumerate(ARM_JOINT_NAMES):
        deg = math.degrees(arm_joints_rad[name])
        if offsets_deg is not None:
            deg -= offsets_deg.get(name, 0.0)
        out[i] = deg
    out[GRIPPER_INDEX] = gripper_angle_to_position(gripper_rad)
    return out

def real_frame_to_sim(
    real_joints: np.ndarray, offsets_deg: dict[str, float] | None = None
) -> tuple[dict[str, float], float]:
    """Convert a real-frame 6-vector back into sim-frame joints (radians).

    The inverse of :func:`sim_frame_to_real`: with ``offsets_deg`` given, a servo
    readback ``theta`` maps to the joint's true model angle ``theta + offset`` so
    the sim mirror and any replanning start from where the arm physically is.
    Omit ``offsets_deg`` (the default) to recover the raw servo readback the
    calibration fit consumes.
    """
    arm_joints_rad = {}
    for i, name in enumerate(ARM_JOINT_NAMES):
        deg = float(real_joints[i])
        if offsets_deg is not None:
            deg += offsets_deg.get(name, 0.0)
        arm_joints_rad[name] = math.radians(deg)
    gripper_rad = gripper_position_to_angle(real_joints[GRIPPER_INDEX])
    return arm_joints_rad, gripper_rad


def load_joint_zero_offsets(path: Any) -> dict[str, float]:
    """Read the latest session joint-zero offsets (degrees) from a store written
    by the calibration driver (``config/joint_zeros.json``).

    Returns the fitted arm-joint offsets in the "add to the sim joints" sense,
    ready to hand to :func:`sim_frame_to_real` / :func:`real_frame_to_sim`.
    """
    import json
    from pathlib import Path

    store = json.loads(Path(path).read_text())
    latest = store.get("latest")
    if latest is None:
        raise ValueError(f"{path} has no 'latest' calibration entry.")
    return {name: float(value) for name, value in latest["offsets_deg"].items()}

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

def make_so101_leader(
    port: str,
    robot_id: str,
    *,
    calibration_dir: str | None = None,
) -> Any:
    """Construct a lerobot ``SO101Leader`` (imported lazily)."""
    try:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        return SO101Leader(SO101LeaderConfig(port=port, id=robot_id, calibration_dir=calibration_dir))
    except ModuleNotFoundError:
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig
        
        leader_cfg = SO101LeaderConfig(port=port, id=robot_id, calibration_dir=calibration_dir)
        return make_teleoperator_from_config(leader_cfg)

