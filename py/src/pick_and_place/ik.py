# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Closed-form IK for the simple (vertical) grasp pose.

The gripper roll axis points straight up, so the approach is fixed (down) and
only the two elbow branches of the planar 2R arm remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pick_and_place import transforms as tf
from pick_and_place.geometry import GRIPPER_TARGET_POSITION
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics
from pick_and_place.transforms import Mat4


@dataclass(frozen=True)
class IkBranch:
    joints: dict[str, float]
    elbow: str  # 'up' or 'down'


def _normalize_angle(angle: float) -> float:
    result = angle % (2 * np.pi)
    if result > np.pi:
        result -= 2 * np.pi
    if result <= -np.pi:
        result += 2 * np.pi
    return result


def _solve_2r(
    l1: float, l2: float, target_radial: float, target_height: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Planar 2R IK. Returns ((shoulder, elbow) up, (shoulder, elbow) down) or
    ``None`` if the point is outside the arm's annulus."""
    r2 = target_radial * target_radial + target_height * target_height
    r = np.sqrt(r2)
    if r > l1 + l2 or r < abs(l1 - l2):
        return None

    cos2 = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    elbow_geom = np.arccos(np.clip(cos2, -1.0, 1.0))

    phi = np.arctan2(target_height, target_radial)
    cos_alpha = (l1 * l1 + r2 - l2 * l2) / (2 * l1 * r)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))

    return (phi + alpha, -elbow_geom), (phi - alpha, elbow_geom)


def solve_simple_grasp_ik(
    k: So101Kinematics, world_from_gripper: Mat4
) -> list[IkBranch]:
    """Return the within-limit elbow branches (possibly empty) for the gripper
    pose. Empty means unreachable / outside joint limits."""
    gripper_x = tf.transform_direction(world_from_gripper, np.array((1.0, 0.0, 0.0)))
    gripper_z = tf.transform_direction(world_from_gripper, np.array((0.0, 0.0, 1.0)))

    target = tf.transform_point(world_from_gripper, GRIPPER_TARGET_POSITION)
    approach = -gripper_z
    closing = gripper_x

    dx = target[0] - k.pan_axis[0]
    dy = target[1] - k.pan_axis[1]
    if np.hypot(dx, dy) < 1e-4:
        return []
    azimuth = np.arctan2(dy, dx)
    shoulder_pan = -azimuth
    radial_dir = np.array((np.cos(azimuth), np.sin(azimuth), 0.0))
    plane_normal = np.array((-np.sin(azimuth), np.cos(azimuth), 0.0))

    wrist = target - approach * k.tool_length
    target_radial = (
        (wrist[0] - k.pan_axis[0]) * radial_dir[0]
        + (wrist[1] - k.pan_axis[1]) * radial_dir[1]
        - k.shoulder_lift_radial
    )
    target_height = wrist[2] - k.shoulder_lift_height

    solutions = _solve_2r(
        k.upper_arm.length, k.lower_arm.length, target_radial, target_height
    )
    if solutions is None:
        return []

    upper_rest = np.arctan2(k.upper_arm.height, k.upper_arm.radial)
    lower_rest = np.arctan2(k.lower_arm.height, k.lower_arm.radial)
    elbow_rest = lower_rest - upper_rest
    tool_pitch = np.arctan2(approach[2], float(np.dot(approach, radial_dir)))

    zero_roll_x = np.cross(approach, plane_normal)
    zero_roll_x /= np.linalg.norm(zero_roll_x)
    zero_roll_y = plane_normal
    roll_angle = (
        np.arctan2(np.dot(closing, zero_roll_y), np.dot(closing, zero_roll_x))
        - k.wrist_roll_zero_twist
    )

    branches: list[IkBranch] = []
    for elbow, (shoulder_geom, elbow_geom) in (
        ("up", solutions[0]),
        ("down", solutions[1]),
    ):
        shoulder_lift = upper_rest - shoulder_geom
        elbow_flex = elbow_rest - elbow_geom
        wrist_flex = -shoulder_lift - elbow_flex - tool_pitch

        joints = {
            "shoulder_pan": _normalize_angle(shoulder_pan),
            "shoulder_lift": _normalize_angle(shoulder_lift),
            "elbow_flex": _normalize_angle(elbow_flex),
            "wrist_flex": _normalize_angle(wrist_flex),
            "wrist_roll": _normalize_angle(roll_angle),
        }

        within = True
        for name in ARM_JOINT_NAMES:
            limit = k.joint_limits[name]
            if joints[name] < limit.min or joints[name] > limit.max:
                within = False
                break
        if within:
            branches.append(IkBranch(joints=joints, elbow=elbow))

    return branches
