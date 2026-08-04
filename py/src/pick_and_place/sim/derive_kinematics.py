# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure :class:`So101Kinematics` off a compiled MuJoCo model.

The constants are read straight off the model at its reference (all-joints-zero)
pose, which exposes joint world anchors and axes directly via ``mjData``. This is
the only place MuJoCo is needed to *obtain* the arm's geometry; using it needs
nothing but :mod:`pick_and_place.core.kinematics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pick_and_place.core.geometry import GRIPPER_TARGET_POSITION
from pick_and_place.core.kinematics import JointLimit, PlanarSegment, So101Kinematics
from pick_and_place.spec.robot import ARM_JOINT_NAMES
from pick_and_place.core.transforms import Vec3


@dataclass(frozen=True)
class _JointFrame:
    position: Vec3
    axis: Vec3


def _joint_frame(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> _JointFrame:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return _JointFrame(position=data.xanchor[jid].copy(), axis=data.xaxis[jid].copy())


def _joint_limit(model: mujoco.MjModel, name: str) -> JointLimit:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        return JointLimit(float(lo), float(hi))
    return JointLimit(-np.inf, np.inf)


def derive_kinematics(model: mujoco.MjModel) -> So101Kinematics:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    pan = _joint_frame(model, data, "shoulder_pan")
    lift = _joint_frame(model, data, "shoulder_lift")
    elbow = _joint_frame(model, data, "elbow_flex")
    wrist_flex = _joint_frame(model, data, "wrist_flex")

    pan_axis = pan.position.copy()

    # Radial axis: horizontal, perpendicular to the (lateral) pitch axis, oriented
    # outward toward the arm. lift.axis x up, projected to the floor.
    radial_dir = np.cross(lift.axis, np.array((0.0, 0.0, 1.0)))
    radial = np.array((radial_dir[0], radial_dir[1]))
    radial /= np.linalg.norm(radial)
    to_wrist = wrist_flex.position[:2] - pan_axis[:2]
    if float(np.dot(radial, to_wrist)) < 0:
        radial = -radial

    def radial_of(position: Vec3) -> float:
        d = position[:2] - pan_axis[:2]
        return float(d[0] * radial[0] + d[1] * radial[1])

    def segment(frm: Vec3, to: Vec3) -> PlanarSegment:
        dr = radial_of(to) - radial_of(frm)
        dh = float(to[2] - frm[2])
        return PlanarSegment(radial=dr, height=dh, length=float(np.hypot(dr, dh)))

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    gripper_pos = data.xpos[gid].copy()
    gripper_rot = data.xmat[gid].reshape(3, 3)
    target = gripper_pos + gripper_rot @ GRIPPER_TARGET_POSITION

    gripper_x = gripper_rot[:, 0] / np.linalg.norm(gripper_rot[:, 0])
    a = target - wrist_flex.position
    a /= np.linalg.norm(a)
    pitch_axis = np.array((0.0, 1.0, 0.0))  # holds at pan = 0 (reference pose)
    ideal_x = np.cross(a, pitch_axis)
    ideal_x /= np.linalg.norm(ideal_x)
    ideal_y = pitch_axis
    wrist_roll_zero_twist = float(
        np.arctan2(np.dot(gripper_x, ideal_y), np.dot(gripper_x, ideal_x))
    )

    return So101Kinematics(
        pan_axis=pan_axis,
        shoulder_lift_radial=radial_of(lift.position),
        shoulder_lift_height=float(lift.position[2]),
        upper_arm=segment(lift.position, elbow.position),
        lower_arm=segment(elbow.position, wrist_flex.position),
        tool_length=segment(wrist_flex.position, target).length,
        wrist_roll_zero_twist=wrist_roll_zero_twist,
        joint_limits={name: _joint_limit(model, name) for name in ARM_JOINT_NAMES},
    )
