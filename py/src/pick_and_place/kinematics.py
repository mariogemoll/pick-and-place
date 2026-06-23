# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""SO-101 kinematic constants for the closed-form IK.

The constants are read straight off the compiled MuJoCo model at its reference
(all-joints-zero) pose, which exposes joint world anchors and axes directly via
``mjData``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pick_and_place.geometry import GRIPPER_TARGET_POSITION
from pick_and_place.transforms import Vec3

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


@dataclass(frozen=True)
class PlanarSegment:
    radial: float
    height: float
    length: float


@dataclass(frozen=True)
class JointLimit:
    min: float
    max: float


@dataclass(frozen=True)
class So101Kinematics:
    pan_axis: Vec3  # world (x, y) of the vertical shoulder_pan axis; z unused
    shoulder_lift_radial: float
    shoulder_lift_height: float
    upper_arm: PlanarSegment  # shoulder_lift -> elbow_flex
    lower_arm: PlanarSegment  # elbow_flex -> wrist_flex
    tool_length: float  # wrist_flex -> gripper target along the approach
    wrist_roll_zero_twist: float  # roll offset from the 2.8 deg arm twist
    joint_limits: dict[str, JointLimit]

    def tip_position(self, joints: dict[str, float]) -> Vec3:
        """World position of the gripper IK target for the given arm joints.

        Closed-form forward kinematics of the planar chain, inverting the
        conventions in ``solve_simple_grasp_ik``. Wrist roll spins about the
        approach axis and so does not move the target; only pan and the three
        planar joints matter. Matches the IK target to sub-millimetre.
        """
        azimuth = -joints["shoulder_pan"]
        radial_dir = np.array((np.cos(azimuth), np.sin(azimuth), 0.0))
        up = np.array((0.0, 0.0, 1.0))
        upper_rest = np.arctan2(self.upper_arm.height, self.upper_arm.radial)
        lower_rest = np.arctan2(self.lower_arm.height, self.lower_arm.radial)
        elbow_rest = lower_rest - upper_rest
        shoulder_geom = upper_rest - joints["shoulder_lift"]
        elbow_geom = elbow_rest - joints["elbow_flex"]
        l1, l2 = self.upper_arm.length, self.lower_arm.length
        radial_rel = l1 * np.cos(shoulder_geom) + l2 * np.cos(shoulder_geom + elbow_geom)
        height_rel = l1 * np.sin(shoulder_geom) + l2 * np.sin(shoulder_geom + elbow_geom)
        wrist = (
            np.array((self.pan_axis[0], self.pan_axis[1], 0.0))
            + radial_dir * (self.shoulder_lift_radial + radial_rel)
            + up * (self.shoulder_lift_height + height_rel)
        )
        tool_pitch = -(joints["shoulder_lift"] + joints["elbow_flex"] + joints["wrist_flex"])
        approach = radial_dir * np.cos(tool_pitch) + up * np.sin(tool_pitch)
        return wrist + approach * self.tool_length


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
