# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The SO-101 arm as a planar chain, and its closed-form forward kinematics.

:class:`So101Kinematics` is a measurement of one arm: segment lengths, the world
position of the shoulder-pan axis, the wrist's built-in twist, and the joint
limits. Nothing here builds one — :mod:`pick_and_place.sim.derive_kinematics` reads
the numbers off a compiled MuJoCo model — so planning and IK can consume the
measurement without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pick_and_place.core.transforms import Vec3


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
