# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Implemented so far: phase 1 (neutral -> hover above the source cube) and
phase 2 (hover -> pregrasp at the source cube center).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pick_and_place.geometry import CubeFace, CubePose, VERTICAL_FACES, pregrasp_matrix
from pick_and_place.ik import solve_simple_pregrasp_ik
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics

# Tip-contact height of the source hover above the floor (clears the 3 cm cube
# top by 1 cm). At the grasp the tip sits at the cube-center height, so the
# world-z offset applied to the pregrasp is ``tip_z - pose.z``.
SOURCE_HOVER_TIP_Z = 0.04

# Gripper joint angle at the hover pregrasp: 40 deg open.
GRIPPER_OPEN = math.radians(40.0)

NEUTRAL_ARM_JOINTS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": -math.pi / 2,
}
NEUTRAL_GRIPPER = 0.0

# Phase 1: neutral -> hover pregrasp above the source cube.
STAGE1_DURATION = 2.0
# Phase 2: hover pregrasp -> pregrasp at the source cube center (vertical descent).
STAGE2_DURATION = 1.0


@dataclass(frozen=True)
class Frame:
    """One trajectory sample: arm joint set points plus the gripper set point."""

    joints: dict[str, float]
    gripper: float


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def _lerp_joints(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    return {name: a[name] + (b[name] - a[name]) * alpha for name in ARM_JOINT_NAMES}


@dataclass(frozen=True)
class GraspChoice:
    """The face and elbow used to grasp the source cube, with the joint set
    points solved for the hover and the at-cube pregrasp on that branch."""

    face: CubeFace
    elbow: str
    hover_joints: dict[str, float]
    pregrasp_joints: dict[str, float]


def select_grasp(k: So101Kinematics, source: CubePose) -> GraspChoice:
    """Pick the grasp face and elbow for the source cube.

    Simplified placeholder: tries the vertical faces in preference order and
    takes the first one whose hover and at-cube pregrasp are both reachable on a
    single elbow branch, preferring elbow-up. The full selection (which must also
    satisfy the carry and drop) arrives with the later phases.
    """
    hover_offset = SOURCE_HOVER_TIP_Z - source.z
    for face in VERTICAL_FACES:
        hover = pregrasp_matrix(face, source, hover_offset)
        pregrasp = pregrasp_matrix(face, source)
        if hover is None or pregrasp is None:
            continue
        hover_branches = solve_simple_pregrasp_ik(k, hover)
        pregrasp_branches = solve_simple_pregrasp_ik(k, pregrasp)
        for elbow in ("up", "down"):
            hover_branch = next((b for b in hover_branches if b.elbow == elbow), None)
            pregrasp_branch = next((b for b in pregrasp_branches if b.elbow == elbow), None)
            if hover_branch is None or pregrasp_branch is None:
                continue
            return GraspChoice(
                face=face,
                elbow=elbow,
                hover_joints=hover_branch.joints,
                pregrasp_joints=pregrasp_branch.joints,
            )
    raise ValueError("No reachable grasp for the source cube")


@dataclass(frozen=True)
class PickApproach:
    """Phases 1-2: neutral -> hover -> pregrasp, holding the pregrasp at the end."""

    k: So101Kinematics
    source: CubePose
    grasp: GraspChoice
    stage1_duration: float = STAGE1_DURATION
    stage2_duration: float = STAGE2_DURATION

    @property
    def duration(self) -> float:
        return self.stage1_duration + self.stage2_duration

    def evaluate(self, t: float) -> Frame:
        if t < self.stage1_duration:
            # Phase 1: swing from neutral to the hover, opening the gripper.
            alpha = _smoothstep(t / self.stage1_duration) if self.stage1_duration > 0 else 1.0
            joints = _lerp_joints(NEUTRAL_ARM_JOINTS, self.grasp.hover_joints, alpha)
            gripper = NEUTRAL_GRIPPER + (GRIPPER_OPEN - NEUTRAL_GRIPPER) * alpha
            return Frame(joints=joints, gripper=gripper)

        # Phase 2: descend straight down from the hover to the pregrasp at the
        # cube center, re-solving IK each frame so the tip tracks a vertical
        # line. The joint lerp is a defensive fallback for the rare frame whose
        # interpolated pose has no in-limit branch on the chosen elbow.
        alpha = (
            _smoothstep((t - self.stage1_duration) / self.stage2_duration)
            if self.stage2_duration > 0
            else 1.0
        )
        hover_offset = SOURCE_HOVER_TIP_Z - self.source.z
        matrix = pregrasp_matrix(self.grasp.face, self.source, hover_offset * (1.0 - alpha))
        branch = None
        if matrix is not None:
            branches = solve_simple_pregrasp_ik(self.k, matrix)
            branch = next((b for b in branches if b.elbow == self.grasp.elbow), None)
        joints = (
            branch.joints
            if branch is not None
            else _lerp_joints(self.grasp.hover_joints, self.grasp.pregrasp_joints, alpha)
        )
        return Frame(joints=joints, gripper=GRIPPER_OPEN)


def pick_approach(k: So101Kinematics, source: CubePose) -> PickApproach:
    return PickApproach(k=k, source=source, grasp=select_grasp(k, source))
