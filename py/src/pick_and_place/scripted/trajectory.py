# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Phases: (1) neutral -> hover, (2) hover -> grasp at cube center, (3) close
gripper to grasp, (4) lift and carry the grasped cube to a cruise waypoint above
the target, (5) descend from cruise into the canonical drop pose, (6) release,
lift clear, and flow back to neutral. The release is left to gravity: the
gripper set point opens and the cube falls on its own.

Where to take hold is :mod:`pick_and_place.scripted.grasp`, how to get the cube
across is :mod:`pick_and_place.scripted.carry`, and resuming a partly-run
trajectory is :mod:`pick_and_place.scripted.replan`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from pick_and_place.core import transforms as tf
from pick_and_place.core.geometry import CubeFace, CubePose
from pick_and_place.core.ik import solve_simple_grasp_ik
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.scripted.carry import (
    CarryJointChecker,
    CarryPlan,
    drop_descent_joints,
    nominal_drop_center_z,
    plan_carry_candidates,
)
from pick_and_place.scripted.grasp import (
    GraspChoice,
    free_grasp_candidates,
    grasp_candidates,
)
from pick_and_place.scripted.motion import (
    Frame,
    _joint_distance,
    _joint_move_duration,
    _lerp_joints,
    smoothstep,
    _timed_arc_fraction,
)
from pick_and_place.spec.robot import (
    GRIPPER_GRASP,
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
)


# Vertical lift after release, preserving the chosen drop orientation until the
# open jaws clear the cube.
POSTDROP_LIFT_Z = 0.04


# Phase 2: fixed, gentle vertical descent from the hover onto the cube.
DESCENT_DURATION = 1.6


# Phase 3: hold at the final aligned grasp pose before closing, then close the
# gripper gently onto the cube. The follow-up lift is planned from the synthetic
# grasp pose and gets us clear before trusting readback again.
GRASP_SETTLE_DURATION = 0.4


GRASP_CLOSE_DURATION = 0.45


GRASP_DURATION = GRASP_SETTLE_DURATION + GRASP_CLOSE_DURATION


# Hold the normal episode at its final carry pose before opening the gripper.
DROP_DWELL_DURATION = 0.5


# Phase 5a: fixed dwell at the drop hover while the gripper opens and the cube
# falls clear, before the retreat starts. Not a travel phase — the arm holds.
RELEASE_DURATION = 1.5


# The arm holds at the drop hover until the gripper has opened this far past the
# grasp, giving the released cube time to drop clear of the jaws before the arm
# starts moving (so the retreat doesn't fling it).
RETREAT_OPENING_ANGLE = math.radians(10.0)


@runtime_checkable
class TrajectoryPhase(Protocol):
    @property
    def duration(self) -> float: ...
    def evaluate(self, t: float) -> Frame: ...
    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class ApproachPhase:
    k: So101Kinematics
    start_joints: dict[str, float]
    start_gripper: float
    hover_joints: dict[str, float]

    @property
    def name(self) -> str:
        return "approach"

    @cached_property
    def duration(self) -> float:
        return _joint_move_duration(self.k, self.start_joints, self.hover_joints)

    def evaluate(self, t: float) -> Frame:
        alpha = _timed_arc_fraction(t / self.duration) if self.duration > 0 else 1.0
        joints = _lerp_joints(self.start_joints, self.hover_joints, alpha)
        gripper = self.start_gripper + (GRIPPER_OPEN - self.start_gripper) * alpha
        return Frame(joints=joints, gripper=gripper)


@dataclass(frozen=True)
class DescentPhase:
    k: So101Kinematics
    grasp: GraspChoice

    @property
    def name(self) -> str:
        return "descent"

    @property
    def face(self) -> CubeFace:
        return self.grasp.face

    @property
    def elbow(self) -> str:
        return self.grasp.elbow

    @cached_property
    def duration(self) -> float:
        return DESCENT_DURATION

    def evaluate(self, t: float) -> Frame:
        alpha = smoothstep(t / self.duration) if self.duration > 0 else 1.0
        matrix = tf.with_position(
            self.grasp.hover_matrix,
            tf.get_position(self.grasp.hover_matrix)
            + (
                tf.get_position(self.grasp.grasp_matrix)
                - tf.get_position(self.grasp.hover_matrix)
            )
            * alpha,
        )
        branch = None
        if self.grasp.face != "free":
            branches = solve_simple_grasp_ik(self.k, matrix)
            branch = next((b for b in branches if b.elbow == self.grasp.elbow), None)
        joints = (
            branch.joints
            if branch is not None
            else _lerp_joints(self.grasp.hover_joints, self.grasp.grasp_joints, alpha)
        )
        return Frame(joints=joints, gripper=GRIPPER_OPEN)


@dataclass(frozen=True)
class GraspPhase:
    grasp_joints: dict[str, float]
    start_gripper: float = GRIPPER_OPEN

    @property
    def name(self) -> str:
        return "grasp"

    @cached_property
    def duration(self) -> float:
        return GRASP_DURATION

    def evaluate(self, t: float) -> Frame:
        if t <= GRASP_SETTLE_DURATION:
            return Frame(joints=self.grasp_joints, gripper=self.start_gripper)
        close_t = t - GRASP_SETTLE_DURATION
        alpha = (
            smoothstep(close_t / GRASP_CLOSE_DURATION)
            if GRASP_CLOSE_DURATION > 0
            else 1.0
        )
        gripper = self.start_gripper + (GRIPPER_GRASP - self.start_gripper) * alpha
        return Frame(joints=self.grasp_joints, gripper=gripper)


@dataclass(frozen=True)
class LiftPhase:
    k: So101Kinematics
    grasp_joints: dict[str, float]
    hover_joints: dict[str, float]

    @property
    def name(self) -> str:
        return "lift"

    @cached_property
    def duration(self) -> float:
        return _joint_move_duration(self.k, self.grasp_joints, self.hover_joints)

    def evaluate(self, t: float) -> Frame:
        alpha = _timed_arc_fraction(t / self.duration) if self.duration > 0 else 1.0
        return Frame(
            joints=_lerp_joints(self.grasp_joints, self.hover_joints, alpha),
            gripper=GRIPPER_GRASP,
        )


@dataclass(frozen=True)
class RecoveryLiftPhase(LiftPhase):
    @property
    def name(self) -> str:
        return "recovery_lift"


@dataclass(frozen=True)
class CarryPhase:
    """Long-distance transit from the lifted grasp to the cruise waypoint above
    the target, in joint space -- see ``CarryPlan``'s docstring for why."""

    k: So101Kinematics
    grasp_joints: dict[str, float]
    cruise_joints: dict[str, float]

    @property
    def name(self) -> str:
        return "carry"

    @cached_property
    def duration(self) -> float:
        return _joint_move_duration(self.k, self.grasp_joints, self.cruise_joints)

    def evaluate(self, t: float) -> Frame:
        alpha = _timed_arc_fraction(t / self.duration) if self.duration > 0 else 1.0
        return Frame(
            joints=_lerp_joints(self.grasp_joints, self.cruise_joints, alpha),
            gripper=GRIPPER_GRASP,
        )


@dataclass(frozen=True)
class DropDescentPhase:
    """Final Cartesian approach from the cruise waypoint into the drop pose,
    mirroring ``DescentPhase`` on the pickup side -- see ``CarryPlan``'s
    docstring for why."""

    k: So101Kinematics
    carry: CarryPlan

    @property
    def name(self) -> str:
        return "drop_descent"

    @cached_property
    def duration(self) -> float:
        return DESCENT_DURATION

    def evaluate(self, t: float) -> Frame:
        alpha = smoothstep(t / self.duration) if self.duration > 0 else 1.0
        joints = drop_descent_joints(
            self.k,
            self.carry.cruise_matrix,
            self.carry.drop_matrix,
            self.carry.cruise_joints,
            self.carry.drop_joints,
            self.carry.elbow,
            alpha,
        )
        return Frame(joints=joints, gripper=GRIPPER_GRASP)


@dataclass(frozen=True)
class ReleasePhase:
    predrop_joints: dict[str, float]
    postdrop_joints: dict[str, float]
    start_gripper: float = GRIPPER_GRASP
    pre_release_delay: float = DROP_DWELL_DURATION

    @property
    def name(self) -> str:
        return "release"

    @cached_property
    def duration(self) -> float:
        return self.pre_release_delay + RELEASE_DURATION

    def evaluate(self, t: float) -> Frame:
        elapsed = min(RELEASE_DURATION, max(0.0, t - self.pre_release_delay))
        opening_fraction = RETREAT_OPENING_ANGLE / (GRIPPER_OPEN - self.start_gripper)
        movement_start = opening_fraction * RELEASE_DURATION
        movement_duration = RELEASE_DURATION - movement_start
        movement_phase = (
            min(1.0, max(0.0, (elapsed - movement_start) / movement_duration))
            if movement_duration > 0.0
            else 1.0
        )
        joints = _lerp_joints(
            self.predrop_joints,
            self.postdrop_joints,
            _timed_arc_fraction(movement_phase),
        )
        open_alpha = elapsed / RELEASE_DURATION if RELEASE_DURATION > 0 else 1.0
        gripper = self.start_gripper + (GRIPPER_OPEN - self.start_gripper) * open_alpha
        return Frame(joints=joints, gripper=gripper)


@dataclass(frozen=True)
class RetreatPhase:
    k: So101Kinematics
    start_joints: dict[str, float]
    end_joints: dict[str, float]
    end_gripper: float
    start_gripper: float = GRIPPER_OPEN

    @property
    def name(self) -> str:
        return "retreat"

    @cached_property
    def duration(self) -> float:
        return _joint_move_duration(self.k, self.start_joints, self.end_joints)

    def evaluate(self, t: float) -> Frame:
        alpha = _timed_arc_fraction(t / self.duration) if self.duration > 0 else 1.0
        return Frame(
            joints=_lerp_joints(self.start_joints, self.end_joints, alpha),
            gripper=self.start_gripper
            + (self.end_gripper - self.start_gripper) * smoothstep(alpha),
        )


@dataclass(frozen=True)
class Trajectory:
    phases: tuple[TrajectoryPhase, ...]
    source: CubePose | None = None
    target: CubePose | None = None
    grasp: GraspChoice | None = None
    carry: CarryPlan | None = None
    start_joints: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(NEUTRAL_ARM_JOINTS)
    )
    start_gripper: float = NEUTRAL_GRIPPER
    end_joints: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(NEUTRAL_ARM_JOINTS)
    )
    end_gripper: float = NEUTRAL_GRIPPER

    @cached_property
    def duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def evaluate(self, t: float) -> Frame:
        if not self.phases:
            return Frame(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)
        for phase in self.phases[:-1]:
            if t < phase.duration:
                return phase.evaluate(t)
            t -= phase.duration
        return self.phases[-1].evaluate(max(0.0, min(t, self.phases[-1].duration)))


def trajectory_candidates(
    k: So101Kinematics,
    source: CubePose,
    target: CubePose,
    start_joints: dict[str, float],
    start_gripper: float,
    end_joints: dict[str, float],
    end_gripper: float,
    *,
    free_grasp: bool = False,
    carry_ok: CarryJointChecker | None = None,
) -> Iterator[Trajectory]:
    """Yield full trajectories from start to end in grasp preference order."""
    candidates = free_grasp_candidates(k, source) if free_grasp else grasp_candidates(k, source)
    for grasp in candidates:
        yield from trajectory_candidates_for_grasp(
            k,
            source,
            target,
            start_joints,
            start_gripper,
            end_joints,
            end_gripper,
            grasp,
            free_grasp=free_grasp,
            carry_ok=carry_ok,
        )


def trajectory_candidates_for_grasp(
    k: So101Kinematics,
    source: CubePose,
    target: CubePose,
    start_joints: dict[str, float],
    start_gripper: float,
    end_joints: dict[str, float],
    end_gripper: float,
    grasp: GraspChoice,
    *,
    free_grasp: bool = False,
    carry_ok: CarryJointChecker | None = None,
) -> Iterator[Trajectory]:
    """Yield full trajectories for one selected grasp."""
    drop_cube_center_z = nominal_drop_center_z(target)
    release_delay = 0.0 if free_grasp else DROP_DWELL_DURATION
    for carry in plan_carry_candidates(
        k,
        grasp,
        target,
        drop_cube_center_z=drop_cube_center_z,
        carry_ok=carry_ok,
    ):
        endpoint = carry.drop_matrix
        endpoint_position = tf.get_position(endpoint)
        predrop_joints = carry.drop_joints
        # As with the cruise height, some orientations lose IK reachability at the
        # nominal retreat height well before POSTDROP_LIFT_Z -- fall back to a
        # lower (but still clear-of-the-cube) retreat rather than discarding an
        # otherwise-ideal carry candidate over an unreachable retreat alone.
        postdrop_branch = None
        for lift_z in (POSTDROP_LIFT_Z, 0.03, 0.02, 0.01):
            postdrop_hover = tf.with_position(endpoint, endpoint_position + np.array((0.0, 0.0, lift_z)))
            postdrop_branch = min(
                solve_simple_grasp_ik(k, postdrop_hover),
                key=lambda branch: _joint_distance(predrop_joints, branch.joints),
                default=None,
            )
            if postdrop_branch is not None:
                break
        if postdrop_branch is None:
            continue

        phases = (
            ApproachPhase(k, start_joints, start_gripper, grasp.hover_joints),
            DescentPhase(k, grasp),
            GraspPhase(grasp.grasp_joints),
            (
                RecoveryLiftPhase(k, grasp.grasp_joints, grasp.lift_joints)
                if free_grasp
                else LiftPhase(k, grasp.grasp_joints, grasp.lift_joints)
            ),
            CarryPhase(k, grasp.lift_joints, carry.cruise_joints),
            DropDescentPhase(k, carry),
            ReleasePhase(
                predrop_joints,
                postdrop_branch.joints,
                pre_release_delay=release_delay,
            ),
            RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper),
        )

        yield Trajectory(
            phases=phases,
            source=source,
            target=target,
            grasp=grasp,
            carry=carry,
            start_joints=start_joints,
            start_gripper=start_gripper,
            end_joints=end_joints,
            end_gripper=end_gripper,
        )
