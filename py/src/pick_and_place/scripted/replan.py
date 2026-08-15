# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Re-plan what is left of a trajectory from where the arm actually is.

A run does not stop at a checkpoint because something went wrong — it stops
because the cube moved, the grasp slipped, or the measured pose drifted from the
planned one. Re-planning takes the phase that completed and the arm's measured
state and yields candidates for the remainder, so the parts already executed are
never redone and the cube's current pose is what the rest is planned against.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np

from pick_and_place.core import transforms as tf
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.ik import solve_simple_grasp_ik
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.scripted.carry import nominal_drop_center_z, plan_carry_candidates
from pick_and_place.scripted.grasp import GraspChoice, free_grasp_candidates, grasp_candidates
from pick_and_place.scripted.motion import _joint_distance
from pick_and_place.scripted.trajectory import (
    ApproachPhase,
    CarryPhase,
    DROP_DWELL_DURATION,
    DescentPhase,
    DropDescentPhase,
    GraspPhase,
    LiftPhase,
    POSTDROP_LIFT_Z,
    RecoveryLiftPhase,
    ReleasePhase,
    RetreatPhase,
    Trajectory,
)
from pick_and_place.spec.robot import GRIPPER_OPEN


def replan_remaining_candidates(
    k: So101Kinematics,
    measured_joints: dict[str, float],
    measured_gripper: float,
    completed_phase_name: str | None,
    source: CubePose,
    target: CubePose,
    grasp: GraspChoice | None,
    end_joints: dict[str, float],
    end_gripper: float,
    *,
    free_grasp: bool = False,
) -> Iterator[Trajectory]:
    """Yield remaining-trajectory candidates from the measured state."""

    if completed_phase_name == "retreat":
        yield Trajectory(
            (),
            source,
            target,
            grasp,
            None,
            measured_joints,
            measured_gripper,
            end_joints,
            end_gripper,
        )
        return

    if completed_phase_name == "release":
        yield Trajectory(
            (RetreatPhase(k, measured_joints, end_joints, end_gripper, measured_gripper),),
            source,
            target,
            grasp,
            None,
            measured_joints,
            measured_gripper,
            end_joints,
            end_gripper,
        )
        return

    grasps = (
        [grasp]
        if grasp is not None
        else list(free_grasp_candidates(k, source) if free_grasp else grasp_candidates(k, source))
    )

    drop_cube_center_z = nominal_drop_center_z(target)
    release_delay = 0.0 if free_grasp else DROP_DWELL_DURATION
    for g in grasps:
        for carry in plan_carry_candidates(
            k,
            g,
            target,
            drop_cube_center_z=drop_cube_center_z,
        ):
            endpoint = carry.drop_matrix
            endpoint_position = tf.get_position(endpoint)
            predrop_joints = carry.drop_joints

            # Same fallback ladder as the initial-planning path
            # (trajectory_candidates_for_grasp): some orientations lose IK
            # reachability at the nominal retreat height well before
            # POSTDROP_LIFT_Z. With the smaller canonical drop-pose family,
            # skipping this ladder here meant a replan could reject every carry
            # candidate the initial plan (which does have the ladder) would
            # have accepted, aborting an otherwise-fine episode.
            postdrop_branch = None
            for lift_z in (POSTDROP_LIFT_Z, 0.03, 0.02, 0.01):
                postdrop_hover = tf.with_position(
                    endpoint, endpoint_position + np.array((0.0, 0.0, lift_z))
                )
                postdrop_branch = min(
                    solve_simple_grasp_ik(k, postdrop_hover),
                    key=lambda branch: _joint_distance(predrop_joints, branch.joints),
                    default=None,
                )
                if postdrop_branch is not None:
                    break
            if postdrop_branch is None:
                continue

            phases = []
            start_joints = measured_joints
            if completed_phase_name is None:
                # We are at the very beginning. Next is Approach.
                phases.append(ApproachPhase(k, measured_joints, measured_gripper, g.hover_joints))
                phases.append(DescentPhase(k, g))
                phases.append(GraspPhase(g.grasp_joints, start_gripper=GRIPPER_OPEN))
                phases.append(
                    RecoveryLiftPhase(k, g.grasp_joints, g.lift_joints)
                    if free_grasp
                    else LiftPhase(k, g.grasp_joints, g.lift_joints)
                )
                phases.append(CarryPhase(k, g.lift_joints, carry.cruise_joints))
                phases.append(DropDescentPhase(k, carry))
                phases.append(
                    ReleasePhase(
                        predrop_joints,
                        postdrop_branch.joints,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
            elif completed_phase_name == "approach":
                # Next is Descent. We start from measured hover.
                phases.append(DescentPhase(k, dataclasses.replace(g, hover_joints=measured_joints)))
                phases.append(GraspPhase(g.grasp_joints, start_gripper=measured_gripper))
                phases.append(
                    RecoveryLiftPhase(k, g.grasp_joints, g.lift_joints)
                    if free_grasp
                    else LiftPhase(k, g.grasp_joints, g.lift_joints)
                )
                phases.append(CarryPhase(k, g.lift_joints, carry.cruise_joints))
                phases.append(DropDescentPhase(k, carry))
                phases.append(
                    ReleasePhase(
                        predrop_joints,
                        postdrop_branch.joints,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
            elif completed_phase_name == "descent":
                # Next is Grasp. Near the floor, real-joint readback can map a
                # physically clear pose a millimetre or two below the sim floor.
                # Keep the locked grasp pose as the sim seed and let the real arm
                # continue tracking that command instead of treating the biased
                # readback as ground truth.
                phases.append(GraspPhase(g.grasp_joints, start_gripper=measured_gripper))
                phases.append(
                    RecoveryLiftPhase(k, g.grasp_joints, g.lift_joints)
                    if free_grasp
                    else LiftPhase(k, g.grasp_joints, g.lift_joints)
                )
                phases.append(CarryPhase(k, g.lift_joints, carry.cruise_joints))
                phases.append(DropDescentPhase(k, carry))
                phases.append(
                    ReleasePhase(
                        predrop_joints,
                        postdrop_branch.joints,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
                start_joints = g.grasp_joints
            elif completed_phase_name == "grasp":
                # Next is Lift. Do this short vertical clearance move from the
                # locked grasp pose before measuring again for the carry replan.
                phases.append(
                    RecoveryLiftPhase(k, g.grasp_joints, g.lift_joints)
                    if free_grasp
                    else LiftPhase(k, g.grasp_joints, g.lift_joints)
                )
                phases.append(CarryPhase(k, g.lift_joints, carry.cruise_joints))
                phases.append(DropDescentPhase(k, carry))
                phases.append(
                    ReleasePhase(
                        predrop_joints,
                        postdrop_branch.joints,
                        measured_gripper,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
                start_joints = g.grasp_joints
            elif completed_phase_name in ("lift", "recovery_lift"):
                # Now that the cube and jaws are safely above the floor, seed carry
                # from measured readback again.
                phases.append(CarryPhase(k, measured_joints, carry.cruise_joints))
                phases.append(DropDescentPhase(k, carry))
                phases.append(
                    ReleasePhase(
                        predrop_joints,
                        postdrop_branch.joints,
                        measured_gripper,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
            elif completed_phase_name == "drop_descent":
                # The executor normally runs release directly from the locked
                # drop endpoint, then replans retreat from elevated readback.
                # (There's no "carry" branch here: the executor always skips the
                # cruise-waypoint checkpoint straight into drop_descent, so this
                # function is never asked to replan from a completed "carry".)
                phases.append(
                    ReleasePhase(
                        measured_joints,
                        postdrop_branch.joints,
                        measured_gripper,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
            else:
                continue

            yield Trajectory(
                phases=tuple(phases),
                source=source,
                target=target,
                grasp=g,
                carry=carry,
                start_joints=start_joints,
                start_gripper=measured_gripper,
                end_joints=end_joints,
                end_gripper=end_gripper,
            )
