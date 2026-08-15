# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Carry the grasped cube to a cruise pose above the target.

Between the lift and the drop the cube is held, so the arm may not swing
through anything on the way: candidate carries are sampled along their own path
and rejected if any sampled configuration collides. The drop height is not
constant either — reaching further out costs the arm elevation, so the release
happens lower the further the target sits from the base.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from pick_and_place.core import transforms as tf
from pick_and_place.core.geometry import CubePose, WORLD_UP
from pick_and_place.core.ik import solve_simple_grasp_ik
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.core.transforms import Mat4, Vec3
from pick_and_place.core.workspace_bounds import (
    CUBE_PLACEMENT_SECTOR,
    PAN_AXIS,
    is_cube_drop_allowed,
)
from pick_and_place.scripted.grasp import GraspChoice, free_grasp_candidates
from pick_and_place.scripted.motion import _lerp_joints


# Cube-center height of the level cruise. Above the predrop hover so the cube
# genuinely rises then descends; clears the cube top with room to spare
# mid-traverse.
CARRY_CRUISE_Z = 0.10


# Cube-center height at release. Kept higher than the low simulated drop so
# the physical gripper stays clear of the floor with calibration/readback error.
# For a normal drop this is the reference height, adjusted by target radius in
# ``nominal_drop_center_z``: close to the base the arm places accurately from low
# down, but fully extended (far out) it can't safely reach as low, so the release
# rises. Near the placement zone's inner radius it drops 3 cm below this (down to
# the cube's resting height, i.e. a set-down), and at the outer radius it sits
# 1 cm above.
DROP_CUBE_CENTER_Z = 0.045


NEAR_DROP_CUBE_CENTER_Z = DROP_CUBE_CENTER_Z - 0.03


FAR_DROP_CUBE_CENTER_Z = DROP_CUBE_CENTER_Z + 0.01


# Fractions along the grasp->cruise->drop spline sampled for the carry-clearance
# check. Denser near the drop end, where the arm descends closest to the floor.
_CARRY_CLEARANCE_SAMPLE_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


_DESCENT_CLEARANCE_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


# Checks a single arm joint configuration for unexpected collisions (e.g. jaw
# or wrist-camera geometry against the floor or a fixture). Returns True if the
# configuration is clear. Left as an injected callback rather than importing
# MuJoCo collision helpers directly here, since those live in episodes.py and
# importing them back would create a cycle.
CarryJointChecker = Callable[[dict[str, float]], bool]


@dataclass(frozen=True)
class CarryPlan:
    """Carry from lifted grasp to a selected canonical drop pose.

    The long-distance transit (lift -> cruise) is a joint-space move: always a
    valid arm configuration, immune to the IK dead zones a fixed-elbow Cartesian
    move can hit. The final approach
    (cruise -> drop) is a short Cartesian descent instead, mirroring
    ``DescentPhase`` on the pickup side, so the height into the drop is
    genuinely controlled rather than an incidental side effect of a joint blend.
    """

    mode: str
    elbow: str
    grasp_position: Vec3
    drop_position: Vec3
    # World-from-gripper matrices of the cruise waypoint and the chosen
    # canonical drop pose, both at the target's xy.
    cruise_matrix: Mat4
    drop_matrix: Mat4
    grasp_joints: dict[str, float]
    cruise_joints: dict[str, float]
    drop_joints: dict[str, float]


def _carry_path_clear(
    carry_ok: CarryJointChecker,
    grasp_joints: dict[str, float],
    cruise_joints: dict[str, float],
    elbow: str,
    cruise_matrix: Mat4,
    drop_matrix: Mat4,
    drop_joints: dict[str, float],
    k: So101Kinematics,
) -> bool:
    """Check the same two moves the carry will actually play back -- the
    joint-space lift->cruise transit and the Cartesian cruise->drop descent --
    sampled at a handful of fractions, against a caller-supplied collision
    checker."""
    for fraction in _CARRY_CLEARANCE_SAMPLE_FRACTIONS:
        if not carry_ok(_lerp_joints(grasp_joints, cruise_joints, fraction)):
            return False
    for fraction in _DESCENT_CLEARANCE_SAMPLE_FRACTIONS:
        joints = drop_descent_joints(
            k, cruise_matrix, drop_matrix, cruise_joints, drop_joints, elbow, fraction
        )
        if not carry_ok(joints):
            return False
    return True


def drop_descent_joints(
    k: So101Kinematics,
    cruise_matrix: Mat4,
    drop_matrix: Mat4,
    cruise_joints: dict[str, float],
    drop_joints: dict[str, float],
    elbow: str,
    alpha: float,
) -> dict[str, float]:
    """Joints for the Cartesian straight-line descent from ``cruise_matrix`` to
    ``drop_matrix`` at ``alpha`` in [0, 1], falling back to a joint lerp for any
    sample where IK doesn't return the requested elbow branch."""
    matrix = tf.with_position(
        cruise_matrix,
        tf.get_position(cruise_matrix)
        + (tf.get_position(drop_matrix) - tf.get_position(cruise_matrix)) * alpha,
    )
    branch = next(
        (b for b in solve_simple_grasp_ik(k, matrix) if b.elbow == elbow), None
    )
    return branch.joints if branch is not None else _lerp_joints(cruise_joints, drop_joints, alpha)


def nominal_drop_center_z(target: CubePose) -> float:
    """Cube-center release height for a drop, rising with target radius.

    Close to the base the arm places accurately from low down; fully extended it
    can't safely reach as low, so the release height climbs linearly with radius
    from ``NEAR_DROP_CUBE_CENTER_Z`` at the placement zone's inner radius to
    ``FAR_DROP_CUBE_CENTER_Z`` at its outer radius, clamped outside that band.
    """
    radius = math.hypot(target.x - PAN_AXIS[0], target.y - PAN_AXIS[1])
    r_near = CUBE_PLACEMENT_SECTOR.inner_radius
    r_far = CUBE_PLACEMENT_SECTOR.outer_radius
    frac = min(1.0, max(0.0, (radius - r_near) / (r_far - r_near)))
    return NEAR_DROP_CUBE_CENTER_Z + frac * (FAR_DROP_CUBE_CENTER_Z - NEAR_DROP_CUBE_CENTER_Z)


def plan_carry_candidates(
    k: So101Kinematics,
    grasp: GraspChoice,
    target: CubePose,
    *,
    drop_cube_center_z: float = DROP_CUBE_CENTER_Z,
    carry_ok: CarryJointChecker | None = None,
) -> Iterator[CarryPlan]:
    """Plan joint-space carries for an already-chosen grasp.

    The drop is a single canonical pose at the target, in the same family and
    preference order as a canonical grasp (``_grasp_candidates``), just aimed
    at ``target`` instead of a real cube. Once the jaws close, the held cube's
    orientation is a rigid, irrelevant don't-care, so there's no drop-side
    orientation search: whichever face-on/top-down pose reaches the target is
    the drop.
    """
    if not is_cube_drop_allowed(target.x, target.y):
        return
    grasp_position = tf.get_position(grasp.lift_matrix)
    drop_position = np.array((target.x, target.y, drop_cube_center_z))
    drop_pose = CubePose(x=target.x, y=target.y, z=drop_cube_center_z, yaw=0.0)

    # Prefer the nominal cruise height (best floor/frame clearance along the
    # carry), but some orientations are only IK-reachable lower down -- e.g. a
    # side grasp held level loses reachability at some target azimuths well
    # before it reaches CARRY_CRUISE_Z. Falling back to a lower height there
    # avoids discarding an otherwise-ideal (low joint-cost) orientation in
    # favour of one that needlessly reconfigures the arm just to clear cruise.
    cruise_heights = sorted(
        {max(h, drop_cube_center_z) for h in (CARRY_CRUISE_Z, 0.09, 0.08, 0.07)},
        reverse=True,
    )

    # Widen only the outer radius to the placement zone's, mirroring
    # ``free_grasp_candidates``: the canonical family's proven envelope for
    # radius/azimuth otherwise stays at the pickup bounds. The placement
    # zone's own (smaller) inner radius and (wider) azimuth were tuned for the
    # old fully-flexible SO(3) drop search, not the canonical-pose family --
    # verified empirically that the canonical family finds zero candidates
    # between the placement and pickup inner radii.
    for drop in free_grasp_candidates(k, drop_pose):
        cruise_branch = None
        for cruise_z in cruise_heights:
            # Raise the drop pose's *existing* position by the height delta,
            # rather than overwriting it outright: ``drop.grasp_matrix``'s
            # position is the wrist origin, which for a near-top-down grasp
            # sits well above the target contact point (the jaw-tip-to-wrist
            # offset baked into ``canonical_grasp_matrix``). Overwriting it
            # with a raw ``(target.x, target.y, cruise_z)`` would place the
            # *wrist* at cruise height, leaving the jaw tip far too low.
            cruise_matrix = tf.with_position(
                drop.grasp_matrix,
                tf.get_position(drop.grasp_matrix)
                + WORLD_UP * (cruise_z - drop_cube_center_z),
            )
            cruise_branch = next(
                (
                    b
                    for b in solve_simple_grasp_ik(k, cruise_matrix)
                    if b.elbow == drop.elbow
                ),
                None,
            )
            if cruise_branch is not None:
                break
        if cruise_branch is None:
            continue
        if carry_ok is not None and not _carry_path_clear(
            carry_ok,
            grasp.lift_joints,
            cruise_branch.joints,
            drop.elbow,
            cruise_matrix,
            drop.grasp_matrix,
            drop.grasp_joints,
            k,
        ):
            continue
        yield CarryPlan(
            mode="joint",
            elbow=drop.elbow,
            grasp_position=grasp_position,
            drop_position=drop_position,
            cruise_matrix=cruise_matrix,
            drop_matrix=drop.grasp_matrix,
            grasp_joints=dict(grasp.lift_joints),
            cruise_joints=dict(cruise_branch.joints),
            drop_joints=dict(drop.grasp_joints),
        )
