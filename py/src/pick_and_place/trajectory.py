# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Phases: (1) neutral -> hover, (2) hover -> grasp at cube center, (3) close
gripper to grasp, (4) lift and carry the grasped cube to a cruise waypoint above
the target, (5) descend from cruise into the canonical drop pose, (6) release,
lift clear, and flow back to neutral. The release is left to gravity: the
gripper set point opens and the cube falls on its own.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pick_and_place.geometry import (
    CubeFace,
    CubePose,
    CANONICAL_PREGRASP_DISTANCE,
    WORLD_UP,
    canonical_grasp_matrix,
    canonical_pregrasp_matrix,
)
from pick_and_place.ik import solve_simple_grasp_ik
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics
from pick_and_place import transforms as tf
from pick_and_place.transforms import Mat4, Vec3
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY,
    CUBE_PLACEMENT_OVERLAY,
    is_cube_drop_allowed,
)

# Number of intermediate heights checked along the hover→grasp descent when
# selecting a grasp. Catching joint-limit violations between endpoints prevents
# the arm from falling back to the joint lerp mid-descent.
_N_DESCENT_CHECKS = 8

# Tip-contact height of the source hover above the floor (clears the 3 cm cube
# top by 1 cm). At the grasp the tip sits at the cube-center height, so the
# world-z offset applied to the grasp is ``tip_z - pose.z``.
SOURCE_HOVER_TIP_Z = 0.04
# Recovery grasps may approach along an arbitrary tool axis. After closing, lift
# the held cube vertically to this world height before folding into the carry.
RECOVERY_LIFT_CUBE_Z = 0.08
# Cube-center height at release. Kept higher than the low simulated drop so
# the physical gripper stays clear of the floor with calibration/readback error.
DROP_CUBE_CENTER_Z = 0.045
RECOVERY_DROP_CUBE_CENTER_Z = 0.05
# Vertical lift after release, preserving the chosen drop orientation until the
# open jaws clear the cube.
POSTDROP_LIFT_Z = 0.04

# Full-range canonical grasp limits and search order.
MIN_CANONICAL_GRASP_RADIUS = CANONICAL_PICKUP_OVERLAY.inner_radius
MAX_CANONICAL_GRASP_RADIUS = CANONICAL_PICKUP_OVERLAY.outer_radius
MAX_RECOVERY_GRASP_RADIUS = CUBE_PLACEMENT_OVERLAY.outer_radius
MIN_CANONICAL_AZIMUTH = CANONICAL_PICKUP_OVERLAY.azimuth_min
MAX_CANONICAL_AZIMUTH = CANONICAL_PICKUP_OVERLAY.azimuth_max
# Lift the canonical side grip slightly above the cube center. At the far edge of
# the pickup sector, the tilted jaw would otherwise put its low collision box a
# few millimetres through the floor while still being IK-feasible.
CANONICAL_GRASP_Z_OFFSET = 0.005
_HORIZONTAL_GRASP_RADIUS = 0.36
_SQUARE_TOP_DOWN_PITCH = math.pi / 2.0
_CANONICAL_PITCHES = (
    _SQUARE_TOP_DOWN_PITCH,
    *(
        math.radians(deg)
        for deg in sorted(
            (deg for deg in range(10, 171, 2) if deg != 90),
            key=lambda deg: abs(deg - 90),
        )
    ),
)
_OUTER_HORIZONTAL_PITCHES = tuple(
    math.radians(deg)
    for deg in sorted(range(10, 61, 2), key=lambda deg: (abs(deg - 16), deg))
)
_CANONICAL_ROLL_OFFSETS = tuple(
    math.radians(deg) for deg in (0, -10, 10, -20, 20, -30, 30, -45, 45)
)


def _canonical_pitch_order(radius: float) -> tuple[float, ...]:
    if radius <= _HORIZONTAL_GRASP_RADIUS:
        return _CANONICAL_PITCHES
    return (
        *_OUTER_HORIZONTAL_PITCHES,
        *(pitch for pitch in _CANONICAL_PITCHES if pitch not in _OUTER_HORIZONTAL_PITCHES),
    )


def _roll_grasp_about_tool_axis(grasp: Mat4, roll_offset: float) -> Mat4:
    if roll_offset == 0.0:
        return grasp
    out = grasp.copy()
    out[:3, :3] = grasp[:3, :3] @ tf.rot_z(roll_offset)[:3, :3]
    return out

# Gripper joint angle at the hover grasp: 40 deg open.
GRIPPER_OPEN = math.radians(40.0)
# Gripper joint angle commanded during the grasp.
GRIPPER_GRASP = 0.10

NEUTRAL_ARM_JOINTS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": -math.pi / 2,
}
NEUTRAL_GRIPPER = 0.0

REST_ARM_JOINTS: dict[str, float] = {
    "shoulder_pan": math.radians(4.967032967032967),
    "shoulder_lift": math.radians(-95.16483516483517),
    "elbow_flex": math.radians(96.13186813186813),
    "wrist_flex": math.radians(73.71428571428571),
    "wrist_roll": math.radians(-86.46153846153847),
}
REST_GRIPPER = math.radians((10.5 - 2.3) / 96.2 * 130 - 10)


# --- Phase timing -----------------------------------------------------------
# Travel phases (approach, carry, retreat) derive their duration from the
# distance they cover, so the arm holds a roughly constant speed and longer
# moves simply take longer. Contact phases (descent onto the cube, gripper
# close, release dwell) keep fixed, gentle durations — there's no meaningful
# distance to scale and slowing them is what makes the grasp/release reliable.

# Angular speed of the fastest-moving joint through a joint-space move, rad/s.
# Governs phase 1 (approach) and the phase 5 retreat.
JOINT_SPEED = 1.5
# Cartesian speed of the gripper/cube tip along the carry, m/s. Governs phase 4.
CARTESIAN_SPEED = 0.45
# Floor on any speed-derived phase so short moves still have room to ease in/out.
MIN_TRAVEL_DURATION = 0.5

# Phase 2: fixed, gentle vertical descent from the hover onto the cube.
DESCENT_DURATION = 1.0
# Phase 3: close the gripper onto the cube. The follow-up lift is planned from
# the synthetic grasp pose and gets us clear before trusting readback again, so
# this can be short rather than a long contact dwell.
GRASP_DURATION = 0.35
# Hold the normal episode at its final carry pose before opening the gripper.
DROP_DWELL_DURATION = 0.5
# Phase 5a: fixed dwell at the drop hover while the gripper opens and the cube
# falls clear, before the retreat starts. Not a travel phase — the arm holds.
RELEASE_DURATION = 1.5
# The arm holds at the drop hover until the gripper has opened this far past the
# grasp, giving the released cube time to drop clear of the jaws before the arm
# starts moving (so the retreat doesn't fling it).
RETREAT_OPENING_ANGLE = math.radians(10.0)


# Points sampled along each joint-space move to measure the tip's Cartesian path.
_TIP_PATH_SAMPLES = 24


def _max_joint_travel(*waypoints: dict[str, float]) -> float:
    """Largest total angular travel of any single joint across ``waypoints``,
    measured the direct way each joint is lerped between consecutive poses."""
    return max(
        sum(abs(waypoints[i + 1][name] - waypoints[i][name]) for i in range(len(waypoints) - 1))
        for name in ARM_JOINT_NAMES
    )


def _tip_path_length(k: So101Kinematics, *waypoints: dict[str, float]) -> float:
    """Cartesian length of the gripper-tip path as the arm lerps straight through
    ``waypoints`` in joint space — the path the approach/retreat actually trace."""
    length = 0.0
    previous = k.tip_position(waypoints[0])
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for i in range(1, _TIP_PATH_SAMPLES + 1):
            point = k.tip_position(_lerp_joints(a, b, i / _TIP_PATH_SAMPLES))
            length += float(np.linalg.norm(point - previous))
            previous = point
    return length


def _joint_move_duration(k: So101Kinematics, *waypoints: dict[str, float]) -> float:
    """Duration of a joint-space move through ``waypoints``: long enough to hold
    the gripper tip at ``CARTESIAN_SPEED`` *and* keep every joint under
    ``JOINT_SPEED`` (the cap that bounds tip-static reconfigurations like a wrist
    roll). Floored at ``MIN_TRAVEL_DURATION``."""
    tip_time = _tip_path_length(k, *waypoints) / CARTESIAN_SPEED
    joint_time = _max_joint_travel(*waypoints) / JOINT_SPEED
    return max(MIN_TRAVEL_DURATION, tip_time, joint_time)


def _cartesian_move_duration(distance: float) -> float:
    """Duration of a Cartesian move of ``distance`` metres at ``CARTESIAN_SPEED``,
    floored at ``MIN_TRAVEL_DURATION``."""
    return max(MIN_TRAVEL_DURATION, distance / CARTESIAN_SPEED)


# Cube-center height of the level cruise. Above the predrop hover so the cube
# genuinely rises then descends; clears the cube top with room to spare
# mid-traverse.
CARRY_CRUISE_Z = 0.10
# Fraction of the carry spent smoothly accelerating in and decelerating out.
_CARRY_EASE_FRACTION = 0.2

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


def _shortest_delta(a: float, b: float) -> float:
    """Signed angular difference ``b - a`` wrapped to ``[-pi, pi]``."""
    d = (b - a) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


@dataclass(frozen=True)
class GraspChoice:
    """The face and elbow used to grasp the source cube, with the joint set
    points solved for the hover and the at-cube grasp on that branch."""

    face: CubeFace
    elbow: str
    pitch: float
    roll_offset: float
    closing_azimuth: float
    camera_outward: float
    hover_joints: dict[str, float]
    grasp_joints: dict[str, float]
    hover_matrix: Mat4
    grasp_matrix: Mat4
    lift_joints: dict[str, float]
    lift_matrix: Mat4
    inward_normal: Vec3


def _normalize_angle(angle: float) -> float:
    result = angle % (2.0 * math.pi)
    if result > math.pi:
        result -= 2.0 * math.pi
    if result <= -math.pi:
        result += 2.0 * math.pi
    return result


def _square_to_cube_face(nominal: float, cube_yaw: float) -> float:
    quarter = math.pi / 2.0
    return cube_yaw + round((nominal - cube_yaw) / quarter) * quarter


def _face_from_closing(closing_azimuth: float, cube_yaw: float) -> CubeFace:
    local = _normalize_angle(closing_azimuth - cube_yaw)
    index = int(round(local / (math.pi / 2.0))) % 4
    return ("+x", "+y", "-x", "-y")[index]


def _canonical_approach_vector(radial_azimuth: float, pitch: float) -> Vec3:
    horizontal = math.cos(pitch)
    return np.array(
        (
            math.cos(radial_azimuth) * horizontal,
            math.sin(radial_azimuth) * horizontal,
            -math.sin(pitch),
        )
    )


def _grasp_candidates(
    k: So101Kinematics,
    source: CubePose,
    *,
    max_radius: float,
    min_radius: float = MIN_CANONICAL_GRASP_RADIUS,
    min_azimuth: float = MIN_CANONICAL_AZIMUTH,
    max_azimuth: float = MAX_CANONICAL_AZIMUTH,
) -> Iterator[GraspChoice]:
    """Yield full-range canonical poses in preference order.

    ``source`` need not be a real cube: only its position and yaw are used, so
    the same search also produces canonical *drop* poses at a target point
    (see ``plan_carry_candidates``), with ``min_radius``/``max_radius``/
    ``min_azimuth``/``max_azimuth`` widened to the placement sector.

    The jaw-closing axis is perpendicular to the radial line from the pan axis
    and snapped to the nearest cube face. The approach starts square top-down and
    tilts only as far as needed to make both the contact grasp and the 3 cm
    pregrasp reachable, preferring the orientation with the wrist camera facing
    outward from the base.
    """
    radius = math.hypot(source.x - k.pan_axis[0], source.y - k.pan_axis[1])
    if (
        radius < min_radius - 1e-9
        or radius > max_radius + 1e-9
    ):
        return
    azimuth = math.atan2(source.y - k.pan_axis[1], source.x - k.pan_axis[0])
    if (
        azimuth < min_azimuth - 1e-9
        or azimuth > max_azimuth + 1e-9
    ):
        return

    closings = tuple(
        _square_to_cube_face(nominal, source.yaw)
        for nominal in (azimuth + math.pi / 2.0, azimuth - math.pi / 2.0)
    )
    radial = np.array((math.cos(azimuth), math.sin(azimuth), 0.0))
    first_reachable_pitch: float | None = None
    pending_inward: list[tuple[float, float, GraspChoice]] = []
    for pitch in _canonical_pitch_order(radius):
        approach = _canonical_approach_vector(azimuth, pitch)
        pitch_candidates: list[tuple[float, float, GraspChoice]] = []
        for closing in closings:
            base_grasp = canonical_grasp_matrix(source, closing, approach)
            unrolled_grasp = tf.with_position(
                base_grasp,
                tf.get_position(base_grasp) + WORLD_UP * CANONICAL_GRASP_Z_OFFSET,
            )
            face = _face_from_closing(closing, source.yaw)
            inward_normal = unrolled_grasp[:3, 0].copy()
            for roll_offset in _CANONICAL_ROLL_OFFSETS:
                grasp = _roll_grasp_about_tool_axis(unrolled_grasp, roll_offset)
                hover = canonical_pregrasp_matrix(grasp, approach, CANONICAL_PREGRASP_DISTANCE)
                recovery_lift = tf.with_position(
                    grasp,
                    tf.get_position(grasp)
                    + WORLD_UP * max(0.0, RECOVERY_LIFT_CUBE_Z - source.z),
                )
                grasp_branches = solve_simple_grasp_ik(k, grasp)
                hover_branches = solve_simple_grasp_ik(k, hover)
                lift_branches = solve_simple_grasp_ik(k, recovery_lift)
                if not grasp_branches or not hover_branches or not lift_branches:
                    continue
                camera_outward = float(np.dot(grasp[:3, 1], radial))
                for elbow in ("up", "down"):
                    grasp_branch = next((b for b in grasp_branches if b.elbow == elbow), None)
                    hover_branch = next((b for b in hover_branches if b.elbow == elbow), None)
                    lift_branch = next((b for b in lift_branches if b.elbow == elbow), None)
                    if grasp_branch is None or hover_branch is None or lift_branch is None:
                        continue
                    descent_ok = all(
                        any(
                            b.elbow == elbow
                            for b in solve_simple_grasp_ik(
                                k,
                                tf.with_position(
                                    grasp,
                                    tf.get_position(grasp)
                                    - approach
                                    * CANONICAL_PREGRASP_DISTANCE
                                    * (1.0 - i / _N_DESCENT_CHECKS),
                                ),
                            )
                        )
                        for i in range(1, _N_DESCENT_CHECKS)
                    )
                    if not descent_ok:
                        continue
                    lift_ok = all(
                        any(
                            b.elbow == elbow
                            for b in solve_simple_grasp_ik(
                                k,
                                tf.with_position(
                                    grasp,
                                    tf.get_position(grasp)
                                    + (tf.get_position(recovery_lift) - tf.get_position(grasp))
                                    * (i / _N_DESCENT_CHECKS),
                                ),
                            )
                        )
                        for i in range(1, _N_DESCENT_CHECKS)
                    )
                    if not lift_ok:
                        continue
                    pitch_candidates.append(
                        (
                            camera_outward,
                            abs(roll_offset),
                            GraspChoice(
                                face=face,
                                elbow=elbow,
                                pitch=pitch,
                                roll_offset=roll_offset,
                                closing_azimuth=closing,
                                camera_outward=camera_outward,
                                hover_joints=hover_branch.joints,
                                grasp_joints=grasp_branch.joints,
                                hover_matrix=hover,
                                grasp_matrix=grasp,
                                lift_joints=lift_branch.joints,
                                lift_matrix=recovery_lift,
                                inward_normal=inward_normal,
                            ),
                        ),
                    )

        if not pitch_candidates:
            continue
        if first_reachable_pitch is None:
            first_reachable_pitch = pitch
        pitch_candidates.sort(
            key=lambda item: (
                item[1] > 0.0,
                item[0] <= 0.0,
                0 if item[2].elbow == "up" else 1,
                item[1],
                -item[0],
            )
        )
        outward = [item for item in pitch_candidates if item[0] > 0.0]
        if outward:
            for _, _, candidate in outward:
                yield candidate
            for _, _, candidate in pending_inward:
                yield candidate
            return
        pending_inward.extend(pitch_candidates)

    if first_reachable_pitch is not None:
        for _, _, candidate in sorted(
            pending_inward,
            key=lambda item: (
                item[1] > 0.0,
                0 if item[2].elbow == "up" else 1,
                item[1],
                -item[0],
            ),
        ):
            yield candidate


def grasp_candidates(k: So101Kinematics, source: CubePose) -> Iterator[GraspChoice]:
    """Yield normal pickup grasps inside the smoke-tested canonical envelope."""
    yield from _grasp_candidates(k, source, max_radius=MAX_CANONICAL_GRASP_RADIUS)


def free_grasp_candidates(k: So101Kinematics, source: CubePose) -> Iterator[GraspChoice]:
    """Recovery can reach into the broader cleanup area."""
    yield from _grasp_candidates(k, source, max_radius=MAX_RECOVERY_GRASP_RADIUS)


def select_grasp(k: So101Kinematics, source: CubePose) -> GraspChoice:
    """Return the first IK-feasible grasp from ``grasp_candidates``."""
    candidate = next(grasp_candidates(k, source), None)
    if candidate is None:
        raise ValueError("No reachable grasp for the source cube")
    return candidate


def _smootherstep_integral(t: float) -> float:
    """Integral of smootherstep from 0 to ``t`` — distance travelled while speed
    ramps smoothly from zero to cruise speed."""
    c = min(1.0, max(0.0, t))
    return c**6 - 3.0 * c**5 + 2.5 * c**4


def _timed_arc_fraction(phase: float) -> float:
    """Arc-length fraction at a playback phase: smooth acceleration over the
    first window, constant speed through the middle, smooth deceleration at the
    end."""
    p = min(1.0, max(0.0, phase))
    ease = _CARRY_EASE_FRACTION
    total_area = 1.0 - ease
    if p < ease:
        return ease * _smootherstep_integral(p / ease) / total_area
    if p <= 1.0 - ease:
        return (ease * 0.5 + p - ease) / total_area
    return 1.0 - ease * _smootherstep_integral((1.0 - p) / ease) / total_area


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
    move can hit (see ``docs/GENERAL_CARRY_AND_DROP.txt``). The final approach
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


def _joint_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(
        sum(_shortest_delta(a[name], b[name]) ** 2 for name in ARM_JOINT_NAMES)
    )


# Fractions along the grasp->cruise->drop spline sampled for the carry-clearance
# check. Denser near the drop end, where the arm descends closest to the floor.
_CARRY_CLEARANCE_SAMPLE_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_DESCENT_CLEARANCE_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


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
        joints = _drop_descent_joints(
            k, cruise_matrix, drop_matrix, cruise_joints, drop_joints, elbow, fraction
        )
        if not carry_ok(joints):
            return False
    return True


def _drop_descent_joints(
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
    for drop in _grasp_candidates(k, drop_pose, max_radius=MAX_RECOVERY_GRASP_RADIUS):
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
        alpha = _smoothstep(t / self.duration) if self.duration > 0 else 1.0
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
        alpha = _smoothstep(t / self.duration) if self.duration > 0 else 1.0
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
        alpha = _smoothstep(t / self.duration) if self.duration > 0 else 1.0
        joints = _drop_descent_joints(
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
            + (self.end_gripper - self.start_gripper) * _smoothstep(alpha),
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
    drop_cube_center_z = RECOVERY_DROP_CUBE_CENTER_Z if free_grasp else DROP_CUBE_CENTER_Z
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

    drop_cube_center_z = RECOVERY_DROP_CUBE_CENTER_Z if free_grasp else DROP_CUBE_CENTER_Z
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
