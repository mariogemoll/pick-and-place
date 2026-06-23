# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Phases: (1) neutral -> hover, (2) hover -> grasp at cube center, (3) close
gripper to grasp, (4) lift and carry the grasped cube up and over to the hover
above the target, (5) release, lift clear, and flow back to neutral. The release
is left to gravity: the gripper set point opens and the cube falls on its own.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from typing import Literal, Protocol, runtime_checkable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pick_and_place.geometry import (
    CubeFace,
    CubePose,
    GRIPPER_TARGET_POSITION,
    CANONICAL_PREGRASP_DISTANCE,
    SAFETY_MARGIN,
    WORLD_UP,
    canonical_grasp_matrix,
    canonical_pregrasp_matrix,
    world_from_cube,
)
from pick_and_place.ik import solve_simple_grasp_ik
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics
from pick_and_place import transforms as tf
from pick_and_place.transforms import Mat4, Vec3
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY,
    CUBE_PLACEMENT_OVERLAY,
    is_cube_drop_allowed,
    is_vertical_grip_allowed,
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
# Cube-center height at release. Normal episodes release slightly higher than
# cleanup recovery, where keeping the established low drop is more important.
DROP_CUBE_CENTER_Z = 0.05
RECOVERY_DROP_CUBE_CENTER_Z = 0.03
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
# Metres of carry arc that one radian of cube yaw is counted as when retiming the
# carry. Set so a pure in-place rotation is paced at JOINT_SPEED rather than
# whipping the wrist through the low-translation stretch of the path.
CARRY_ROTATION_WEIGHT = CARTESIAN_SPEED / JOINT_SPEED
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


# Cube-center height of the level cruise. Above the predrop hover (2 cm) so the
# cube genuinely rises then descends; clears the cube top with room to spare
# mid-traverse.
CARRY_CRUISE_Z = 0.08
# The side-view carry is one C2 spline through four waypoints: leave the pick
# vertically, round into a level cruise, hold the cruise, round down and arrive
# vertically. The waypoint phases below place the cruise in the middle 20 %.
_CARRY_WAYPOINT_PHASES = (0.0, 0.4, 0.6, 1.0)
# Horizontal travel fraction spent rounding into / out of the cruise.
_CARRY_CORNER_TRAVEL = 0.25
# How many points along the carry to check for reachability when planning it.
_CARRY_SAMPLES = 24
# Angular resolution of the free drop-pitch search.
_DROP_PITCH_SAMPLES = 16
_VERTICAL_DROP_TOOL_PITCH = -math.pi / 2.0
# In the vertical annulus, prefer near-vertical drops but allow a little pitch
# freedom so the wrist/camera mount can stay near the neutral -90 deg roll.
_VERTICAL_DROP_PITCH_OFFSETS = (
    0.0,
    math.radians(-5.0),
    math.radians(5.0),
    math.radians(-10.0),
    math.radians(10.0),
    math.radians(-15.0),
    math.radians(15.0),
)
# Resolution of the arc-length table used to retime the geometric curve.
_CARRY_ARC_LENGTH_SAMPLES = 2048
# Fraction of the carry spent smoothly accelerating in and decelerating out.
_CARRY_EASE_FRACTION = 0.2

DropOrientation = Literal["free", "target"]


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


def _spline_joints_through_waypoint(
    start: dict[str, float],
    waypoint: dict[str, float],
    end: dict[str, float],
    waypoint_phase: float,
    phase: float,
) -> dict[str, float]:
    """C2 joint spline from ``start`` through a non-stopping ``waypoint`` to ``end``.

    Both segments share the waypoint velocity and have zero acceleration at the
    join, so the arm flows through the clearance hover without pausing.
    """
    p = min(1.0, max(0.0, phase))
    if p <= 0.0:
        return dict(start)
    if p >= 1.0:
        return dict(end)
    out: dict[str, float] = {}
    for name in ARM_JOINT_NAMES:
        waypoint_velocity = 0.5 * (end[name] - waypoint[name]) / (1.0 - waypoint_phase)
        if p <= waypoint_phase:
            out[name] = _quintic_hermite(
                start[name],
                waypoint[name],
                0.0,
                waypoint_velocity,
                waypoint_phase,
                p / waypoint_phase,
            )
        else:
            out[name] = _quintic_hermite(
                waypoint[name],
                end[name],
                waypoint_velocity,
                0.0,
                1.0 - waypoint_phase,
                (p - waypoint_phase) / (1.0 - waypoint_phase),
            )
    return out


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
) -> Iterator[GraspChoice]:
    """Yield full-range canonical grasps in preference order.

    The jaw-closing axis is perpendicular to the radial line from the pan axis
    and snapped to the nearest cube face. The approach starts square top-down and
    tilts only as far as needed to make both the contact grasp and the 3 cm
    pregrasp reachable, preferring the orientation with the wrist camera facing
    outward from the base.
    """
    radius = math.hypot(source.x - k.pan_axis[0], source.y - k.pan_axis[1])
    if (
        radius < MIN_CANONICAL_GRASP_RADIUS - 1e-9
        or radius > max_radius + 1e-9
    ):
        return
    azimuth = math.atan2(source.y - k.pan_axis[1], source.x - k.pan_axis[0])
    if (
        azimuth < MIN_CANONICAL_AZIMUTH - 1e-9
        or azimuth > MAX_CANONICAL_AZIMUTH + 1e-9
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


def _quintic_hermite(
    start: float,
    end: float,
    start_velocity: float,
    end_velocity: float,
    duration: float,
    u: float,
) -> float:
    """Quintic Hermite with derivatives expressed against the geometry parameter.

    Matching position, velocity, and acceleration at each end makes adjacent
    segments C2. Waypoint acceleration is zero, while internal waypoint velocity
    stays nonzero so the carry flows through rather than pausing.
    """
    v0 = start_velocity * duration
    v1 = end_velocity * duration
    delta = end - start - v0
    velocity_delta = v1 - v0
    c3 = 10.0 * delta - 4.0 * velocity_delta
    c4 = -15.0 * delta + 7.0 * velocity_delta
    c5 = 6.0 * delta - 3.0 * velocity_delta
    return start + v0 * u + c3 * u**3 + c4 * u**4 + c5 * u**5


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


@dataclass(frozen=True)
class CarryPlan:
    """The geometric carry of the grasped cube from the pick to the target hover.

    The cube travels either as a straight Cartesian chord (shortest) or, when
    that chord would leave the annular workspace, as a polar arc about the pan
    axis (radius and azimuth swept between the endpoints, which keeps the path
    inside the sector by construction). The target specifies position only; the
    final cube orientation is selected by the position-only drop planner.
    """

    mode: str  # 'straight' or 'polar'
    # Rigid cube→gripper transform captured at the grasp; the gripper follows the
    # cube through the carry so the held cube stays flush and lands on target.
    cube_from_gripper: Mat4
    pan_axis: Vec3
    grasp_position: Vec3
    drop_position: Vec3
    grasp_rotation: np.ndarray
    drop_rotation: np.ndarray
    grasp_radius: float
    drop_radius: float
    grasp_azimuth: float
    drop_azimuth: float
    # (parameter, generalized arc length) samples for retiming the curve. The
    # length includes translation and the planner-selected orientation sweep.
    arc_table: tuple[tuple[float, float], ...] = ()

    @property
    def drop_yaw(self) -> float:
        """World yaw of the planner-selected (possibly tilted) drop pose."""
        return float(Rotation.from_matrix(self.drop_rotation).as_euler("xyz")[2])


def _carry_path(grasp_z: float, drop_z: float, parameter: float) -> tuple[float, float]:
    """Side-view carry path: travel fraction in [0, 1] and world-z height.

    The endpoint tangent velocities are purely vertical and the internal tangent
    velocities purely horizontal, producing one smooth rounded ascent and
    descent through the level cruise.
    """
    p = min(1.0, max(0.0, parameter))
    ascent_velocity = (CARRY_CRUISE_Z - grasp_z) * 2.0
    descent_velocity = (drop_z - CARRY_CRUISE_Z) * 2.0
    cruise_velocity = 1.0
    # (travel, height, travel_velocity, height_velocity) at each waypoint.
    points = (
        (0.0, grasp_z, 0.0, ascent_velocity),
        (_CARRY_CORNER_TRAVEL, CARRY_CRUISE_Z, cruise_velocity, 0.0),
        (1.0 - _CARRY_CORNER_TRAVEL, CARRY_CRUISE_Z, cruise_velocity, 0.0),
        (1.0, drop_z, 0.0, descent_velocity),
    )
    end_index = next(i for i, phase in enumerate(_CARRY_WAYPOINT_PHASES) if p <= phase)
    i = max(0, end_index - 1)
    start_phase = _CARRY_WAYPOINT_PHASES[i]
    duration = _CARRY_WAYPOINT_PHASES[i + 1] - start_phase
    u = (p - start_phase) / duration
    s_travel, s_height, s_tv, s_hv = points[i]
    e_travel, e_height, e_tv, e_hv = points[i + 1]
    travel = _quintic_hermite(s_travel, e_travel, s_tv, e_tv, duration, u)
    height = _quintic_hermite(s_height, e_height, s_hv, e_hv, duration, u)
    return travel, height


def _carry_geometry_pose(plan: CarryPlan, parameter: float) -> tuple[Vec3, np.ndarray]:
    """World position and rotation of the carried cube."""
    travel, height = _carry_path(plan.grasp_position[2], plan.drop_position[2], parameter)
    if plan.mode == "straight":
        x = plan.grasp_position[0] + (plan.drop_position[0] - plan.grasp_position[0]) * travel
        y = plan.grasp_position[1] + (plan.drop_position[1] - plan.grasp_position[1]) * travel
    else:
        radius = plan.grasp_radius + (plan.drop_radius - plan.grasp_radius) * travel
        azimuth = plan.grasp_azimuth + (plan.drop_azimuth - plan.grasp_azimuth) * travel
        x = plan.pan_axis[0] + radius * math.cos(azimuth)
        y = plan.pan_axis[1] + radius * math.sin(azimuth)
    rotations = Rotation.from_matrix(np.stack((plan.grasp_rotation, plan.drop_rotation)))
    rotation = Slerp((0.0, 1.0), rotations)(min(1.0, max(0.0, travel))).as_matrix()
    return np.array((x, y, height)), rotation


def _carry_geometry_matrix(plan: CarryPlan, parameter: float) -> Mat4:
    """World cube pose at a geometry parameter. This defines shape only; playback
    timing is applied separately by ``_carry_cube_matrix``."""
    position, rotation = _carry_geometry_pose(plan, parameter)
    matrix = tf.identity()
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def _build_arc_table(plan: CarryPlan) -> tuple[tuple[float, float], ...]:
    """Cumulative generalized arc length along the carry."""
    table: list[tuple[float, float]] = [(0.0, 0.0)]
    previous_position, previous_rotation = _carry_geometry_pose(plan, 0.0)
    length = 0.0
    for i in range(1, _CARRY_ARC_LENGTH_SAMPLES + 1):
        parameter = i / _CARRY_ARC_LENGTH_SAMPLES
        position, rotation = _carry_geometry_pose(plan, parameter)
        translation = float(np.linalg.norm(position - previous_position))
        angle = Rotation.from_matrix(previous_rotation.T @ rotation).magnitude()
        length += translation + angle * CARRY_ROTATION_WEIGHT
        table.append((parameter, length))
        previous_position, previous_rotation = position, rotation
    return tuple(table)


def _length_to_parameter(table: tuple[tuple[float, float], ...], length: float) -> float:
    clamped = min(table[-1][1], max(0.0, length))
    end = next((j for j, sample in enumerate(table) if sample[1] >= clamped), len(table) - 1)
    b = table[max(1, end)]
    a = table[max(0, end - 1)]
    span = b[1] - a[1]
    alpha = 0.0 if span == 0 else (clamped - a[1]) / span
    return a[0] + (b[0] - a[0]) * alpha


def _carry_cube_matrix(plan: CarryPlan, phase: float) -> Mat4:
    """Traverse the geometric curve by arc length, with one global C2
    ease-in/out. Speed therefore changes only at the start and end, not at the
    waypoints."""
    target_length = _timed_arc_fraction(phase) * plan.arc_table[-1][1]
    return _carry_geometry_matrix(plan, _length_to_parameter(plan.arc_table, target_length))


def _pushed_cube(pose: CubePose, inward_normal: Vec3, push: float) -> CubePose:
    """Cube translated ``push`` metres toward the fixed jaw (opposite the face's
    inward normal). Faces are vertical, so this only moves x/y."""
    if push == 0.0:
        return pose
    return dataclasses.replace(
        pose,
        x=pose.x - float(inward_normal[0]) * push,
        y=pose.y - float(inward_normal[1]) * push,
        z=pose.z - float(inward_normal[2]) * push,
    )


def plan_carry_candidates(
    k: So101Kinematics,
    grasp: GraspChoice,
    source: CubePose,
    target: CubePose,
    *,
    drop_orientation: DropOrientation = "free",
    drop_cube_center_z: float = DROP_CUBE_CENTER_Z,
) -> Iterator[CarryPlan]:
    """Plan the carry for an already-chosen grasp (face + elbow).

    ``drop_orientation="free"`` treats the target as position-only and searches
    the drop tool pitch. ``"target"`` preserves the target cube orientation.
    When the target is inside the vertical pickup/drop zone, near-vertical drops
    are yielded before other free-pitch drops; if preflight rejects them, callers
    can still fall through to the broader free-pitch search.
    """
    grasp_gripper = grasp.grasp_matrix
    if not is_cube_drop_allowed(target.x, target.y):
        return
    low_grasp_cube = world_from_cube(_pushed_cube(source, grasp.inward_normal, SAFETY_MARGIN))
    cube_from_gripper = np.linalg.inv(low_grasp_cube) @ grasp_gripper
    lifted_grasp_cube = grasp.lift_matrix @ np.linalg.inv(cube_from_gripper)

    target_xy = (target.x, target.y)
    drop_position = np.array((*target_xy, drop_cube_center_z))

    grasp_position = tf.get_position(lifted_grasp_cube)
    base = {
        "cube_from_gripper": cube_from_gripper,
        "pan_axis": np.asarray(k.pan_axis, dtype=np.float64),
        "grasp_position": grasp_position,
        "drop_position": drop_position,
        "grasp_radius": math.hypot(
            grasp_position[0] - k.pan_axis[0], grasp_position[1] - k.pan_axis[1]
        ),
        "drop_radius": math.hypot(
            drop_position[0] - k.pan_axis[0], drop_position[1] - k.pan_axis[1]
        ),
        "grasp_azimuth": math.atan2(
            grasp_position[1] - k.pan_axis[1], grasp_position[0] - k.pan_axis[0]
        ),
        "drop_azimuth": math.atan2(
            drop_position[1] - k.pan_axis[1], drop_position[0] - k.pan_axis[0]
        ),
    }
    plans: list[tuple[tuple[int, float, float, int], CarryPlan]] = []
    target_allows_vertical_drop = drop_orientation == "free" and is_vertical_grip_allowed(
        target.x, target.y
    )

    def consider_plan(plan: CarryPlan, mode_index: int, priority: int) -> None:
        max_wrist_distance = 0.0
        wrist_cost = 0.0
        for i in range(_CARRY_SAMPLES + 1):
            gripper_matrix = _carry_geometry_matrix(plan, i / _CARRY_SAMPLES) @ cube_from_gripper
            branch = next(
                (b for b in solve_simple_grasp_ik(k, gripper_matrix) if b.elbow == grasp.elbow),
                None,
            )
            if branch is None:
                return
            wrist_distance = abs(
                _shortest_delta(NEUTRAL_ARM_JOINTS["wrist_roll"], branch.joints["wrist_roll"])
            )
            max_wrist_distance = max(max_wrist_distance, wrist_distance)
            wrist_cost += wrist_distance * wrist_distance
        score = (priority, wrist_cost / (_CARRY_SAMPLES + 1), max_wrist_distance, mode_index)
        plans.append((score, plan))

    def drop_gripper_matrix(tool_pitch: float, wrist_roll: float) -> Mat4:
        """Construct a kinematically compatible gripper pose at the drop center."""
        azimuth = math.atan2(target_xy[1] - k.pan_axis[1], target_xy[0] - k.pan_axis[0])
        cube_from_gripper_rotation = cube_from_gripper[:3, :3]
        cube_from_gripper_offset = cube_from_gripper[:3, 3]
        matrix = tf.identity()
        for _ in range(6):
            radial = np.array((math.cos(azimuth), math.sin(azimuth), 0.0))
            lateral = np.array((-math.sin(azimuth), math.cos(azimuth), 0.0))
            approach = radial * math.cos(tool_pitch) + WORLD_UP * math.sin(tool_pitch)
            zero_roll_x = np.cross(approach, lateral)
            zero_roll_x /= np.linalg.norm(zero_roll_x)
            roll = wrist_roll + k.wrist_roll_zero_twist
            gripper_x = zero_roll_x * math.cos(roll) + lateral * math.sin(roll)
            gripper_z = -approach
            gripper_y = np.cross(gripper_z, gripper_x)
            gripper_rotation = np.column_stack((gripper_x, gripper_y, gripper_z))
            cube_rotation = gripper_rotation @ cube_from_gripper_rotation.T
            matrix[:3, :3] = gripper_rotation
            matrix[:3, 3] = drop_position + cube_rotation @ cube_from_gripper_offset
            target = tf.transform_point(matrix, GRIPPER_TARGET_POSITION)
            azimuth = math.atan2(target[1] - k.pan_axis[1], target[0] - k.pan_axis[0])
        return matrix

    def evaluate_free_drop(mode: str, tool_pitch: float, mode_index: int) -> None:
        drop_gripper = drop_gripper_matrix(tool_pitch, NEUTRAL_ARM_JOINTS["wrist_roll"])
        drop_cube = drop_gripper @ np.linalg.inv(cube_from_gripper)
        vertical_offset = abs(_shortest_delta(_VERTICAL_DROP_TOOL_PITCH, tool_pitch))
        is_near_vertical = vertical_offset <= max(abs(x) for x in _VERTICAL_DROP_PITCH_OFFSETS)
        priority = 0 if target_allows_vertical_drop and is_near_vertical else 1
        consider_plan(
            CarryPlan(
                mode=mode,
                grasp_rotation=lifted_grasp_cube[:3, :3],
                drop_rotation=drop_cube[:3, :3],
                **base,
            ),
            mode_index,
            priority,
        )

    def evaluate_target_drop(mode: str, mode_index: int) -> None:
        drop_cube = world_from_cube(dataclasses.replace(target, z=drop_cube_center_z))
        consider_plan(
            CarryPlan(
                mode=mode,
                grasp_rotation=lifted_grasp_cube[:3, :3],
                drop_rotation=drop_cube[:3, :3],
                **base,
            ),
            mode_index,
            0,
        )

    for mode_index, mode in enumerate(("straight", "polar")):
        if drop_orientation == "free":
            if target_allows_vertical_drop:
                pitch_samples = tuple(
                    _VERTICAL_DROP_TOOL_PITCH + offset for offset in _VERTICAL_DROP_PITCH_OFFSETS
                )
            else:
                pitch_samples = np.linspace(-math.pi, math.pi, _DROP_PITCH_SAMPLES, endpoint=False)
            for tool_pitch in pitch_samples:
                evaluate_free_drop(mode, float(tool_pitch), mode_index)
        elif drop_orientation == "target":
            evaluate_target_drop(mode, mode_index)
        else:
            raise ValueError(f"unknown drop_orientation: {drop_orientation}")

    for _, plan in sorted(plans, key=lambda item: item[0]):
        yield dataclasses.replace(plan, arc_table=_build_arc_table(plan))


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
    k: So101Kinematics
    carry: CarryPlan
    elbow: str
    grasp_joints: dict[str, float]
    predrop_joints: dict[str, float]

    @property
    def name(self) -> str:
        return "carry"

    @cached_property
    def duration(self) -> float:
        return _cartesian_move_duration(self.carry.arc_table[-1][1])

    def evaluate(self, t: float) -> Frame:
        phase = min(1.0, t / self.duration) if self.duration > 0 else 1.0
        gripper_matrix = _carry_cube_matrix(self.carry, phase) @ self.carry.cube_from_gripper
        branch = next(
            (b for b in solve_simple_grasp_ik(self.k, gripper_matrix) if b.elbow == self.elbow),
            None,
        )
        joints = (
            branch.joints
            if branch is not None
            else _lerp_joints(self.grasp_joints, self.predrop_joints, _smoothstep(phase))
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
    drop_orientation: DropOrientation = "free"
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
    drop_orientation: DropOrientation = "free",
    free_grasp: bool = False,
) -> Iterator[Trajectory]:
    """Yield full trajectories from start to end in grasp preference order."""
    candidates = free_grasp_candidates(k, source) if free_grasp else grasp_candidates(k, source)
    drop_cube_center_z = RECOVERY_DROP_CUBE_CENTER_Z if free_grasp else DROP_CUBE_CENTER_Z
    release_delay = 0.0 if free_grasp else DROP_DWELL_DURATION
    for grasp in candidates:
        for carry in plan_carry_candidates(
            k,
            grasp,
            source,
            target,
            drop_orientation=drop_orientation,
            drop_cube_center_z=drop_cube_center_z,
        ):
            endpoint = _carry_geometry_matrix(carry, 1.0) @ carry.cube_from_gripper
            predrop_branch = next(
                (b for b in solve_simple_grasp_ik(k, endpoint) if b.elbow == grasp.elbow), None
            )
            if predrop_branch is None:
                continue
            postdrop_hover = tf.with_position(
                endpoint, tf.get_position(endpoint) + np.array((0.0, 0.0, POSTDROP_LIFT_Z))
            )
            postdrop_branch = next(
                (b for b in solve_simple_grasp_ik(k, postdrop_hover) if b.elbow == grasp.elbow),
                None,
            )
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
                CarryPhase(k, carry, grasp.elbow, grasp.lift_joints, predrop_branch.joints),
                ReleasePhase(
                    predrop_branch.joints,
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
                drop_orientation=drop_orientation,
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
    drop_orientation: DropOrientation = "free",
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
            drop_orientation,
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
            drop_orientation,
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
            source,
            target,
            drop_orientation=drop_orientation,
            drop_cube_center_z=drop_cube_center_z,
        ):
            endpoint = _carry_geometry_matrix(carry, 1.0) @ carry.cube_from_gripper
            predrop_branch = next(
                (b for b in solve_simple_grasp_ik(k, endpoint) if b.elbow == g.elbow), None
            )
            if predrop_branch is None:
                continue

            postdrop_hover = tf.with_position(
                endpoint, tf.get_position(endpoint) + np.array((0.0, 0.0, POSTDROP_LIFT_Z))
            )

            postdrop_branch = next(
                (b for b in solve_simple_grasp_ik(k, postdrop_hover) if b.elbow == g.elbow), None
            )
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
                phases.append(CarryPhase(k, carry, g.elbow, g.lift_joints, predrop_branch.joints))
                phases.append(
                    ReleasePhase(
                        predrop_branch.joints,
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
                phases.append(CarryPhase(k, carry, g.elbow, g.lift_joints, predrop_branch.joints))
                phases.append(
                    ReleasePhase(
                        predrop_branch.joints,
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
                phases.append(CarryPhase(k, carry, g.elbow, g.lift_joints, predrop_branch.joints))
                phases.append(
                    ReleasePhase(
                        predrop_branch.joints,
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
                phases.append(CarryPhase(k, carry, g.elbow, g.lift_joints, predrop_branch.joints))
                phases.append(
                    ReleasePhase(
                        predrop_branch.joints,
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
                phases.append(CarryPhase(k, carry, g.elbow, measured_joints, predrop_branch.joints))
                phases.append(
                    ReleasePhase(
                        predrop_branch.joints,
                        postdrop_branch.joints,
                        measured_gripper,
                        pre_release_delay=release_delay,
                    )
                )
                phases.append(RetreatPhase(k, postdrop_branch.joints, end_joints, end_gripper))
            elif completed_phase_name == "carry":
                # The executor normally runs release directly from the locked
                # carry endpoint, then replans retreat from elevated readback.
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
                drop_orientation=drop_orientation,
                start_joints=start_joints,
                start_gripper=measured_gripper,
                end_joints=end_joints,
                end_gripper=end_gripper,
            )
