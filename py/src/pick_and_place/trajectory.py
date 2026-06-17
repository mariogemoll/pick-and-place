# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Phases: (1) neutral -> hover, (2) hover -> pregrasp at cube center, (3) close
gripper to grasp, (4) lift and carry the grasped cube up and over to the hover
above the target, (5) release, lift clear, and flow back to neutral. The release
is left to gravity: the gripper set point opens and the cube falls on its own.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pick_and_place.geometry import (
    CubeFace,
    CubePose,
    SAFETY_MARGIN,
    VERTICAL_FACES,
    WORLD_UP,
    pregrasp_matrix,
    simple_pregrasp_matrix,
    world_from_cube,
    world_from_cube_contact,
)
from pick_and_place.ik import solve_simple_pregrasp_ik
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics
from pick_and_place import transforms as tf
from pick_and_place.transforms import Mat4, Vec3

# Number of intermediate heights checked along the hover→pregrasp descent when
# selecting a grasp. Catching joint-limit violations between endpoints prevents
# the arm from falling back to the joint lerp mid-descent.
_N_DESCENT_CHECKS = 8

# Tip-contact height of the source hover above the floor (clears the 3 cm cube
# top by 1 cm). At the grasp the tip sits at the cube-center height, so the
# world-z offset applied to the pregrasp is ``tip_z - pose.z``.
SOURCE_HOVER_TIP_Z = 0.04
# Tip-contact height of the hover the carry ends at, above the target (lower
# than the source hover, ready for a gentle release).
PREDROP_HOVER_TIP_Z = 0.02
# Tip-contact height of the hover the retreat lifts to after releasing, clearing
# the dropped cube before flowing back to neutral.
POSTDROP_HOVER_TIP_Z = 0.04

# Gripper joint angle at the hover pregrasp: 40 deg open.
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
    "shoulder_pan": math.radians(5.492788249022637),
    "shoulder_lift": math.radians(-95.10455666443607),
    "elbow_flex": math.radians(89.41623426698676),
    "wrist_flex": math.radians(75.45878504436078),
    "wrist_roll": math.radians(-86.5807853208365)
}
REST_GRIPPER = math.radians((10 - 2.3) / 96.2 * 130 - 10)


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
# Phase 3: fixed dwell to close the gripper onto the cube.
GRASP_DURATION = 1.0
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
CARRY_CRUISE_Z = 0.03
# The side-view carry is one C2 spline through four waypoints: leave the pick
# vertically, round into a level cruise, hold the cruise, round down and arrive
# vertically. The waypoint phases below place the cruise in the middle 20 %.
_CARRY_WAYPOINT_PHASES = (0.0, 0.4, 0.6, 1.0)
# Horizontal travel fraction spent rounding into / out of the cruise.
_CARRY_CORNER_TRAVEL = 0.25
# How many points along the carry to check for reachability when planning it.
_CARRY_SAMPLES = 24
# Resolution of the arc-length table used to retime the geometric curve.
_CARRY_ARC_LENGTH_SAMPLES = 2048
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
                start[name], waypoint[name], 0.0, waypoint_velocity, waypoint_phase, p / waypoint_phase
            )
        else:
            out[name] = _quintic_hermite(
                waypoint[name], end[name], waypoint_velocity, 0.0,
                1.0 - waypoint_phase, (p - waypoint_phase) / (1.0 - waypoint_phase),
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
    points solved for the hover and the at-cube pregrasp on that branch."""

    face: CubeFace
    elbow: str
    hover_joints: dict[str, float]
    pregrasp_joints: dict[str, float]


def _face_naturalness(k: So101Kinematics, face: CubeFace, source: CubePose) -> float:
    """Dot product of the face outward-normal with the cube→robot direction.

    Higher means the face is pointing more toward the robot, i.e. is the most
    natural side to approach from. Used to sort candidates before trying IK, so
    the arm never falls through to a far-side face when a near-side one is only
    slightly roll-blocked.
    """
    dx = k.pan_axis[0] - source.x
    dy = k.pan_axis[1] - source.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    c, s = math.cos(source.yaw), math.sin(source.yaw)
    normals: dict[CubeFace, tuple[float, float]] = {
        "+x": (c, s), "-x": (-c, -s), "+y": (-s, c), "-y": (s, -c)
    }
    nx, ny = normals[face]
    return (nx * dx + ny * dy) / dist


def grasp_candidates(k: So101Kinematics, source: CubePose) -> Iterator[GraspChoice]:
    """Yield every IK-feasible grasp in preference order (naturalness, then elbow-up).

    Faces are tried in order of naturalness (outward normal most aligned with
    cube→robot direction first). For each candidate the entire hover→pregrasp
    descent is verified at ``_N_DESCENT_CHECKS`` intermediate heights so the IK
    never needs to fall back to the joint lerp mid-descent.
    """
    hover_offset = SOURCE_HOVER_TIP_Z - source.z
    sorted_faces = sorted(VERTICAL_FACES, key=lambda f: _face_naturalness(k, f, source), reverse=True)
    for face in sorted_faces:
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
            descent_ok = True
            for i in range(1, _N_DESCENT_CHECKS):
                frac = i / _N_DESCENT_CHECKS
                inter = pregrasp_matrix(face, source, hover_offset * (1.0 - frac))
                if inter is None:
                    descent_ok = False
                    break
                if not any(b.elbow == elbow for b in solve_simple_pregrasp_ik(k, inter)):
                    descent_ok = False
                    break
            if not descent_ok:
                continue
            yield GraspChoice(
                face=face,
                elbow=elbow,
                hover_joints=hover_branch.joints,
                pregrasp_joints=pregrasp_branch.joints,
            )


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
    inside the sector by construction). Cube poses on the floor have zero roll
    and pitch, so the only orientation change is in yaw.
    """

    mode: str  # 'straight' or 'polar'
    # Rigid cube→gripper transform captured at the grasp; the gripper follows the
    # cube through the carry so the held cube stays flush and lands on target.
    cube_from_gripper: Mat4
    pan_axis: Vec3
    grasp_position: Vec3
    drop_position: Vec3
    grasp_yaw: float
    drop_yaw: float
    grasp_radius: float
    drop_radius: float
    grasp_azimuth: float
    drop_azimuth: float
    # (parameter, generalized arc length) samples for retiming the curve. The
    # length folds in the cube's yaw sweep (see ``_build_arc_table``) so rotation
    # is paced too, not just translation.
    arc_table: tuple[tuple[float, float], ...] = ()


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


def _carry_geometry_pose(plan: CarryPlan, parameter: float) -> tuple[float, float, float, float]:
    """World (x, y, z, yaw) of the carried cube at a geometry parameter."""
    travel, height = _carry_path(plan.grasp_position[2], plan.drop_position[2], parameter)
    if plan.mode == "straight":
        x = plan.grasp_position[0] + (plan.drop_position[0] - plan.grasp_position[0]) * travel
        y = plan.grasp_position[1] + (plan.drop_position[1] - plan.grasp_position[1]) * travel
    else:
        radius = plan.grasp_radius + (plan.drop_radius - plan.grasp_radius) * travel
        azimuth = plan.grasp_azimuth + (plan.drop_azimuth - plan.grasp_azimuth) * travel
        x = plan.pan_axis[0] + radius * math.cos(azimuth)
        y = plan.pan_axis[1] + radius * math.sin(azimuth)
    yaw = plan.grasp_yaw + _shortest_delta(plan.grasp_yaw, plan.drop_yaw) * travel
    return x, y, height, yaw


def _carry_geometry_matrix(plan: CarryPlan, parameter: float) -> Mat4:
    """World cube pose at a geometry parameter. This defines shape only; playback
    timing is applied separately by ``_carry_cube_matrix``."""
    x, y, height, yaw = _carry_geometry_pose(plan, parameter)
    return tf.translation(x, y, height) @ tf.rot_z(yaw)


def _build_arc_table(plan: CarryPlan) -> tuple[tuple[float, float], ...]:
    """Cumulative (parameter, generalized arc length) along the carry. The length
    adds the cube's translation to its yaw sweep scaled by ``CARRY_ROTATION_WEIGHT``,
    so retiming by this length paces rotation and translation together — a cube
    that mostly spins in place no longer whips the wrist."""
    table: list[tuple[float, float]] = [(0.0, 0.0)]
    px, py, pz, pyaw = _carry_geometry_pose(plan, 0.0)
    length = 0.0
    for i in range(1, _CARRY_ARC_LENGTH_SAMPLES + 1):
        parameter = i / _CARRY_ARC_LENGTH_SAMPLES
        x, y, z, yaw = _carry_geometry_pose(plan, parameter)
        translation = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
        rotation = abs(_shortest_delta(pyaw, yaw)) * CARRY_ROTATION_WEIGHT
        length += translation + rotation
        table.append((parameter, length))
        px, py, pz, pyaw = x, y, z, yaw
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


def plan_carry(
    k: So101Kinematics, grasp: GraspChoice, source: CubePose, target: CubePose
) -> CarryPlan | None:
    """Plan the carry for an already-chosen grasp (face + elbow).

    The cube path is fixed (grasp pose → target hover); we pick the path *mode*:
    prefer the straight chord, fall back to the polar arc. A mode is accepted
    only if the chosen elbow keeps the arm within joint limits across the *whole*
    sweep, not just the endpoints. That whole-path check is what prevents the
    wrist from being driven past its limit mid-carry (which the per-frame IK
    would otherwise resolve by silently falling back to a joint lerp, whipping
    the gripper).
    """
    grasp_gripper = simple_pregrasp_matrix(grasp.face, source)
    if grasp_gripper is None:
        return None
    inward_normal = tf.transform_direction(world_from_cube_contact(grasp.face, source), WORLD_UP)
    grasp_cube = world_from_cube(_pushed_cube(source, inward_normal, SAFETY_MARGIN))
    cube_from_gripper = np.linalg.inv(grasp_cube) @ grasp_gripper

    drop_offset = PREDROP_HOVER_TIP_Z - target.z
    drop_cube = world_from_cube(dataclasses.replace(target, z=target.z + drop_offset))

    grasp_position = tf.get_position(grasp_cube)
    drop_position = tf.get_position(drop_cube)
    base = {
        "cube_from_gripper": cube_from_gripper,
        "pan_axis": np.asarray(k.pan_axis, dtype=np.float64),
        "grasp_position": grasp_position,
        "drop_position": drop_position,
        "grasp_yaw": source.yaw,
        "drop_yaw": target.yaw,
        "grasp_radius": math.hypot(grasp_position[0] - k.pan_axis[0], grasp_position[1] - k.pan_axis[1]),
        "drop_radius": math.hypot(drop_position[0] - k.pan_axis[0], drop_position[1] - k.pan_axis[1]),
        "grasp_azimuth": math.atan2(grasp_position[1] - k.pan_axis[1], grasp_position[0] - k.pan_axis[0]),
        "drop_azimuth": math.atan2(drop_position[1] - k.pan_axis[1], drop_position[0] - k.pan_axis[0]),
    }
    for mode in ("straight", "polar"):
        plan = CarryPlan(mode=mode, **base)
        feasible = True
        for i in range(_CARRY_SAMPLES + 1):
            gripper_matrix = _carry_geometry_matrix(plan, i / _CARRY_SAMPLES) @ cube_from_gripper
            if not any(b.elbow == grasp.elbow for b in solve_simple_pregrasp_ik(k, gripper_matrix)):
                feasible = False
                break
        if feasible:
            return dataclasses.replace(plan, arc_table=_build_arc_table(plan))
    return None



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
    def name(self) -> str: return "approach"

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
    face: CubeFace
    source: CubePose
    elbow: str
    hover_joints: dict[str, float]
    pregrasp_joints: dict[str, float]

    @property
    def name(self) -> str: return "descent"

    @cached_property
    def duration(self) -> float: return DESCENT_DURATION

    def evaluate(self, t: float) -> Frame:
        alpha = _smoothstep(t / self.duration) if self.duration > 0 else 1.0
        hover_offset = SOURCE_HOVER_TIP_Z - self.source.z
        matrix = pregrasp_matrix(self.face, self.source, hover_offset * (1.0 - alpha))
        branch = None
        if matrix is not None:
            branches = solve_simple_pregrasp_ik(self.k, matrix)
            branch = next((b for b in branches if b.elbow == self.elbow), None)
        joints = (
            branch.joints
            if branch is not None
            else _lerp_joints(self.hover_joints, self.pregrasp_joints, alpha)
        )
        return Frame(joints=joints, gripper=GRIPPER_OPEN)

@dataclass(frozen=True)
class GraspPhase:
    pregrasp_joints: dict[str, float]
    start_gripper: float = GRIPPER_OPEN

    @property
    def name(self) -> str: return "grasp"

    @cached_property
    def duration(self) -> float: return GRASP_DURATION

    def evaluate(self, t: float) -> Frame:
        alpha = _smoothstep(t / self.duration) if self.duration > 0 else 1.0
        gripper = self.start_gripper + (GRIPPER_GRASP - self.start_gripper) * alpha
        return Frame(joints=self.pregrasp_joints, gripper=gripper)

@dataclass(frozen=True)
class CarryPhase:
    k: So101Kinematics
    carry: CarryPlan
    elbow: str
    pregrasp_joints: dict[str, float]
    predrop_joints: dict[str, float]

    @property
    def name(self) -> str: return "carry"

    @cached_property
    def duration(self) -> float:
        return _cartesian_move_duration(self.carry.arc_table[-1][1])

    def evaluate(self, t: float) -> Frame:
        phase = min(1.0, t / self.duration) if self.duration > 0 else 1.0
        gripper_matrix = _carry_cube_matrix(self.carry, phase) @ self.carry.cube_from_gripper
        branch = next(
            (b for b in solve_simple_pregrasp_ik(self.k, gripper_matrix) if b.elbow == self.elbow),
            None,
        )
        joints = (
            branch.joints
            if branch is not None
            else _lerp_joints(self.pregrasp_joints, self.predrop_joints, _smoothstep(phase))
        )
        return Frame(joints=joints, gripper=GRIPPER_GRASP)

@dataclass(frozen=True)
class RetreatPhase:
    k: So101Kinematics
    predrop_joints: dict[str, float]
    postdrop_joints: dict[str, float]
    end_joints: dict[str, float]
    end_gripper: float
    start_gripper: float = GRIPPER_GRASP

    @property
    def name(self) -> str: return "retreat"

    @cached_property
    def hover_duration(self) -> float: return RELEASE_DURATION

    @cached_property
    def return_duration(self) -> float:
        return _joint_move_duration(self.k, self.predrop_joints, self.postdrop_joints, self.end_joints)

    @cached_property
    def duration(self) -> float: return self.hover_duration + self.return_duration

    def evaluate(self, t: float) -> Frame:
        elapsed = min(self.duration, t)
        opening_fraction = RETREAT_OPENING_ANGLE / (GRIPPER_OPEN - self.start_gripper)
        movement_start = opening_fraction * self.hover_duration
        retreat_duration = self.duration - movement_start
        if retreat_duration <= 0.0:
            hover_phase = 1.0
            movement_phase = 1.0
        else:
            hover_phase = (self.hover_duration - movement_start) / retreat_duration
            movement_phase = min(1.0, max(0.0, (elapsed - movement_start) / retreat_duration))
        joints = _spline_joints_through_waypoint(
            self.predrop_joints,
            self.postdrop_joints,
            self.end_joints,
            hover_phase,
            movement_phase,
        )
        if elapsed <= self.hover_duration:
            open_alpha = elapsed / self.hover_duration if self.hover_duration > 0 else 1.0
            gripper = self.start_gripper + (GRIPPER_OPEN - self.start_gripper) * open_alpha
        else:
            close_alpha = _smoothstep((elapsed - self.hover_duration) / self.return_duration) if self.return_duration > 0 else 1.0
            gripper = GRIPPER_OPEN + (self.end_gripper - GRIPPER_OPEN) * close_alpha
        return Frame(joints=joints, gripper=gripper)

@dataclass(frozen=True)
class Trajectory:
    phases: tuple[TrajectoryPhase, ...]
    source: CubePose | None = None
    target: CubePose | None = None
    grasp: GraspChoice | None = None
    carry: CarryPlan | None = None
    start_joints: dict[str, float] = dataclasses.field(default_factory=lambda: dict(NEUTRAL_ARM_JOINTS))
    start_gripper: float = NEUTRAL_GRIPPER
    end_joints: dict[str, float] = dataclasses.field(default_factory=lambda: dict(NEUTRAL_ARM_JOINTS))
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
) -> Iterator[Trajectory]:
    """Yield full trajectories from start to end in grasp preference order."""
    for grasp in grasp_candidates(k, source):
        carry = plan_carry(k, grasp, source, target)
        if carry is None:
            continue
        endpoint = _carry_geometry_matrix(carry, 1.0) @ carry.cube_from_gripper
        predrop_branch = next(
            (b for b in solve_simple_pregrasp_ik(k, endpoint) if b.elbow == grasp.elbow), None
        )
        if predrop_branch is None:
            continue
        postdrop_hover = pregrasp_matrix(grasp.face, target, POSTDROP_HOVER_TIP_Z - target.z)
        if postdrop_hover is None:
            continue
        postdrop_branch = next(
            (b for b in solve_simple_pregrasp_ik(k, postdrop_hover) if b.elbow == grasp.elbow), None
        )
        if postdrop_branch is None:
            continue
            
        phases = (
            ApproachPhase(k, start_joints, start_gripper, grasp.hover_joints),
            DescentPhase(k, grasp.face, source, grasp.elbow, grasp.hover_joints, grasp.pregrasp_joints),
            GraspPhase(grasp.pregrasp_joints),
            CarryPhase(k, carry, grasp.elbow, grasp.pregrasp_joints, predrop_branch.joints),
            RetreatPhase(k, predrop_branch.joints, postdrop_branch.joints, end_joints, end_gripper),
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

def replan_remaining_phases(
    k: So101Kinematics,
    measured_joints: dict[str, float],
    measured_gripper: float,
    completed_phase_name: str | None,
    source: CubePose,
    target: CubePose,
    grasp: GraspChoice | None,
    end_joints: dict[str, float],
    end_gripper: float,
) -> Trajectory | None:
    """Plan the remaining trajectory phases starting exactly from the measured state."""
    
    if completed_phase_name == "retreat":
        return Trajectory((), source, target, grasp, None, measured_joints, measured_gripper, end_joints, end_gripper)

    # If we haven't locked in a grasp yet (before Carry phase starts), we need to search or use the provided one
    # Actually, if grasp is None, we should pick the best candidate. If provided, we verify it.
    grasps = [grasp] if grasp is not None else list(grasp_candidates(k, source))
    
    for g in grasps:
        carry = plan_carry(k, g, source, target)
        if carry is None:
            continue
            
        endpoint = _carry_geometry_matrix(carry, 1.0) @ carry.cube_from_gripper
        predrop_branch = next(
            (b for b in solve_simple_pregrasp_ik(k, endpoint) if b.elbow == g.elbow), None
        )
        if predrop_branch is None:
            continue
            
        postdrop_hover = pregrasp_matrix(g.face, target, POSTDROP_HOVER_TIP_Z - target.z)
        if postdrop_hover is None:
            continue
            
        postdrop_branch = next(
            (b for b in solve_simple_pregrasp_ik(k, postdrop_hover) if b.elbow == g.elbow), None
        )
        if postdrop_branch is None:
            continue

        phases = []
        if completed_phase_name is None:
            # We are at the very beginning. Next is Approach.
            phases.append(ApproachPhase(k, measured_joints, measured_gripper, g.hover_joints))
            phases.append(DescentPhase(k, g.face, source, g.elbow, g.hover_joints, g.pregrasp_joints))
            phases.append(GraspPhase(g.pregrasp_joints, start_gripper=GRIPPER_OPEN))
            phases.append(CarryPhase(k, carry, g.elbow, g.pregrasp_joints, predrop_branch.joints))
            phases.append(RetreatPhase(k, predrop_branch.joints, postdrop_branch.joints, end_joints, end_gripper, start_gripper=GRIPPER_GRASP))
        elif completed_phase_name == "approach":
            # Next is Descent. We start from measured hover.
            phases.append(DescentPhase(k, g.face, source, g.elbow, measured_joints, g.pregrasp_joints))
            phases.append(GraspPhase(g.pregrasp_joints, start_gripper=measured_gripper))
            phases.append(CarryPhase(k, carry, g.elbow, g.pregrasp_joints, predrop_branch.joints))
            phases.append(RetreatPhase(k, predrop_branch.joints, postdrop_branch.joints, end_joints, end_gripper, start_gripper=GRIPPER_GRASP))
        elif completed_phase_name == "descent":
            # Next is Grasp. We start from measured pregrasp.
            phases.append(GraspPhase(measured_joints, start_gripper=measured_gripper))
            phases.append(CarryPhase(k, carry, g.elbow, measured_joints, predrop_branch.joints))
            phases.append(RetreatPhase(k, predrop_branch.joints, postdrop_branch.joints, end_joints, end_gripper, start_gripper=GRIPPER_GRASP))
        elif completed_phase_name == "grasp":
            # Next is Carry. We start from measured post-grasp.
            phases.append(CarryPhase(k, carry, g.elbow, measured_joints, predrop_branch.joints))
            phases.append(RetreatPhase(k, predrop_branch.joints, postdrop_branch.joints, end_joints, end_gripper, start_gripper=measured_gripper))
        elif completed_phase_name == "carry":
            # Next is Retreat. We start from measured predrop.
            phases.append(RetreatPhase(k, measured_joints, postdrop_branch.joints, end_joints, end_gripper, start_gripper=measured_gripper))

        return Trajectory(
            phases=tuple(phases),
            source=source,
            target=target,
            grasp=g,
            carry=carry,
            start_joints=measured_joints,
            start_gripper=measured_gripper,
            end_joints=end_joints,
            end_gripper=end_gripper,
        )
        
    return None
