# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place trajectory, built phase by phase.

This drives the MuJoCo model through position actuators under real physics, so
each phase emits joint *set points* that a servo tracks rather than poses written
straight to ``qpos``.

Phases: (1) neutral -> hover, (2) hover -> pregrasp at cube center, (3) close
gripper to grasp, (4) lift and carry the grasped cube up and over to the hover
above the target.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass

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

# Gripper joint angle at the hover pregrasp: 40 deg open.
GRIPPER_OPEN = math.radians(40.0)
# Gripper joint angle commanded during the grasp: geometry-derived contact position.
GRIPPER_GRASP = 0.190589954318

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
# Phase 3: close the gripper onto the cube.
STAGE3_DURATION = 1.0
# Phase 4: lift and carry the grasped cube over to the hover above the target.
STAGE4_DURATION = 2.0

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
    # (parameter, arc length) samples for retiming the curve by distance.
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


def _carry_geometry_matrix(plan: CarryPlan, parameter: float) -> Mat4:
    """World cube pose at a geometry parameter. This defines shape only; playback
    timing is applied separately by ``_carry_cube_matrix``."""
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
    return tf.translation(x, y, height) @ tf.rot_z(yaw)


def _build_arc_table(plan: CarryPlan) -> tuple[tuple[float, float], ...]:
    table: list[tuple[float, float]] = [(0.0, 0.0)]
    previous = tf.get_position(_carry_geometry_matrix(plan, 0.0))
    length = 0.0
    for i in range(1, _CARRY_ARC_LENGTH_SAMPLES + 1):
        parameter = i / _CARRY_ARC_LENGTH_SAMPLES
        position = tf.get_position(_carry_geometry_matrix(plan, parameter))
        length += float(np.linalg.norm(position - previous))
        table.append((parameter, length))
        previous = position
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


@dataclass(frozen=True)
class PickAndCarry:
    """Phases 1-4: neutral -> hover -> pregrasp -> grasp -> carry to target hover."""

    k: So101Kinematics
    source: CubePose
    target: CubePose
    grasp: GraspChoice
    carry: CarryPlan
    predrop_joints: dict[str, float]
    stage1_duration: float = STAGE1_DURATION
    stage2_duration: float = STAGE2_DURATION
    stage3_duration: float = STAGE3_DURATION
    stage4_duration: float = STAGE4_DURATION

    @property
    def duration(self) -> float:
        return (
            self.stage1_duration
            + self.stage2_duration
            + self.stage3_duration
            + self.stage4_duration
        )

    def evaluate(self, t: float) -> Frame:
        stage1_end = self.stage1_duration
        stage2_end = stage1_end + self.stage2_duration
        stage3_end = stage2_end + self.stage3_duration

        if t < stage1_end:
            # Phase 1: swing from neutral to the hover, opening the gripper.
            alpha = _smoothstep(t / self.stage1_duration) if self.stage1_duration > 0 else 1.0
            joints = _lerp_joints(NEUTRAL_ARM_JOINTS, self.grasp.hover_joints, alpha)
            gripper = NEUTRAL_GRIPPER + (GRIPPER_OPEN - NEUTRAL_GRIPPER) * alpha
            return Frame(joints=joints, gripper=gripper)

        if t < stage2_end:
            # Phase 2: descend straight down from the hover to the pregrasp at the
            # cube center, re-solving IK each frame so the tip tracks a vertical
            # line. The joint lerp is a defensive fallback for the rare frame whose
            # interpolated pose has no in-limit branch on the chosen elbow.
            alpha = (
                _smoothstep((t - stage1_end) / self.stage2_duration)
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

        if t < stage3_end:
            # Phase 3: arm holds at the pregrasp; gripper closes onto the cube.
            alpha = _smoothstep((t - stage2_end) / self.stage3_duration) if self.stage3_duration > 0 else 1.0
            gripper = GRIPPER_OPEN + (GRIPPER_GRASP - GRIPPER_OPEN) * alpha
            return Frame(joints=self.grasp.pregrasp_joints, gripper=gripper)

        # Phase 4: lift and carry the grasped cube up and over to the hover above
        # the target, the gripper following the cube along the planned curve. The
        # plan was validated across the whole sweep, so the per-frame IK resolves
        # cleanly; the joint lerp is only a defensive fallback for edge cases.
        phase = (
            min(1.0, (t - stage3_end) / self.stage4_duration) if self.stage4_duration > 0 else 1.0
        )
        gripper_matrix = _carry_cube_matrix(self.carry, phase) @ self.carry.cube_from_gripper
        branch = next(
            (b for b in solve_simple_pregrasp_ik(self.k, gripper_matrix) if b.elbow == self.grasp.elbow),
            None,
        )
        joints = (
            branch.joints
            if branch is not None
            else _lerp_joints(self.grasp.pregrasp_joints, self.predrop_joints, _smoothstep(phase))
        )
        return Frame(joints=joints, gripper=GRIPPER_GRASP)


def pick_and_carry_candidates(
    k: So101Kinematics, source: CubePose, target: CubePose
) -> Iterator[PickAndCarry]:
    """Yield full pick-and-carry trajectories in grasp preference order.

    A grasp is usable only if the same face and elbow that grasp the source can
    also follow the carry to the target hover; candidates whose carry cannot be
    planned on that branch are skipped.
    """
    for grasp in grasp_candidates(k, source):
        carry = plan_carry(k, grasp, source, target)
        if carry is None:
            continue
        endpoint = _carry_geometry_matrix(carry, 1.0) @ carry.cube_from_gripper
        branch = next(
            (b for b in solve_simple_pregrasp_ik(k, endpoint) if b.elbow == grasp.elbow), None
        )
        if branch is None:
            continue
        yield PickAndCarry(
            k=k,
            source=source,
            target=target,
            grasp=grasp,
            carry=carry,
            predrop_joints=branch.joints,
        )
