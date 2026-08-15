# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Choose where to take hold of the cube.

A grasp is a reachable arm configuration whose gripper is closed around a cube
face, plus the hover pose it descends from. The search enumerates faces,
approach pitches and roll offsets in a fixed, deliberate order — closest to a
top-down approach first — so the same cube pose always yields the same
preference ordering, and the first candidate that survives IK, the joint limits
and the workspace sectors is the one taken.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pick_and_place.core import transforms as tf
from pick_and_place.core.geometry import (
    CANONICAL_PREGRASP_DISTANCE,
    CubeFace,
    CubePose,
    WORLD_UP,
    canonical_grasp_matrix,
    canonical_pregrasp_matrix,
)
from pick_and_place.core.ik import solve_simple_grasp_ik
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.core.transforms import Mat4, Vec3
from pick_and_place.core.workspace_bounds import (
    CANONICAL_PICKUP_SECTOR,
    CUBE_PLACEMENT_SECTOR,
)


# Number of intermediate heights checked along the hover→grasp descent when
# selecting a grasp. Catching joint-limit violations between endpoints prevents
# the arm from falling back to the joint lerp mid-descent.
_N_DESCENT_CHECKS = 8


# Recovery grasps may approach along an arbitrary tool axis. After closing, lift
# the held cube vertically to this world height before folding into the carry.
RECOVERY_LIFT_CUBE_Z = 0.08


# Full-range canonical grasp limits and search order.
MIN_CANONICAL_GRASP_RADIUS = CANONICAL_PICKUP_SECTOR.inner_radius


MAX_CANONICAL_GRASP_RADIUS = CANONICAL_PICKUP_SECTOR.outer_radius


MAX_RECOVERY_GRASP_RADIUS = CUBE_PLACEMENT_SECTOR.outer_radius


MIN_CANONICAL_AZIMUTH = CANONICAL_PICKUP_SECTOR.azimuth_min


MAX_CANONICAL_AZIMUTH = CANONICAL_PICKUP_SECTOR.azimuth_max


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


def fold_cube_yaw(reference: float, cube_yaw: float) -> float:
    """Snap a detected cube yaw to the quarter-turn-equivalent nearest ``reference``.

    A cube grasp repeats every 90 deg (:func:`_square_to_cube_face`), so a
    detection that lands a quarter or half turn off the current target is the
    same physical grasp. Folding the yaw into ``[reference - 45deg,
    reference + 45deg)`` keeps the grasp roll continuous, so the single-tag
    planar-pose ambiguity can no longer flip the wrist 90/180 deg mid-descent.
    """
    quarter = math.pi / 2.0
    return reference + (cube_yaw - reference + quarter / 2.0) % quarter - quarter / 2.0


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
