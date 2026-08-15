# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Interpolation, easing and how long a move takes.

The vocabulary every phase of a trajectory is written in: blend between two
joint dicts, ease a fraction so a move starts and ends at rest, and price a move
in seconds from the joint and Cartesian speed limits. Nothing here knows what
the arm is doing — only how it gets from one configuration to another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.spec.robot import ARM_JOINT_NAMES


JOINT_SPEED = 1.5


# Cartesian speed of the gripper/cube tip along the carry, m/s. Governs phase 4.
CARTESIAN_SPEED = 0.45


# Floor on any speed-derived phase so short moves still have room to ease in/out.
MIN_TRAVEL_DURATION = 0.5


# Points sampled along each joint-space move to measure the tip's Cartesian path.
_TIP_PATH_SAMPLES = 24


# Fraction of the carry spent smoothly accelerating in and decelerating out.
_CARRY_EASE_FRACTION = 0.2


@dataclass(frozen=True)
class Frame:
    """One trajectory sample: arm joint set points plus the gripper set point."""

    joints: dict[str, float]
    gripper: float


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


def smoothstep(t: float) -> float:
    """Ease ``t`` in [0, 1] to [0, 1] with zero slope at both ends.

    The project's one easing curve: every smooth move — a trajectory phase, a
    ramp onto a start pose, a teleop hand-off — starts and stops from rest
    through this, so nothing anywhere begins or ends with a velocity step.
    """
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def ramp_setpoints(start: np.ndarray, target: np.ndarray, steps: int) -> list[np.ndarray]:
    """Split one setpoint change into ``steps`` equal sends, arriving on ``target``.

    A policy that emits setpoints more slowly than the arm can be commanded hands
    the servos a single step per period and nothing in between, so they chase a
    staircase. Spreading the same travel over ``steps`` sends within the period
    covers identical ground with a fraction of the per-send jump. The last element
    is exactly ``target``, so no error accumulates across periods.

    ``steps == 1`` returns ``[target]`` — the undivided step, unchanged.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    return [start + (target - start) * ((i + 1) / steps) for i in range(steps)]


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


def _lerp_joints(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    return {name: a[name] + (b[name] - a[name]) * alpha for name in ARM_JOINT_NAMES}


def shortest_delta(a: float, b: float) -> float:
    """Signed angular difference ``b - a`` wrapped to ``[-pi, pi]``."""
    d = (b - a) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


def _joint_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(
        sum(shortest_delta(a[name], b[name]) ** 2 for name in ARM_JOINT_NAMES)
    )
