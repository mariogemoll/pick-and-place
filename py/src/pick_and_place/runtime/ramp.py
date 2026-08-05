# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Move the physical arm smoothly from wherever it is onto a pose.

Every script that drives the rig needs this before it can do anything else: the
arm is parked somewhere unknown, and the first commanded pose is somewhere else.
Sending it straight there is a step input the servos answer with a lunge, so the
move is eased instead — smoothstepped over a couple of seconds at ``CONTROL_HZ``,
with the duration stretched so no arm joint exceeds a velocity cap.

This is not a trajectory. Nothing about the path is planned, checked for
collisions, or recorded; it is the transition *into* a planned motion, or out of
one and back to a parking pose. ``on_tick`` is what lets a caller keep a
simulator and viewer in step with the real arm while it happens.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from pick_and_place.core.joint_frames import action_to_joints, clamp_and_warn, joints_to_action
from pick_and_place.planning.motion import smoothstep
from pick_and_place.spec.robot import CONTROL_HZ, GRIPPER_INDEX

# Seconds spent ramping onto a pose when the velocity cap does not ask for more.
# Long enough that the arm never lunges from a far parking pose, short enough not
# to be a wait between episodes.
RAMP_DURATION = 2.0


def ramp_duration(
    current: np.ndarray, target: np.ndarray, max_joint_speed: float | None = None
) -> float:
    """How long the move from ``current`` to ``target`` should take, in seconds.

    ``RAMP_DURATION`` unless holding every arm joint under ``max_joint_speed``
    (deg/s) needs longer, so a large move from a far parking pose obeys the same
    velocity cap as the closed-loop run rather than snapping over in a fixed
    window. The gripper is excluded: its 0-100 travel is not an angular rate and
    a full open would otherwise stretch every ramp.
    """
    if max_joint_speed is None or max_joint_speed <= 0.0:
        return RAMP_DURATION
    arm_travel = float(np.max(np.abs(target[:GRIPPER_INDEX] - current[:GRIPPER_INDEX])))
    return max(RAMP_DURATION, arm_travel / max_joint_speed)


def ramp_follower(
    follower,
    target_real: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    warned: set[str],
    *,
    max_joint_speed: float | None = None,
    duration: float | None = None,
    viewer=None,
    on_tick: Callable[[float, np.ndarray], None] | None = None,
) -> bool:
    """Smoothstep ``follower`` from its measured pose onto ``target_real``.

    Both poses are in the real frame (arm degrees, gripper 0-100). Every command
    is clamped to ``[low, high]`` through :func:`clamp_and_warn`, so a joint that
    would have been driven past its limit says so once instead of silently.

    ``duration`` overrides the computed one for a caller that has its own pace.
    ``on_tick(alpha, command)`` runs after each command is sent and before the
    tick's remaining time is slept away — enough room to advance a simulator and
    sync a viewer at the same eased fraction. Passing a ``viewer`` stops the ramp
    as soon as the window closes.

    Returns False if it stopped early for that reason, True if it ran to the end.
    """
    current = action_to_joints(follower.get_observation(), target_real)
    delta = target_real - current
    if duration is None:
        duration = ramp_duration(current, target_real, max_joint_speed)
    steps = max(1, round(duration * CONTROL_HZ))
    period = 1.0 / CONTROL_HZ
    for i in range(1, steps + 1):
        if viewer is not None and not viewer.is_running():
            return False
        step_start = time.monotonic()
        alpha = smoothstep(i / steps)
        command = clamp_and_warn(current + alpha * delta, low, high, warned)
        follower.send_action(joints_to_action(command))
        if on_tick is not None:
            on_tick(alpha, command)
        remaining = period - (time.monotonic() - step_start)
        if remaining > 0.0:
            time.sleep(remaining)
    return True
