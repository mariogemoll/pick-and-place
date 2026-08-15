# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fold a wrist-camera cube estimate back into the descent that is running.

The descent is planned from a cube pose measured before the episode and corrected
during it by :mod:`pick_and_place.runtime.wrist_servo`. This is the other half of
that loop: what the control loop does with each new estimate.

Two things make it more than an assignment.

**Yaw is ambiguous.** A cube looks the same every 90 degrees and a single visible
tag cannot tell which quarter turn it is, so a raw reading flips between them.
Taken at face value the re-solved grasp would spin the wrist right around
mid-descent. Each reading is folded onto the quarter turn nearest the pose
already being tracked, which is the one the arm is lined up with.

**Readings are noisy and the arm is heavy.** Steering straight at each estimate
would jerk the arm at the detector's rate, so the tracked pose moves a fraction
of the way toward each new reading instead. The grasp is re-derived from that
smoothed pose, keeping the locked face and elbow — the descent corrects *where*
the cube is, never *how* it is being picked up, which was decided before the arm
committed to an approach.
"""

from __future__ import annotations

import dataclasses

from pick_and_place.core.geometry import CubePose
from pick_and_place.scripted.grasp import fold_cube_yaw, grasp_candidates
from pick_and_place.scripted.motion import shortest_delta

#: How far the tracked pose moves toward each new reading. Low enough that a
#: single bad solve cannot throw the descent, high enough to converge within it.
SERVO_SMOOTHING = 0.1


@dataclasses.dataclass(frozen=True)
class ServoCorrection:
    """What one accepted estimate changes.

    ``tracked`` is the smoothed pose the grasp is re-solved against and the one
    the next correction smooths from. ``measured`` is the reading itself, yaw
    folded but otherwise unsmoothed — the simulated cube is moved there so the
    viewer and the recorded ``sim_qpos`` show what the camera actually saw rather
    than a filtered version of it.
    """

    phase: object
    tracked: CubePose
    measured: CubePose


def follow_estimate(phase, tracked: CubePose, estimate: CubePose, kinematics) -> ServoCorrection:
    """Steer ``phase`` toward ``estimate``, starting from the currently ``tracked`` pose."""
    measured = dataclasses.replace(estimate, yaw=fold_cube_yaw(tracked.yaw, estimate.yaw))
    smoothed = dataclasses.replace(
        measured,
        x=tracked.x * (1.0 - SERVO_SMOOTHING) + measured.x * SERVO_SMOOTHING,
        y=tracked.y * (1.0 - SERVO_SMOOTHING) + measured.y * SERVO_SMOOTHING,
        yaw=tracked.yaw + shortest_delta(tracked.yaw, measured.yaw) * SERVO_SMOOTHING,
    )

    if phase.grasp.face != "free":
        updated = next(
            (
                g
                for g in grasp_candidates(kinematics, smoothed)
                if g.face == phase.grasp.face and g.elbow == phase.grasp.elbow
            ),
            None,
        )
        if updated is not None:
            phase = dataclasses.replace(phase, grasp=updated)

    return ServoCorrection(phase=phase, tracked=smoothed, measured=measured)


def regrasp_after_descent(phase, tracked: CubePose, kinematics, *, free_grasp: bool, current):
    """The grasp the descent actually ended on, for rebuilding grasp and lift from it.

    Under ``free_grasp`` the descent's own re-solved grasp *is* the answer. Otherwise
    the locked face and elbow are re-derived one last time from the final tracked
    pose, so the grasp phase closes where the servo left the jaws rather than where
    the pre-episode plan expected them. ``current`` stands if that pose admits no
    candidate on the locked face — the arm is already there, and the plan it flew
    down on is a better guess than nothing.
    """
    if free_grasp:
        return phase.grasp
    for candidate in grasp_candidates(kinematics, tracked):
        if candidate.face == phase.face and candidate.elbow == phase.elbow:
            return candidate
    return current
