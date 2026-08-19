# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Draw the random poses an episode starts from.

Where the cube starts, where it must end up, and what the arm is doing when the
episode begins. Sampling only — :mod:`pick_and_place.runtime.episodes` takes a draw from
here and vets it against live physics, and
:mod:`pick_and_place.scripted.scenario_sampling` composes the cube and target draws into
the declared reset distribution.
"""

from __future__ import annotations

import math

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.spec.robot import GRIPPER_GRASP, GRIPPER_OPEN, NEUTRAL_ARM_JOINTS
from pick_and_place.core.workspace_bounds import (
    AZIMUTH_MAX,
    AZIMUTH_MIN,
    CANONICAL_PICKUP_SECTOR,
    CUBE_PLACEMENT_SECTOR,
    PAN_AXIS,
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
    is_cube_recovery_target_allowed,
    is_target_plate_position_allowed,
)

# ±radians of random joint perturbation applied to the neutral start/end pose.
_NEAR_NEUTRAL_JOINT_SCALE = 0.4
# Per-joint overrides of the perturbation scale. ``shoulder_lift``, ``elbow_flex``
# and ``wrist_flex`` are held tighter than the rest because they are the levers
# that tilt the gripper down toward the floor: a full ±0.4 swing on them is what
# drives the near-neutral start/end pose down toward the ground. Tightening them to
# ±0.2 keeps almost every sampled pose above the clearance gate (≈99% pass), so the
# gate rarely has to resample while still guaranteeing the floor margin.
_JOINT_SCALE_OVERRIDES: dict[str, float] = {
    "shoulder_lift": 0.2,
    "elbow_flex": 0.2,
    "wrist_flex": 0.2,
}
# Shoulder-pan half-range for empty-gripper search poses. The end effector stays
# inside the workspace at neutral out to about ±0.83 rad; 0.75 leaves margin for
# the simultaneous near-neutral perturbations of the other joints.
HUNT_PAN_SCALE = 0.75

# A held cube gets a much gentler search. It only needs to clear the overhead
# line of sight to the plate, not sweep the whole workspace.
CARRY_HUNT_PAN_SCALE = 0.35

PICKUP_YAW_DEVIATION = math.pi / 4.0


def pickup_yaw_from_azimuth(azimuth: float, deviation: float = 0.0) -> float:
    """Return cube yaw relative to the local pickup azimuth frame."""
    return azimuth + deviation


def sample_cube(rng: np.random.Generator) -> CubePose:
    """Sample a cube pose in the canonical pick-lift sector."""
    r_inner = CANONICAL_PICKUP_SECTOR.inner_radius
    r_outer = CANONICAL_PICKUP_SECTOR.outer_radius
    while True:
        # Uniform radial sampling to prevent points bunching up at the outer edge.
        r = rng.uniform(r_inner, r_outer)
        theta = rng.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
        x = PAN_AXIS[0] + r * math.cos(theta)
        y = PAN_AXIS[1] + r * math.sin(theta)
        if is_cube_pickup_allowed(x, y):
            break
    yaw = pickup_yaw_from_azimuth(
        theta,
        rng.uniform(-PICKUP_YAW_DEVIATION, PICKUP_YAW_DEVIATION),
    )
    return CubePose(
        x=x,
        y=y,
        z=CUBE_HALF_SIZE,
        yaw=yaw,
    )


def sample_recovery_cube(rng: np.random.Generator) -> CubePose:
    """Sample a conservative pickup-zone target for unrecorded cube recovery."""
    while True:
        pose = sample_cube(rng)
        if is_cube_recovery_target_allowed(pose.x, pose.y):
            return pose


def sample_target(rng: np.random.Generator) -> CubePose:
    """Sample a target in the broader drop sector with room for the plate."""
    r_inner = CUBE_PLACEMENT_SECTOR.inner_radius
    r_outer = CUBE_PLACEMENT_SECTOR.outer_radius
    while True:
        r = rng.uniform(r_inner, r_outer)
        theta = rng.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
        x = PAN_AXIS[0] + r * math.cos(theta)
        y = PAN_AXIS[1] + r * math.sin(theta)
        if is_cube_drop_allowed(x, y) and is_target_plate_position_allowed(x, y):
            return CubePose(x=x, y=y, z=CUBE_HALF_SIZE)


def sample_near_neutral(rng: np.random.Generator) -> tuple[dict[str, float], float]:
    """Return arm joints and gripper perturbed slightly from the neutral pose.

    Each joint is perturbed by ±its scale (``_JOINT_SCALE_OVERRIDES`` for the
    tightened joints, else ``_NEAR_NEUTRAL_JOINT_SCALE``).
    """
    joints = {}
    for name, value in NEUTRAL_ARM_JOINTS.items():
        scale = _JOINT_SCALE_OVERRIDES.get(name, _NEAR_NEUTRAL_JOINT_SCALE)
        joints[name] = value + rng.uniform(-scale, scale)
    gripper = float(rng.uniform(0.0, GRIPPER_OPEN))
    return joints, gripper


def sample_hunt_pose(
    rng: np.random.Generator, *, carrying: bool = False
) -> tuple[dict[str, float], float]:
    """Return a search pose: a wide shoulder-pan swing, the rest near neutral.

    The arm itself can sit between the fixed overhead camera and the cube or
    drop-zone square, so the look-around search swings the pan far wider than the
    near-neutral start jitter to clear the view from a range of angles. The tilt
    joints stay near neutral, keeping the gripper well above the floor."""
    joints, gripper = sample_near_neutral(rng)
    scale = CARRY_HUNT_PAN_SCALE if carrying else HUNT_PAN_SCALE
    joints["shoulder_pan"] = NEUTRAL_ARM_JOINTS["shoulder_pan"] + rng.uniform(
        -scale, scale
    )
    if carrying:
        gripper = GRIPPER_GRASP
    return joints, gripper
