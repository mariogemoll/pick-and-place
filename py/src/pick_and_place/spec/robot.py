# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Identity, ordering and standing postures of the SO-101's joints.

The names are the ones the MJCF gives its joints and the ones lerobot's
``SO101Follower`` reports and accepts, which is why a single tuple can serve the
simulator, the hardware driver, the recorder and the policies. The *order* is
part of the contract: every 6-vector of joint values in this project — a follower
action, a recorded observation, a policy's output — is indexed by it.

The postures are the same kind of shared fact: where the arm parks, where it
starts, and how far the jaws open are properties of this rig, agreed on by the
planner that moves it, the recorder that logs it and the policies trained on
what came out. Angles are in radians, in the simulator's frame.
"""

from __future__ import annotations

import math

# The five arm joints, base to tip.
ARM_JOINT_NAMES: tuple[str, str, str, str, str] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# The arm joints followed by the gripper: the layout of every joint 6-vector.
JOINT_NAMES: tuple[str, ...] = (*ARM_JOINT_NAMES, "gripper")

GRIPPER_INDEX = len(ARM_JOINT_NAMES)

# Gripper joint angle held while approaching and while the jaws hover around the
# cube: 40 deg open, wide enough to clear a 30 mm face on the way down.
GRIPPER_OPEN = math.radians(40.0)

# Gripper joint angle commanded to close on the cube. Past where the jaws meet
# the faces, so the servo keeps loading against them rather than stopping short.
GRIPPER_GRASP = 0.10

# The pose every episode starts from: the arm straight up, wrist rolled square.
NEUTRAL_ARM_JOINTS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": -math.pi / 2,
}
NEUTRAL_GRIPPER = 0.0

# Where the real arm is parked: folded down onto itself, which is the only pose
# it can be powered off in without dropping. Measured off the physical rig,
# which is why the values are not round.
REST_ARM_JOINTS: dict[str, float] = {
    "shoulder_pan": math.radians(4.967032967032967),
    "shoulder_lift": math.radians(-95.16483516483517),
    "elbow_flex": math.radians(96.13186813186813),
    "wrist_flex": math.radians(73.71428571428571),
    "wrist_roll": math.radians(-86.46153846153847),
}
REST_GRIPPER = math.radians((10.5 - 2.3) / 96.2 * 130 - 10)
