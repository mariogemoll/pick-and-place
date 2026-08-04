# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Identity and ordering of the SO-101's joints.

The names are the ones the MJCF gives its joints and the ones lerobot's
``SO101Follower`` reports and accepts, which is why a single tuple can serve the
simulator, the hardware driver, the recorder and the policies. The *order* is
part of the contract: every 6-vector of joint values in this project — a follower
action, a recorded observation, a policy's output — is indexed by it.
"""

from __future__ import annotations

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
