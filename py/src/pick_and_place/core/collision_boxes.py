# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Hand-tuned box collision model for the SO-101 arm.

The values are the asset: they were fitted offline (voxel-overlap box
fitting plus manual mm-level tuning of the gripper jaws) against the stock
SO-101 meshes and validated in pick-and-place simulation.

Poses are local to the stock body frames of
``SO-ARM100/Simulation/SO101/so101_new_calib.xml``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Contact parameters for the gripper jaw geoms. Part of the tuned asset:
#: grasp stability depends on these as much as on the box shapes.
GRIP_FRICTION = (2.0, 0.01, 0.001)
GRIP_CONDIM = 4
GRIP_SOLREF = (0.002, 1.0)
GRIP_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)  # tuned dmin/dmax/width, default midpoint/power


@dataclass(frozen=True)
class Box:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #: Apply the gripper contact parameters above.
    grip: bool = False


#: Collision boxes keyed by stock body name.
COLLISION_BOXES: dict[str, tuple[Box, ...]] = {
    "base": (
        Box(
            name="base_col0",
            pos=(0.0108262962, -8.97657e-09, 0.0455999985),
            quat=(0.5, 0.5, 0.5, -0.5),
            size=(0.024, 0.042, 0.026),
        ),
        Box(
            name="base_col1",
            pos=(0.0208262962, -8.97657e-09, 0.00859999845),
            quat=(0.5, 0.5, 0.5, -0.5),
            size=(0.011, 0.044, 0.056),
        ),
    ),
    "shoulder": (
        Box(
            name="shoulder_col0",
            pos=(-0.0181991995, 0.000162462614, 0.0188999997),
            quat=(0.0, 0.707106781, -0.707106781, 0.0),
            size=(0.025, 0.031, 0.028),
        ),
        Box(
            name="shoulder_col1",
            pos=(-0.0321991995, 0.00116246261, -0.0381000003),
            quat=(0.0, 0.707106781, -0.707106781, 0.0),
            size=(0.028, 0.019, 0.027),
        ),
    ),
    "upper_arm": (
        Box(
            name="upper_arm_col0",
            pos=(-0.039084999, 0.000899999723, 0.0201499995),
            quat=(-0.5, -0.5, 0.5, 0.5),
            size=(0.012, 0.034, 0.052),
        ),
        Box(
            name="upper_arm_col1",
            pos=(-0.112084999, -0.0131000003, 0.0191499995),
            quat=(-0.5, -0.5, 0.5, 0.5),
            size=(0.026, 0.025, 0.019),
        ),
    ),
    "lower_arm": (
        Box(
            name="lower_arm_col0",
            pos=(-0.0385499965, -0.000550141265, 0.0201997877),
            quat=(0.499974717, 0.500025282, -0.499974717, -0.500025282),
            size=(0.012, 0.032, 0.050),
        ),
        Box(
            name="lower_arm_col1",
            pos=(-0.117549996, 0.00344985871, 0.0202001922),
            quat=(0.499974717, 0.500025282, -0.499974717, -0.500025282),
            size=(0.018, 0.028, 0.027),
        ),
    ),
    "wrist": (
        Box(
            name="wrist_col0",
            pos=(-0.000795616758, -0.00584594182, 0.0221499983),
            quat=(-0.00478076322, -0.00478066147, 0.707090645, 0.707090595),
            size=(0.012, 0.032, 0.017),
        ),
        Box(
            name="wrist_col1",
            pos=(-0.0000924933516, -0.0578411875, 0.028150002),
            quat=(-0.00478076322, -0.00478066147, 0.707090645, 0.707090595),
            size=(0.016, 0.026, 0.007),
        ),
        Box(
            name="wrist_col2",
            pos=(-0.00237626471, -0.0368701509, 0.0221500008),
            quat=(-0.00478076322, -0.00478066147, 0.707090645, 0.707090595),
            size=(0.018, 0.032, 0.012),
        ),
    ),
    "gripper": (
        Box(
            name="gripper_servo_col",
            pos=(0.0088, 0.0002, -0.0234),
            quat=(0.707, -0.009, 0.707, 0.009),
            size=(0.012, 0.020, 0.012),
        ),
        # Jaw boxes: col0a/b cover the jaw root; indices 1-5 are the
        # hand-tuned staircase boxes ascending toward the jaw tip.
        Box(
            name="fixed_jaw_col0a",
            pos=(-0.01125, 0.0, -0.0285),
            size=(0.02325, 0.024, 0.012),
            grip=True,
        ),
        Box(
            name="fixed_jaw_col0b",
            pos=(-0.00225, -0.00225, -0.00675),
            size=(0.03225, 0.02625, 0.00825),
            grip=True,
        ),
        Box(
            name="fixed_jaw_col1",
            pos=(-0.0244, 0.0, -0.041025),
            size=(0.00941, 0.016, 0.00412),
            grip=True,
        ),
        Box(
            name="fixed_jaw_col2",
            pos=(-0.02291, 0.0, -0.05485),
            size=(0.00962, 0.01134, 0.00981),
            grip=True,
        ),
        Box(
            name="fixed_jaw_col3",
            pos=(-0.0176, 0.0, -0.0745),
            size=(0.00601, 0.00805, 0.0098),
            grip=True,
        ),
        Box(
            name="fixed_jaw_col4",
            pos=(-0.01492, -0.00018, -0.0893),
            size=(0.00503, 0.00703, 0.005),
            grip=True,
        ),
        Box(
            name="fixed_jaw_col5",
            pos=(-0.01189, -0.00015, -0.099363),
            size=(0.004, 0.00545, 0.005063),
            grip=True,
        ),
    ),
    "moving_jaw_so101_v1": (
        Box(
            name="moving_jaw_col0",
            pos=(-1.01141632e-06, -0.00600266783, 0.0189),
            size=(0.00999898836, 0.0159973321, 0.0240000002),
            grip=True,
        ),
        Box(
            name="moving_jaw_col1",
            pos=(0.0, -0.0398, 0.01835),
            size=(0.00841, 0.02243, 0.01),
            grip=True,
        ),
        Box(
            name="moving_jaw_col2",
            pos=(-0.004, -0.0669, 0.019),
            size=(0.006, 0.005, 0.007),
            grip=True,
        ),
        Box(
            name="moving_jaw_col3",
            pos=(-0.00695, -0.07695, 0.01902),
            size=(0.005, 0.00505, 0.006),
            grip=True,
        ),
    ),
}
