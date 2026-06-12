# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Collision boxes for the SO-101 printed wrist-camera mount.

Poses are local to the ``gripper`` body frame of
``SO-ARM100/Simulation/SO101/so101_new_calib.xml``. MuJoCo box sizes are
half-extents.

The camera PCB/module and lens cylinder are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


WRIST_CAMERA_MOUNT_COLLISION_BOXES: tuple[Box, ...] = (
    Box(
        name="camera_mount_col0",
        pos=(0.0022, 0.04737, 0.00338),
        quat=(0.92388, 0.382683, 0.0, 0.0),
        size=(0.018, 0.014, 0.002),
    ),
    Box(
        name="camera_mount_col1",
        pos=(0.002414, 0.07091, 0.00478),
        quat=(0.976296, -0.21644, 0.0, 0.0),
        size=(0.018, 0.017823, 0.002608),
    ),
    Box(
        name="camera_mount_col2",
        pos=(0.001, 0.026, -0.016666),
        quat=(0.707107, 0.707107, 0.0, 0.0),
        size=(0.012, 0.011174, 0.002),
    ),
    Box(
        name="camera_mount_col3",
        pos=(0.002414, 0.032, -0.00715),
        size=(0.018, 0.008, 0.002),
    ),
)
