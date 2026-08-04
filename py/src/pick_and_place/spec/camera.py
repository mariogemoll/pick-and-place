# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Nominal optics of the UVC camera modules the rig is built from.

What the datasheet says, not what a particular unit measured: the simulator
authors its cameras from these, and a rig that has never been calibrated falls
back to them. ``calibration_required`` marks that as a stopgap — every value
here is approximate until :mod:`pick_and_place.core.camera_calibration` finds a
measured file for that camera.
"""

from __future__ import annotations

from typing import Any

_NOMINAL_CAMERA_INTRINSICS: dict[str, Any] = {
    "model": "standard",
    "width": 1920,
    "height": 1080,
    "camera_matrix": [
        [1240.0, 0.0, 907.0],
        [0.0, 1240.0, 522.0],
        [0.0, 0.0, 1.0],
    ],
    "dist_coeffs": [-0.428, 0.203, 0.0, -0.001, -0.049],
    "fovy_deg": 47.0,
    "fovx_deg": 75.5,
    "approximate": True,
    "calibration_required": True,
}

WRIST_CAMERA_INTRINSICS: dict[str, Any] = dict(_NOMINAL_CAMERA_INTRINSICS)
OVERHEAD_CAMERA_INTRINSICS: dict[str, Any] = dict(_NOMINAL_CAMERA_INTRINSICS)

#: The rig's cameras, by the name the scene gives them.
CAMERA_INTRINSICS_BY_NAME = {
    "wrist_camera": WRIST_CAMERA_INTRINSICS,
    "overhead_camera": OVERHEAD_CAMERA_INTRINSICS,
}
