# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Nominal optics of the UVC camera modules the rig is built from.

What the datasheet says, not what a particular unit measured. The simulator
authors its cameras from these and never reads a measured file, so a scored
evaluation and a recorded dataset are both reproducible from a clone; domain
randomization perturbs around them, and its envelope is what covers the
deviation any particular rig has. ``approximate`` and ``calibration_required``
are addressed to the *real* rig, where a datasheet is a poor stand-in for a
measured lens — in simulation the camera simply is these numbers, so there is
nothing they approximate.

**The rectified pinhole is the standard.** The physical modules are wide-angle
and barrel-distorted, so a real frame is undistorted before anything consumes
it during dataset conversion, while a MuJoCo camera is an ideal
pinhole to begin with. The two meet at the undistorted view, and the only
quantity that has to agree there is the vertical field of view — every crop
downstream keeps full height. ``fovy_deg`` is therefore the load-bearing value
here, and a rig with different modules needs to match *it*, not the module.

That angle is a choice, not a property of the lens: undistorting a wide-angle
image lets you keep as much field as you are willing to stretch at the edges.
This project rectifies to ``fy = 1240`` on 1920x1080, i.e. 47 degrees vertical,
which gives 75.4 degrees horizontally at 16:9, 60.2 at the 4:3 recording size,
and 47.0 at the 512x512 square a policy is fed.

Only ``fovy_deg`` reaches the simulator. MuJoCo cameras put the principal point
at the image centre and have no distortion model, so ``camera_matrix``'s offset
centre and ``dist_coeffs`` are used solely by the real-rig undistortion path.
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
