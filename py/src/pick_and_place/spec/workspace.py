# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Physical measurements of the rig and the task.

Every value here describes something that exists in the world: a printed cube, a
sheet of paper on a table, an aluminium frame surveyed against the robot base.
The simulator reproduces these, the detectors look for them, and the planners
reason about them — so they are declared once, here, and never restated.
"""

from __future__ import annotations

# Half-edge of the pick cube; the 30 mm faces carry 30 mm AprilTag stickers.
CUBE_HALF_SIZE = 0.015

# Tag ids stickered onto the cube's six faces, in MuJoCo cube-texture order
# (right, left, up, down, front, back). With the cube unrotated those map to the
# world directions -X, +X, -Y, +Y, +Z, -Z respectively.
CUBE_APRILTAG_IDS: tuple[int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5)

# Half-edge of the square paper target the cube is dropped onto.
DROP_ZONE_HALF_SIZE = 0.05

# Pose of the workspace frame in world coordinates, surveyed against the robot
# base. The frame carries the corner AprilTag plates, so this pose is what ties
# the detected tags to world XY.
WORKSPACE_FRAME_POS = (0.279579, 0.0000305, 0.0)
WORKSPACE_FRAME_QUAT = (-0.707107, 0.0, 0.0, -0.707107)

# Corner plates bolted to the workspace frame, as (tag id, compass name,
# position in the frame's local coordinates).
WORKSPACE_FRAME_APRILTAG_PLATES: tuple[tuple[int, str, tuple[float, float, float]], ...] = (
    (12, "ne", (0.230, 0.230, 0.0025)),
    (13, "nw", (-0.230, 0.230, 0.0025)),
    (14, "sw", (-0.230, -0.230, 0.0025)),
    (15, "se", (0.230, -0.230, 0.0025)),
)
