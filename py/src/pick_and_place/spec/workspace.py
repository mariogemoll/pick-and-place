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

# tagStandard41h12 geometry: the marker is nine cells wide and its black border
# spans five of them, so the quad a detector returns has 5/9 of the edge of the
# printed graphic.
APRILTAG_BORDER_FRACTION = 5.0 / 9.0

# Edge of the printed AprilTag graphic on a cube face: a 20 mm tag on a 30 mm
# sticker.
CUBE_APRILTAG_SIZE = 0.020

# Half-edge of the square paper target the cube is dropped onto.
DROP_ZONE_HALF_SIZE = 0.05

# Pose of the workspace frame in world coordinates, surveyed against the robot
# base. The frame carries the corner AprilTag plates, so this pose is what ties
# the detected tags to world XY.
WORKSPACE_FRAME_POS = (0.279579, 0.0000305, 0.0)
WORKSPACE_FRAME_QUAT = (-0.707107, 0.0, 0.0, -0.707107)

# Half-edge of a square corner plate; the 60 mm plate carries a 40 mm tag.
WORKSPACE_FRAME_APRILTAG_PLATE_HALF_SIZE = 0.03

# Edge of the printed AprilTag graphic on a corner plate.
WORKSPACE_FRAME_APRILTAG_SIZE = 0.040

# Corner plates bolted to the workspace frame, as (tag id, compass name,
# position in the frame's local coordinates).
WORKSPACE_FRAME_APRILTAG_PLATES: tuple[tuple[int, str, tuple[float, float, float]], ...] = (
    (12, "ne", (0.230, 0.230, 0.0025)),
    (13, "nw", (-0.230, 0.230, 0.0025)),
    (14, "sw", (-0.230, -0.230, 0.0025)),
    (15, "se", (0.230, -0.230, 0.0025)),
)

# A sheet of 3 mm grey EVA foam covers the table inside the square the frame
# rails enclose. It is laid on the table, so its top face is 3 mm above world
# Z=0.
FOAM_FLOOR_THICKNESS = 0.003

# The sheet is cut around everything else that stands on the table inside the
# square. At each corner that is the AprilTag plate: the cut runs from the
# plate's inner edge (230 mm - 30 mm) straight out to the rail, taking the
# 2.6 mm of foam that would otherwise be left between plate and rail with it.
FOAM_FLOOR_CORNER_CUTOUT_INNER = 0.200

# At the north edge it is the robot: the printed camera-arm base plate the arm
# is bolted to, which is what stands on the table there and reaches 52.2 mm into
# the square. It is a wedge, not a box — its edges run out at 30 degrees from
# north, from 68.8 mm across at its south tip to 129.0 mm where it crosses the
# rail — and the sheet is cut to that outline. The plate's two feet have a
# V-shaped gap between them; the cut spans it rather than following it in.
FOAM_FLOOR_BASE_CUTOUT_TIP_Y = 0.2104
FOAM_FLOOR_BASE_CUTOUT_TIP_HALF_WIDTH = 0.034379
FOAM_FLOOR_BASE_CUTOUT_FLARE = 0.57735

# Height of the surface anything on the table inside the square rests on. The
# foam covers all of it the cube may be placed on — the cut-outs are excluded
# from the placement bounds — so this is one number and not a function of xy.
WORKSPACE_FLOOR_Z = FOAM_FLOOR_THICKNESS

# Centre height of a cube resting on that surface. Everything that reasons about
# a cube on the table — the sampler, the planner, both localizers, the evaluation
# oracle — takes its z from here rather than assuming the table.
CUBE_REST_Z = WORKSPACE_FLOOR_Z + CUBE_HALF_SIZE
