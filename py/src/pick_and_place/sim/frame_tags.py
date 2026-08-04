# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Where the workspace-frame AprilTags sit, read off the compiled model.

The four corner plates are bolted to the frame and never move, so their world
poses follow from the model alone. That makes them the fixed points everything
that solves an extrinsic pose measures against: the detector finds the same
tags in a real image, and the difference between the two is the camera pose.

Both functions return the *top face* of each plate — the surface the sticker is
printed on — rather than the geom center, since that is what a camera sees.
"""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from pick_and_place.spec.workspace import (
    APRILTAG_BORDER_FRACTION,
    WORKSPACE_FRAME_APRILTAG_PLATES,
    WORKSPACE_FRAME_APRILTAG_SIZE,
)

#: Tag id -> the geom ``environment`` names for that corner plate.
TAG_GEOMS: dict[int, str] = {
    tag_id: f"workspace_frame_tag_{corner_name}"
    for tag_id, corner_name, _ in WORKSPACE_FRAME_APRILTAG_PLATES
}

#: Edge of the black border the detector actually returns a quad for.
DETECTED_EDGE_M = WORKSPACE_FRAME_APRILTAG_SIZE * APRILTAG_BORDER_FRACTION

#: Tag-local corner offsets in the order pupil-apriltags returns ``det.corners``.
_CORNERS_LOCAL = (DETECTED_EDGE_M / 2.0) * np.array(
    ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
)


def _tag_faces(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[int, tuple[NDArray, NDArray]]:
    """Center and orientation of each visible tag face, in world coordinates."""
    faces: dict[int, tuple[NDArray, NDArray]] = {}
    mujoco.mj_forward(model, data)
    for tag_id, geom_name in TAG_GEOMS.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        center = data.geom_xpos[geom_id] + rotation[:, 2] * model.geom_size[geom_id][2]
        faces[tag_id] = (center, rotation)
    return faces


def tag_world_points(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, NDArray]:
    """Return visible-face centers for the fixed workspace-frame tag geoms."""
    points = {tag_id: center for tag_id, (center, _) in _tag_faces(model, data).items()}
    if len(points) < len(TAG_GEOMS):
        raise ValueError(f"need all {len(TAG_GEOMS)} workspace-frame tags, found {sorted(points)}")
    return points


def tag_world_corners(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, NDArray]:
    """Return AprilTag-detected corners in pupil-apriltags corner order.

    Each tag supplies four known PnP points, allowing a usable pose estimate
    when just one workspace tag is visible.
    """
    return {
        tag_id: center + _CORNERS_LOCAL @ rotation.T
        for tag_id, (center, rotation) in _tag_faces(model, data).items()
    }
