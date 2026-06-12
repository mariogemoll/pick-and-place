# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Attach the SO-101 hex-nut wrist-camera mount and UVC camera module."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.camera_module import add_camera_module
from pick_and_place.wrist_camera_mount_collision_boxes import (
    WRIST_CAMERA_MOUNT_COLLISION_BOXES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WRIST_CAMERA_MOUNT_STL = (
    REPO_ROOT
    / "SO-ARM100"
    / "Optional"
    / "SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module"
    / "stl"
    / "SO-ARM101_camera_wrist_mount.stl"
)

# Normalize the optional-part STL from millimeters and CAD axes into the
# meter-based gripper frame before adding it to the robot.
_MM_TO_M = 0.001
_MOUNT_SOURCE_ROTATION = np.array(
    (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )
)
# Fitted in pickplace-lab so the locating nib seats flush in the wrist.
_MOUNT_VISUAL_POS = (-0.015086, 0.024001, -0.031666)
_MOUNT_RGBA = (1.0, 0.82, 0.12, 1.0)

# Board-center pose in the stock gripper frame, fitted to the mount's plate.
_CAMERA_POS = (0.0025, 0.073357, 0.007515)
_CAMERA_QUAT = (-0.976296, 0.21644, 0.0, 0.0)
_COLLISION_RGBA = (0.2, 0.8, 0.2, 0.5)


def _canonical_mount_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return the binary STL triangles in canonical meter-based coordinates."""
    data = WRIST_CAMERA_MOUNT_STL.read_bytes()
    triangle_count = int.from_bytes(data[80:84], "little")
    triangles = np.frombuffer(
        data,
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
        count=triangle_count,
        offset=84,
    )
    vertices = triangles["vertices"].reshape(-1, 3).astype(float)
    vertices = vertices @ _MOUNT_SOURCE_ROTATION.T * _MM_TO_M
    faces = np.arange(len(vertices), dtype=int).reshape(-1, 3)
    return vertices, faces


def add_wrist_camera(spec: mujoco.MjSpec) -> None:
    """Attach the printed mount, camera module, and render camera to ``spec``."""
    vertices, faces = _canonical_mount_mesh()
    mount_mesh = spec.add_mesh(name=WRIST_CAMERA_MOUNT_STL.stem)
    mount_mesh.uservert = vertices.flatten()
    mount_mesh.userface = faces.flatten()
    collision_default = spec.find_default("collision")
    mount = spec.body("gripper").add_body(name="wrist_camera_mount")
    mount.add_geom(
        name="wrist_camera_mount_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        pos=_MOUNT_VISUAL_POS,
        meshname=mount_mesh.name,
        rgba=_MOUNT_RGBA,
        contype=0,
        conaffinity=0,
        group=2,
    )
    for box in WRIST_CAMERA_MOUNT_COLLISION_BOXES:
        mount.add_geom(
            default=collision_default,
            name=box.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box.pos,
            quat=box.quat,
            size=box.size,
            rgba=_COLLISION_RGBA,
        )

    mount.add_site(
        name="wrist_camera_attach",
        pos=_CAMERA_POS,
        quat=_CAMERA_QUAT,
        size=(0.0015, 0.0015, 0.0015),
        group=3,
    )
    camera_module = add_camera_module(
        mount,
        prefix="wrist_",
        pos=_CAMERA_POS,
        quat=_CAMERA_QUAT,
        collision_default=collision_default,
    )
    camera_module.add_camera(name="wrist_camera")
