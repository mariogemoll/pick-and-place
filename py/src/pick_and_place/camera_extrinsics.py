# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measured camera extrinsics for sim2real calibration."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.camera_intrinsics import REPO_ROOT

LOCAL_CAMERA_EXTRINSICS_DIR = REPO_ROOT / "config" / "camera_extrinsics"


def load_camera_extrinsics(path: Path) -> dict[str, Any]:
    """Load one camera extrinsics JSON file."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"camera extrinsics must be a JSON object: {path}")
    return data


def load_local_camera_extrinsics(
    directory: Path = LOCAL_CAMERA_EXTRINSICS_DIR,
) -> dict[str, dict[str, Any]]:
    """Load local camera extrinsics JSON files from ``directory``."""
    if not directory.exists():
        return {}

    extrinsics: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        data = load_camera_extrinsics(path)
        cameras = data.get("cameras")
        if isinstance(cameras, dict):
            for name, camera_extrinsics in cameras.items():
                if isinstance(camera_extrinsics, dict):
                    extrinsics[name] = camera_extrinsics
            continue

        camera_name = data.get("camera") or path.stem
        extrinsics[str(camera_name)] = data
    return extrinsics


def apply_camera_extrinsics_to_spec(
    spec: mujoco.MjSpec,
    camera_extrinsics_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    """Override named ``MjSpec`` camera poses from measured extrinsics."""
    applied: list[str] = []
    for camera in spec.cameras:
        extrinsics = camera_extrinsics_by_name.get(camera.name)
        if extrinsics is None:
            continue
        camera.pos = tuple(float(v) for v in extrinsics["pos"])
        camera.quat = tuple(float(v) for v in extrinsics["quat"])
        applied.append(camera.name)
    return applied


#: Marker colours: where the scene *authors* the camera, and where it was
#: *measured* to be. Red is what a checkout without the calibration renders from.
AUTHORED_MARKER_RGBA = (0.9, 0.1, 0.1, 1.0)
MEASURED_MARKER_RGBA = (0.1, 0.85, 0.2, 1.0)
#: Markers sit in their own geom group so the viewer can toggle them with '5'.
MARKER_GEOM_GROUP = 5
MARKER_SPHERE_RADIUS = 0.003
MARKER_ARROW_LENGTH = 0.06
MARKER_ARROW_RADIUS = 0.0008


def add_camera_extrinsics_markers(
    spec: mujoco.MjSpec,
    camera_extrinsics_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    """Mark each calibrated camera's authored and measured pose in the scene.

    Adds, per camera with measured extrinsics, a red ball and view-direction
    arrow at the pose the scene authors and a green pair at the pose that was
    solved from the workspace AprilTags. Both are parented to the camera's own
    body, so they follow it and the offset between them is the calibration
    residual the renderer would otherwise apply invisibly.

    Call *before* :func:`apply_camera_extrinsics_to_spec` — it reads the
    authored pose off the spec.
    """
    marked: list[str] = []
    for camera in spec.cameras:
        extrinsics = camera_extrinsics_by_name.get(camera.name)
        if extrinsics is None:
            continue
        poses = (
            ("authored", np.asarray(camera.pos, float), np.asarray(camera.quat, float),
             AUTHORED_MARKER_RGBA),
            ("measured", np.asarray(extrinsics["pos"], float),
             np.asarray(extrinsics["quat"], float), MEASURED_MARKER_RGBA),
        )
        for label, position, quaternion, rgba in poses:
            body = camera.parent
            body.add_geom(
                name=f"{camera.name}_{label}_pose",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=tuple(position),
                size=(MARKER_SPHERE_RADIUS, 0.0, 0.0),
                rgba=rgba,
                group=MARKER_GEOM_GROUP,
                contype=0,
                conaffinity=0,
            )
            # A MuJoCo camera looks down its own -Z, so this is its sight line.
            direction = np.zeros(3)
            mujoco.mju_rotVecQuat(direction, np.array([0.0, 0.0, -1.0]), quaternion)
            tip = position + direction * MARKER_ARROW_LENGTH
            body.add_geom(
                name=f"{camera.name}_{label}_axis",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=(*position, *tip),
                size=(MARKER_ARROW_RADIUS, 0.0, 0.0),
                rgba=rgba,
                group=MARKER_GEOM_GROUP,
                contype=0,
                conaffinity=0,
            )
        marked.append(camera.name)
    return marked


def camera_pose_delta_mm_deg(
    authored: dict[str, Any] | tuple[np.ndarray, np.ndarray],
    measured: dict[str, Any],
) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between an authored and measured pose."""
    if isinstance(authored, dict):
        authored = (np.asarray(authored["pos"], float), np.asarray(authored["quat"], float))
    position, quaternion = authored
    other_position = np.asarray(measured["pos"], float)
    other_quaternion = np.asarray(measured["quat"], float)
    millimetres = float(np.linalg.norm(other_position - position) * 1000.0)
    relative = np.zeros(4)
    conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    mujoco.mju_mulQuat(relative, other_quaternion, conjugate)
    degrees = float(np.degrees(2.0 * np.arccos(np.clip(abs(relative[0]), -1.0, 1.0))))
    return millimetres, degrees


def apply_camera_extrinsics_to_model(
    model: mujoco.MjModel,
    camera_extrinsics_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    """Override named compiled-model camera poses from measured extrinsics."""
    applied: list[str] = []
    for camera_name, extrinsics in camera_extrinsics_by_name.items():
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            continue
        model.cam_pos[camera_id] = np.array(extrinsics["pos"], dtype=float)
        model.cam_quat[camera_id] = np.array(extrinsics["quat"], dtype=float)
        applied.append(camera_name)
    return applied


def save_camera_extrinsics(
    model: mujoco.MjModel,
    camera_name: str,
    *,
    path: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write one camera's current parent-relative MuJoCo pose to JSON."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise KeyError(f"unknown camera {camera_name!r}")

    if path is None:
        path = LOCAL_CAMERA_EXTRINSICS_DIR / f"{camera_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "solved": datetime.date.today().isoformat(),
        "frame": "parent-body-relative (MuJoCo cam_pos/cam_quat, quat wxyz)",
        "cameras": {
            camera_name: {
                "pos": model.cam_pos[camera_id].tolist(),
                "quat": model.cam_quat[camera_id].tolist(),
            }
        },
    }
    if meta:
        payload.update(meta)

    if path.is_file():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text())
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
