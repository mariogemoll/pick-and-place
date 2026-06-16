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
