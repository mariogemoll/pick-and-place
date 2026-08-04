# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Read the rig's measured camera calibration off disk.

Intrinsics are solved from a ChArUco board, extrinsics from the four
workspace-frame AprilTags. Both describe one physical rig rather than the
project, so the files are machine-local and gitignored; everything that renders
or detects against them falls back to
:mod:`pick_and_place.spec.camera` when they are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pick_and_place.core.paths import REPO_ROOT
from pick_and_place.spec.camera import CAMERA_INTRINSICS_BY_NAME

LOCAL_CAMERA_INTRINSICS_DIR = REPO_ROOT / "config" / "camera_intrinsics"
LOCAL_CAMERA_EXTRINSICS_DIR = REPO_ROOT / "config" / "camera_extrinsics"


def load_camera_intrinsics(path: Path) -> dict[str, Any]:
    """Load one camera intrinsics JSON file."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"camera intrinsics must be a JSON object: {path}")
    return data


def load_local_camera_intrinsics(
    directory: Path = LOCAL_CAMERA_INTRINSICS_DIR,
) -> dict[str, dict[str, Any]]:
    """Load local camera intrinsics JSON files named after known cameras."""
    overrides: dict[str, dict[str, Any]] = {}
    for camera_name in CAMERA_INTRINSICS_BY_NAME:
        path = directory / f"{camera_name}.json"
        if path.exists():
            overrides[camera_name] = load_camera_intrinsics(path)
    return overrides


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
