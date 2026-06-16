# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Nominal camera intrinsics for the camera modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_CAMERA_INTRINSICS_DIR = REPO_ROOT / "config" / "camera_intrinsics"

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

CAMERA_INTRINSICS_BY_NAME = {
    "wrist_camera": WRIST_CAMERA_INTRINSICS,
    "overhead_camera": OVERHEAD_CAMERA_INTRINSICS,
}


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
