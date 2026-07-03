# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export camera calibration for dataset visualizers and other consumers.

The output is intentionally repo-agnostic JSON: camera names map to intrinsics,
world-to-camera extrinsics, and image size. It contains no pick-and-place Python
imports or MuJoCo-specific objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco

from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import (
    CAMERA_INTRINSICS_BY_NAME,
    load_local_camera_intrinsics,
)
from pick_and_place.image_rectify import rectified_square_camera_matrix
from pick_and_place.scene import build_environment


def _as_float_matrix(value: Any, rows: int, cols: int) -> list[list[float]] | None:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    if len(value) == rows and all(isinstance(row, list) for row in value):
        matrix = value
    elif len(value) == rows * cols:
        matrix = [value[i * cols : (i + 1) * cols] for i in range(rows)]
    else:
        return None
    return [[float(item) for item in row[:cols]] for row in matrix[:rows]]


def cv_world_to_camera_matrix(
    camera_position: Any,
    camera_rotation: Any,
) -> list[list[float]]:
    """Return OpenCV-style world-to-camera matrix from MuJoCo camera pose."""
    pos = [float(v) for v in camera_position]
    rot = [[float(v) for v in row] for row in camera_rotation]
    transform = [
        [rot[0][0], rot[1][0], rot[2][0]],
        [-rot[0][1], -rot[1][1], -rot[2][1]],
        [-rot[0][2], -rot[1][2], -rot[2][2]],
    ]
    return [row + [-sum(row[i] * pos[i] for i in range(3))] for row in transform] + [
        [0.0, 0.0, 0.0, 1.0]
    ]


def _camera_key(camera_name: str) -> str:
    return camera_name.removesuffix("_camera")


def _intrinsics_by_name() -> dict[str, dict[str, Any]]:
    intrinsics = {name: dict(value) for name, value in CAMERA_INTRINSICS_BY_NAME.items()}
    intrinsics.update(load_local_camera_intrinsics())
    return intrinsics


def export_camera_calibrations(square_size: int | None = None) -> dict[str, dict[str, Any]]:
    """Build the calibrated environment and return generic camera calibration.

    By default the intrinsics describe the raw, lens-distorted camera at its
    native resolution, matching what a recorded (unconverted) dataset's video
    pixels are. Pass ``square_size`` to instead describe the rectified,
    center-cropped square that ``convert_dataset_resolution.py`` produces --
    extrinsics are unaffected by that conversion, but the pinhole intrinsics
    (focal length, principal point, image size) are not the same matrix once
    the frame has been undistorted, cropped, and resized.
    """
    spec = build_environment()
    model = spec.compile()
    data = mujoco.MjData(model)
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)

    intrinsics_by_name = _intrinsics_by_name()
    calibrations: dict[str, dict[str, Any]] = {}
    for camera_id in range(model.ncam):
        name = model.camera(camera_id).name
        intrinsics = intrinsics_by_name.get(name)
        if intrinsics is None:
            continue
        camera_matrix = _as_float_matrix(intrinsics.get("camera_matrix"), 3, 3)
        if camera_matrix is None:
            continue

        width = intrinsics.get("width")
        height = intrinsics.get("height")
        if square_size is not None:
            payload_intrinsics = rectified_square_camera_matrix(intrinsics, square_size)
            image_size = [square_size, square_size]
        else:
            payload_intrinsics = camera_matrix
            image_size = (
                [int(width), int(height)] if width is not None and height is not None else None
            )

        payload: dict[str, Any] = {
            "camera": name,
            "intrinsics": payload_intrinsics,
            "world_to_camera": cv_world_to_camera_matrix(
                data.cam_xpos[camera_id].tolist(),
                data.cam_xmat[camera_id].reshape(3, 3).tolist(),
            ),
        }
        if image_size is not None:
            payload["image_size"] = image_size
        calibrations[_camera_key(name)] = payload
    return calibrations


def write_camera_calibrations(path: Path, square_size: int | None = None) -> Path:
    """Write generic camera calibration JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export_camera_calibrations(square_size), indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/camera_calibrations.json"),
        help="output JSON path",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=None,
        help=(
            "emit intrinsics for the rectified square crop convert_dataset_resolution.py "
            "produces (e.g. 512), instead of the raw native-resolution camera"
        ),
    )
    args = parser.parse_args()
    path = write_camera_calibrations(args.output, args.square_size)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
