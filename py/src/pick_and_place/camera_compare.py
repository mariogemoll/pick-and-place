# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compare the real overhead camera with the calibrated MuJoCo camera.

This is a lightweight visual check after AprilTag extrinsics calibration. It
renders the generated environment through ``overhead_camera`` and blends the sim
render over either a live camera frame or a captured image.

Example:

    cd py
    python -m pick_and_place.camera_compare \
        --camera 0 \
        --intrinsics /path/to/overhead_intrinsics.json
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.cam_align_solve import parse_index_or_path
from pick_and_place.scene import build_environment

WINDOW_TITLE = "camera_compare  (m mode  , . alpha  q quit)"


def load_intrinsics(
    path: Path,
    width: int,
    height: int,
    cv2_module: Any,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Load calibrated intrinsics and create an undistort map for ``width``/``height``."""
    import json

    data = json.loads(path.read_text())
    matrix = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(data["dist_coeffs"], dtype=float)
    sx = width / float(data["width"])
    sy = height / float(data["height"])
    matrix[0, :] *= sx
    matrix[1, :] *= sy

    focal = float(matrix[1, 1])
    rect_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    undistort_map = cv2_module.initUndistortRectifyMap(
        matrix,
        dist,
        None,
        rect_matrix,
        (width, height),
        cv2_module.CV_16SC2,
    )
    return rect_matrix, undistort_map


class RealSource:
    """Provide RGB frames from a static image or OpenCV camera."""

    def __init__(
        self,
        *,
        image_path: Path | None,
        camera: int | str | None,
        width: int,
        height: int,
        fps: int,
        cv2_module: Any,
    ):
        self.width = width
        self.height = height
        self.cv2 = cv2_module
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None

        if image_path is not None:
            bgr = self.cv2.imread(str(image_path))
            if bgr is None:
                raise SystemExit(f"could not read image {image_path}")
            self._frame = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

        if camera is not None:
            self._cap = self.cv2.VideoCapture(camera)
            if not self._cap.isOpened():
                raise SystemExit(f"could not open camera {camera!r}")
            self._cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(self.cv2.CAP_PROP_FPS, fps)
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb

    def read(self, width: int, height: int) -> np.ndarray:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = self.cv2.resize(frame, (width, height), interpolation=self.cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()


def draw_hud(bgr: np.ndarray, *, mode: str, alpha: float, intrinsics: Path | None) -> np.ndarray:
    """Draw a compact status line over a BGR frame."""
    out = bgr.copy()
    label = f"mode={mode} alpha={alpha:.2f}  m toggle  , . alpha  q quit"
    if intrinsics is not None:
        label += f"  intrinsics={intrinsics.name}"
    cv2 = __import__("cv2")
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", help="OpenCV camera index or device path")
    source.add_argument("--real-image", type=Path, help="captured real overhead frame")
    parser.add_argument("--camera-name", default="overhead_camera")
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--width", type=int, default=960, help="overlay/render width")
    parser.add_argument("--height", type=int, default=540, help="overlay/render height")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    import cv2

    spec = build_environment()
    model = spec.compile()
    data = mujoco.MjData(model)
    applied = apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    if args.camera_name not in applied:
        print(f"Warning: no local extrinsics applied for {args.camera_name!r}")

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if camera_id < 0:
        raise SystemExit(f"unknown camera {args.camera_name!r}")

    requested_size = (args.width, args.height)
    framebuffer_size = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight))
    scale = min(
        1.0,
        framebuffer_size[0] / args.width,
        framebuffer_size[1] / args.height,
    )
    args.width = max(1, int(round(args.width * scale)))
    args.height = max(1, int(round(args.height * scale)))
    if requested_size != (args.width, args.height):
        print(
            f"Clamping render to {args.width}x{args.height} "
            f"(offscreen framebuffer limit {framebuffer_size[0]}x{framebuffer_size[1]})."
        )

    intrinsics = args.intrinsics
    if intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
        intrinsics = candidate if candidate.exists() else None

    undistort_map = None
    if intrinsics is not None:
        rect_matrix, undistort_map = load_intrinsics(intrinsics, args.width, args.height, cv2)
        rect_fy = float(rect_matrix[1, 1])
        model.cam_fovy[camera_id] = float(
            np.degrees(2.0 * np.arctan((args.height / 2.0) / rect_fy))
        )
    else:
        print("Warning: no calibrated intrinsics supplied; comparing raw frame to nominal pinhole")

    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    real = RealSource(
        image_path=args.real_image,
        camera=parse_index_or_path(args.camera) if args.camera is not None else None,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        cv2_module=cv2,
    )

    mode = "blend"
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = real.read(args.width, args.height)
            if undistort_map is not None:
                frame = cv2.remap(frame, *undistort_map, cv2.INTER_LINEAR)

            renderer.update_scene(data, camera=args.camera_name)
            sim = renderer.render()
            if mode == "edges":
                edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), 60, 160)
                out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out[edges > 0] = (0, 255, 0)
            else:
                blended = cv2.addWeighted(frame, alpha, sim, 1.0 - alpha, 0.0)
                out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            cv2.imshow(WINDOW_TITLE, draw_hud(out, mode=mode, alpha=alpha, intrinsics=intrinsics))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mode = "edges" if mode == "blend" else "blend"
            elif key == ord(","):
                alpha = float(np.clip(alpha - 0.05, 0.0, 1.0))
            elif key == ord("."):
                alpha = float(np.clip(alpha + 0.05, 0.0, 1.0))
            if args.real_image is not None and key == -1:
                cv2.waitKey(0)
    finally:
        renderer.close()
        real.close()
        cv2.destroyWindow(WINDOW_TITLE)


if __name__ == "__main__":
    main()
