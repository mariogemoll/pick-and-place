#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reproduce a saved ``view_workspace_camera.py`` workspace capture.

The viewer consumes a capture directory written with the ``s`` key by
``view_workspace_camera.py``.  It rectifies the saved camera pixels exactly as
the live viewer did, reconstructs the saved MuJoCo scene poses, and shows
``real | reproduced | overlay`` in one window.

Example:

    cd py
    python scripts/replay_workspace_camera_capture.py \
        out/workspace_camera_captures/capture_20260712_123456_000000

Press ``+``/``-`` to change the real-image contribution to the overlay, and
``q`` or Escape to quit.

Pass ``--export output.png`` to write the rectified camera and 3D render side
by side, or add ``--export-layout 3d`` to write only the 3D render.
Pass ``--export-normal`` and ``--export-3d`` to write those views to separate
files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.camera_compare import load_intrinsics
from pick_and_place.paper_detection import add_paper_target_marker, place_paper_target_marker
from pick_and_place.scene import build_scene
from pick_and_place.trajectory import REST_ARM_JOINTS
from pick_and_place.workspace_overlays import is_cube_drop_allowed

WINDOW_TITLE = "workspace capture replay  (+/- overlay, q / Esc quit)"
EXPORT_LAYOUTS = ("side-by-side", "normal", "3d")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return payload


def _array(payload: dict[str, Any], key: str, size: int) -> np.ndarray:
    try:
        value = np.asarray(payload[key], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"capture poses are missing a valid {key!r}") from exc
    if value.size != size:
        raise SystemExit(f"capture pose {key!r} must contain {size} values")
    return value.reshape(size)


def _load_capture(capture_dir: Path, cv2_module) -> tuple[np.ndarray, Path, dict[str, Any]]:
    poses_path = capture_dir / "poses.json"
    image_path = capture_dir / "workspace_raw.png"
    intrinsics_path = capture_dir / "workspace_intrinsics.json"
    for path in (poses_path, image_path, intrinsics_path):
        if not path.is_file():
            raise SystemExit(f"not a workspace-camera capture; missing {path}")

    image_bgr = cv2_module.imread(str(image_path), cv2_module.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"could not read workspace image: {image_path}")
    return cv2_module.cvtColor(image_bgr, cv2_module.COLOR_BGR2RGB), intrinsics_path, _read_json(
        poses_path
    )


def _restore_scene(
    poses: dict[str, Any], *, render_width: int, render_height: int
) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Build the environment and restore every workspace-camera capture pose."""
    try:
        camera = poses["cameras"]["workspace_camera"]
        cube = poses["cube"]
    except (KeyError, TypeError) as exc:
        raise SystemExit("capture poses do not contain workspace_camera and cube data") from exc
    if not isinstance(camera, dict) or not isinstance(cube, dict):
        raise SystemExit("capture camera and cube poses must be JSON objects")

    spec = build_scene(include_environment=True, tabletop=True, apriltag_cube=True)
    add_paper_target_marker(spec)
    spec.worldbody.add_camera(
        name="workspace_camera",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        fovy=float(camera.get("fovy_deg", 60.0)),
    )
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, render_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, render_height)
    model = spec.compile()
    data = mujoco.MjData(model)
    for joint_name, angle in REST_ARM_JOINTS.items():
        data.joint(joint_name).qpos = angle
    gripper = data.joint("gripper")
    gripper.qpos = model.jnt_range[gripper.id, 0]

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_camera")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    if camera_id < 0 or cube_id < 0:
        raise RuntimeError("reconstructed scene is missing the workspace camera or cube")
    model.cam_pos[camera_id] = _array(camera, "parent_relative_pos_m", 3)
    model.cam_quat[camera_id] = _array(camera, "parent_relative_quat_wxyz", 4)
    model.cam_fovy[camera_id] = float(camera["fovy_deg"])
    model.body_pos[cube_id] = _array(cube, "world_pos_m", 3)
    cube_rotation = _array(cube, "world_rotation_matrix", 9).reshape(3, 3)
    cube_quat = np.empty(4)
    mujoco.mju_mat2Quat(cube_quat, cube_rotation.reshape(-1))
    model.body_quat[cube_id] = cube_quat

    target = poses.get("target")
    if target is not None:
        if not isinstance(target, dict):
            raise SystemExit("capture target pose must be a JSON object or null")
        corners = _array(target, "corners_world_m", 12).reshape(4, 3)
        edge_x = corners[1] - corners[0]
        edge_y = corners[2] - corners[1]
        center = _array(target, "center_world_m", 3)
        place_paper_target_marker(
            model,
            (float(center[0]), float(center[1])),
            float(target["yaw_rad"]),
            (float(np.linalg.norm(edge_x[:2])) / 2.0, float(np.linalg.norm(edge_y[:2])) / 2.0),
            usable=is_cube_drop_allowed(float(center[0]), float(center[1])),
        )
    mujoco.mj_forward(model, data)
    return model, data, camera_id


def _status(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _crop_to_four_thirds(image: np.ndarray) -> np.ndarray:
    """Return a centered 4:3 crop without resampling the image."""
    height, width = image.shape[:2]
    if width * 3 > height * 4:
        crop_width = height * 4 // 3
        left = (width - crop_width) // 2
        return image[:, left : left + crop_width]
    crop_height = width * 3 // 4
    top = (height - crop_height) // 2
    return image[top : top + crop_height, :]


def _export_image(
    output: Path,
    layout: str,
    real_bgr: np.ndarray,
    sim_bgr: np.ndarray,
    cv2_module,
    *,
    jpeg_quality: int,
) -> None:
    """Write the requested camera/3D comparison image."""
    real_bgr = _crop_to_four_thirds(real_bgr)
    sim_bgr = _crop_to_four_thirds(sim_bgr)
    if layout == "side-by-side":
        image = np.hstack((real_bgr, sim_bgr))
    elif layout == "normal":
        image = real_bgr
    else:
        image = sim_bgr
    output.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        params = [
            cv2_module.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
            cv2_module.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2_module.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ]
    if not cv2_module.imwrite(str(output), image, params):
        raise SystemExit(f"could not write export image: {output}")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_dir", type=Path, help="directory written by view_workspace_camera.py"
    )
    parser.add_argument("--width", type=int, default=960, help="view width per panel")
    parser.add_argument("--height", type=int, default=540, help="view height per panel")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="initial real-image contribution to the overlay (0..1)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="PNG",
        help="write one replay image and exit",
    )
    parser.add_argument(
        "--export-layout",
        choices=EXPORT_LAYOUTS,
        default="side-by-side",
        help="export the rectified and 3D views side by side, or only the 3D view",
    )
    parser.add_argument(
        "--export-normal",
        type=Path,
        metavar="PNG",
        help="write the cropped rectified camera view and exit (can be combined with --export-3d)",
    )
    parser.add_argument(
        "--export-3d",
        type=Path,
        metavar="PNG",
        help="write the cropped 3D view and exit (can be combined with --export-normal)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG quality for .jpg/.jpeg exports (0..100; default: 100)",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be between 0 and 1")
    if not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 0 and 100")
    if args.export is not None and (args.export_normal is not None or args.export_3d is not None):
        parser.error("--export cannot be combined with --export-normal or --export-3d")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "this viewer requires opencv-python; install project dependencies first"
        ) from exc

    raw_rgb, intrinsics_path, poses = _load_capture(args.capture_dir, cv2)
    raw_height, raw_width = raw_rgb.shape[:2]
    _, remap = load_intrinsics(intrinsics_path, raw_width, raw_height, cv2)
    real_rgb = cv2.remap(raw_rgb, *remap, cv2.INTER_LINEAR)
    render_width, render_height = args.width, args.height
    if args.export is not None or args.export_normal is not None or args.export_3d is not None:
        # Render at the rectified camera's native aspect ratio, then crop both
        # views identically to 4:3. Rendering directly at 4:3 would squeeze the
        # camera image and change the simulated camera's horizontal field of view.
        scale = max(args.width / real_rgb.shape[1], args.height / real_rgb.shape[0])
        render_width = round(real_rgb.shape[1] * scale)
        render_height = round(real_rgb.shape[0] * scale)
    display_matrix, _ = load_intrinsics(intrinsics_path, render_width, render_height, cv2)
    real_rgb = cv2.resize(
        real_rgb, (render_width, render_height), interpolation=cv2.INTER_LANCZOS4
    )

    model, data, camera_id = _restore_scene(
        poses, render_width=render_width, render_height=render_height
    )
    model.cam_fovy[camera_id] = float(
        np.degrees(2.0 * np.arctan((render_height / 2.0) / display_matrix[1, 1]))
    )
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=render_width, height=render_height)
    # Frame collision boxes share faces with their visual counterparts. Keep the
    # collision-only group hidden so those coplanar surfaces cannot z-fight.
    renderer._scene_option.geomgroup[3] = False
    alpha = args.overlay_alpha
    if args.export is not None or args.export_normal is not None or args.export_3d is not None:
        try:
            renderer.update_scene(data, camera="workspace_camera")
            sim_rgb = renderer.render()
            real_bgr = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR)
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
            if args.export is not None:
                _export_image(
                    args.export,
                    args.export_layout,
                    real_bgr,
                    sim_bgr,
                    cv2,
                    jpeg_quality=args.jpeg_quality,
                )
            if args.export_normal is not None:
                _export_image(
                    args.export_normal,
                    "normal",
                    real_bgr,
                    sim_bgr,
                    cv2,
                    jpeg_quality=args.jpeg_quality,
                )
            if args.export_3d is not None:
                _export_image(
                    args.export_3d,
                    "3d",
                    real_bgr,
                    sim_bgr,
                    cv2,
                    jpeg_quality=args.jpeg_quality,
                )
        finally:
            renderer.close()
        return
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        while True:
            renderer.update_scene(data, camera="workspace_camera")
            sim_rgb = renderer.render()
            real_bgr = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR)
            sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(real_bgr, alpha, sim_bgr, 1.0 - alpha, 0.0)
            panels = (
                _status(real_bgr, "saved workspace camera (rectified)"),
                _status(sim_bgr, "reproduced MuJoCo scene"),
                _status(overlay, f"overlay (real {alpha:.0%}; +/- adjust)"),
            )
            cv2.imshow(WINDOW_TITLE, np.hstack(panels))
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                alpha = min(1.0, alpha + 0.05)
            elif key in (ord("-"), ord("_")):
                alpha = max(0.0, alpha - 0.05)
    finally:
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
