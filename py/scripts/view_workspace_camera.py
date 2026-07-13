#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Show real, simulated, and blended AprilTag-driven camera views.

The four workspace-frame AprilTags (IDs 12--15) continuously solve the real
camera pose.  That pose is applied to a dedicated ``workspace_camera`` in
MuJoCo, so the right-hand view follows the physical camera without needing a
saved extrinsics file.  An optional overhead camera is solved independently
from the same workspace tags; its cube AprilTags and drop-zone square update
their simulated counterparts live.

Example:

    cd py
    python scripts/view_workspace_camera.py --camera 0

Press ``f`` to freeze/unfreeze the workspace camera pose, ``s`` to save a
capture, or ``q``/Escape to quit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.cam_align_solve import parse_index_or_path, solve_camera_pose
from pick_and_place.camera_compare import RealSource, load_intrinsics
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.cube_detection import CUBE_TAG_IDS, CubeTracker
from pick_and_place.paper_detection import (
    PaperTracker,
    add_paper_target_marker,
    detect_paper_target,
    draw_paper_target,
    set_paper_target_marker,
)
from pick_and_place.scene import build_environment
from pick_and_place.workspace_overlays import (
    is_cube_drop_allowed,
    workspace_interior_corners_world,
)

WINDOW_TITLE = "workspace camera  (f freeze, +/- overlay, s save, q / Esc quit)"
OVERHEAD_WINDOW_TITLE = "overhead camera  (f freeze, +/- overlay, s save, q / Esc quit)"
WORKSPACE_TAG_IDS = frozenset((12, 13, 14, 15))


def _draw_tags(
    image: np.ndarray,
    detections: list,
    *,
    source_width: int,
    source_height: int,
) -> None:
    """Draw every detected tag; workspace-reference tags are green."""
    import cv2

    scale = np.array((image.shape[1] / source_width, image.shape[0] / source_height))
    for detection in detections:
        corners = np.round(np.asarray(detection.corners) * scale).astype(np.int32)
        tag_id = int(detection.tag_id)
        color = (0, 255, 0) if tag_id in WORKSPACE_TAG_IDS else (0, 165, 255)
        cv2.polylines(image, [corners.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        label_at = tuple(corners[0] + np.array((3, -5)))
        cv2.putText(
            image,
            str(tag_id),
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _camera_snapshot(model: mujoco.MjModel, data: mujoco.MjData, camera_id: int) -> dict:
    """Return a camera pose in both model-local and world coordinates."""
    return {
        "parent_relative_pos_m": model.cam_pos[camera_id].tolist(),
        "parent_relative_quat_wxyz": model.cam_quat[camera_id].tolist(),
        "world_pos_m": data.cam_xpos[camera_id].tolist(),
        "world_rotation_matrix": data.cam_xmat[camera_id].reshape(3, 3).tolist(),
        "fovy_deg": float(model.cam_fovy[camera_id]),
    }


def _save_capture(
    *,
    capture_dir: Path,
    cv2_module,
    workspace_raw: np.ndarray,
    overhead_raw: np.ndarray | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    workspace_camera_id: int,
    overhead_camera_id: int,
    workspace_intrinsics: Path,
    overhead_intrinsics: Path | None,
    cube_body_id: int,
    target,
) -> Path:
    """Save lossless source frames and all currently reconstructed scene poses."""
    capture_path = capture_dir / datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
    capture_path.mkdir(parents=True, exist_ok=False)

    def write_rgb(name: str, image: np.ndarray) -> None:
        path = capture_path / name
        if not cv2_module.imwrite(str(path), cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write capture image: {path}")

    write_rgb("workspace_raw.png", workspace_raw)
    if overhead_raw is not None:
        write_rgb("overhead_raw.png", overhead_raw)
    (capture_path / "workspace_intrinsics.json").write_text(workspace_intrinsics.read_text())
    if overhead_intrinsics is not None:
        (capture_path / "overhead_intrinsics.json").write_text(overhead_intrinsics.read_text())

    payload = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "images": {
            "workspace": "workspace_raw.png",
            "overhead": "overhead_raw.png" if overhead_raw is not None else None,
            "format": "lossless PNG, raw camera pixels before rectification or overlays",
        },
        "intrinsics": {
            "workspace": {
                "source": str(workspace_intrinsics),
                "capture_copy": "workspace_intrinsics.json",
            },
            "overhead": (
                {
                    "source": str(overhead_intrinsics),
                    "capture_copy": "overhead_intrinsics.json",
                }
                if overhead_intrinsics is not None
                else None
            ),
        },
        "cameras": {
            "workspace_camera": _camera_snapshot(model, data, workspace_camera_id),
            "overhead_camera": _camera_snapshot(model, data, overhead_camera_id),
        },
        "cube": {
            "world_pos_m": data.xpos[cube_body_id].tolist(),
            "world_rotation_matrix": data.xmat[cube_body_id].reshape(3, 3).tolist(),
        },
        "target": None,
    }
    if target is not None:
        payload["target"] = {
            "center_world_m": target.center_world.tolist(),
            "corners_world_m": target.corners_world.tolist(),
            "center_pixel": target.center_px.tolist(),
            "corners_pixel": target.corners_px.tolist(),
            "yaw_rad": target.yaw,
        }
    (capture_path / "poses.json").write_text(json.dumps(payload, indent=2) + "\n")
    return capture_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", help="OpenCV camera index or device path")
    source.add_argument("--real-image", type=Path, help="single captured camera frame")
    parser.add_argument(
        "--overhead-camera",
        help="OpenCV camera index or device path for cube/target tracking",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="calibrated intrinsics JSON for this third camera",
    )
    parser.add_argument(
        "--overhead-intrinsics",
        type=Path,
        default=None,
        help="calibrated intrinsics JSON for --overhead-camera",
    )
    parser.add_argument("--width", type=int, default=960, help="display and render width")
    parser.add_argument("--height", type=int, default=540, help="display and render height")
    parser.add_argument("--camera-width", type=int, default=1920, help="capture width")
    parser.add_argument("--camera-height", type=int, default=1080, help="capture height")
    parser.add_argument("--camera-fps", type=int, default=30, help="requested capture frame rate")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="real-image contribution to the blended view (0..1)",
    )
    parser.add_argument(
        "--target-color",
        choices=("black", "white"),
        default="black",
        help="color of the drop-zone square to detect",
    )
    parser.add_argument(
        "--cube-smooth",
        type=float,
        default=0.3,
        help="cube pose EMA factor (0 = none, 1 = latest pose)",
    )
    parser.add_argument(
        "--target-smooth",
        type=float,
        default=0.3,
        help="target pose EMA factor (0 = none, 1 = latest pose)",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("out/workspace_camera_captures"),
        help="directory where the s key writes lossless images and poses",
    )
    args = parser.parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be between 0 and 1")
    overlay_alpha = args.overlay_alpha

    try:
        import cv2
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise SystemExit(
            "this viewer requires opencv-python and pupil-apriltags; "
            "install project dependencies first"
        ) from exc

    intrinsics = args.intrinsics
    if intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / "workspace_camera.json"
        intrinsics = candidate if candidate.exists() else None
    if intrinsics is None:
        raise SystemExit(
            "calibrated intrinsics are required; pass --intrinsics PATH "
            "(or save config/camera_intrinsics/workspace_camera.json)"
        )

    overhead_intrinsics = args.overhead_intrinsics
    if args.overhead_camera is not None and overhead_intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / "overhead_camera.json"
        overhead_intrinsics = candidate if candidate.exists() else None
    if args.overhead_camera is not None and overhead_intrinsics is None:
        raise SystemExit(
            "overhead camera intrinsics are required; pass --overhead-intrinsics PATH "
            "(or save config/camera_intrinsics/overhead_camera.json)"
        )

    # The PnP solve and the displayed feed both use the same rectified pinhole
    # image.  Consequently the simulated camera's fovy describes that exact
    # image rather than the camera's distorted sensor image.
    display_matrix, _ = load_intrinsics(intrinsics, args.width, args.height, cv2)
    solve_matrix, solve_undistort_map = load_intrinsics(
        intrinsics, args.camera_width, args.camera_height, cv2
    )
    overhead_display_matrix = None
    overhead_solve_matrix = None
    overhead_undistort_map = None
    if overhead_intrinsics is not None:
        overhead_display_matrix, _ = load_intrinsics(
            overhead_intrinsics, args.width, args.height, cv2
        )
        overhead_solve_matrix, overhead_undistort_map = load_intrinsics(
            overhead_intrinsics, args.camera_width, args.camera_height, cv2
        )

    spec = build_environment(apriltag_cube=True)
    add_paper_target_marker(spec)
    spec.worldbody.add_camera(
        name="workspace_camera",
        pos=(0.3, -0.3, 0.6),
        quat=(0.92388, 0.382683, 0.0, 0.0),
        fovy=60.0,
    )
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.height)
    model = spec.compile()
    data = mujoco.MjData(model)
    workspace_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_camera"
    )
    if workspace_camera_id < 0:
        raise SystemExit("could not create 'workspace_camera'")
    overhead_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    if overhead_camera_id < 0:
        raise SystemExit("scene has no 'overhead_camera'")
    model.cam_fovy[workspace_camera_id] = float(
        np.degrees(2.0 * np.arctan((args.height / 2.0) / display_matrix[1, 1]))
    )
    if overhead_display_matrix is not None:
        model.cam_fovy[overhead_camera_id] = float(
            np.degrees(
                2.0 * np.arctan((args.height / 2.0) / overhead_display_matrix[1, 1])
            )
        )
    mujoco.mj_forward(model, data)

    workspace_nominal_pos = model.cam_pos[workspace_camera_id].copy()
    workspace_nominal_quat = model.cam_quat[workspace_camera_id].copy()
    overhead_nominal_pos = model.cam_pos[overhead_camera_id].copy()
    overhead_nominal_quat = model.cam_quat[overhead_camera_id].copy()
    detector = Detector(families="tagStandard41h12", nthreads=4, refine_edges=True)
    cube_tracker = CubeTracker(smooth=args.cube_smooth)
    target_tracker = PaperTracker(alpha=args.target_smooth)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    if cube_body_id < 0:
        raise SystemExit("scene has no 'pick_cube' body")
    workspace_corners = workspace_interior_corners_world()
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    renderer._scene_option.geomgroup[1] = True
    source_view = RealSource(
        image_path=args.real_image,
        camera=parse_index_or_path(args.camera) if args.camera is not None else None,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        cv2_module=cv2,
    )
    overhead_view = None
    if args.overhead_camera is not None:
        overhead_view = RealSource(
            image_path=None,
            camera=parse_index_or_path(args.overhead_camera),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            cv2_module=cv2,
        )

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    if overhead_view is not None:
        cv2.namedWindow(OVERHEAD_WINDOW_TITLE, cv2.WINDOW_NORMAL)
    workspace_pose_frozen = False
    last_workspace_result = None
    try:
        while True:
            raw_frame = source_view.read(args.camera_width, args.camera_height)
            rectified_full = cv2.remap(raw_frame, *solve_undistort_map, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(rectified_full, cv2.COLOR_RGB2GRAY)
            detections = detector.detect(gray)
            if workspace_pose_frozen:
                result = last_workspace_result
            else:
                result = solve_camera_pose(
                    frame_rgb=rectified_full,
                    model=model,
                    data=data,
                    camera_name="workspace_camera",
                    matrix=solve_matrix,
                    dist=np.zeros(5),
                    detector=detector,
                    detections=detections,
                    min_workspace_tags=1,
                    cv2_module=cv2,
                    nominal_pos=workspace_nominal_pos,
                    nominal_quat=workspace_nominal_quat,
                )
                if result is not None:
                    last_workspace_result = result

            overhead_result = None
            overhead_detections = []
            cube_pose = None
            target = None
            overhead_view_rgb = None
            overhead_raw = None
            if overhead_view is not None:
                overhead_raw = overhead_view.read(args.camera_width, args.camera_height)
                overhead_full = cv2.remap(
                    overhead_raw, *overhead_undistort_map, cv2.INTER_LINEAR
                )
                overhead_gray = cv2.cvtColor(overhead_full, cv2.COLOR_RGB2GRAY)
                overhead_detections = detector.detect(overhead_gray)
                overhead_result = solve_camera_pose(
                    frame_rgb=overhead_full,
                    model=model,
                    data=data,
                    camera_name="overhead_camera",
                    matrix=overhead_solve_matrix,
                    dist=np.zeros(5),
                    detector=detector,
                    detections=overhead_detections,
                    min_workspace_tags=1,
                    cv2_module=cv2,
                    nominal_pos=overhead_nominal_pos,
                    nominal_quat=overhead_nominal_quat,
                )
                overhead_position = data.cam_xpos[overhead_camera_id]
                overhead_rotation = data.cam_xmat[overhead_camera_id].reshape(3, 3)
                cube_pose = cube_tracker.update(
                    [
                        detection
                        for detection in overhead_detections
                        if detection.tag_id in CUBE_TAG_IDS
                    ],
                    overhead_solve_matrix,
                    overhead_position,
                    overhead_rotation,
                )
                if cube_pose is not None and not cube_pose.held:
                    quat = np.empty(4)
                    mujoco.mju_mat2Quat(quat, cube_pose.rotation.reshape(-1))
                    model.body_pos[cube_body_id] = cube_pose.position
                    model.body_quat[cube_body_id] = quat

                raw_target = detect_paper_target(
                    overhead_full,
                    overhead_solve_matrix,
                    overhead_position,
                    overhead_rotation,
                    target_color=args.target_color,
                    workspace_corners_world=workspace_corners,
                )
                target = target_tracker.update(raw_target)
                if target is not None:
                    set_paper_target_marker(
                        model, data, target, usable=is_cube_drop_allowed(*target.xy)
                    )
                elif cube_pose is not None and not cube_pose.held:
                    mujoco.mj_forward(model, data)
                overhead_view_rgb = cv2.resize(
                    overhead_full, (args.width, args.height), interpolation=cv2.INTER_AREA
                )

            real_view = cv2.resize(
                rectified_full, (args.width, args.height), interpolation=cv2.INTER_AREA
            )
            renderer.update_scene(data, camera="workspace_camera")
            sim_view = renderer.render()
            if workspace_pose_frozen and result is None:
                pose_status = "Workspace pose frozen before its first solve"
            elif workspace_pose_frozen:
                pose_status = (
                    f"Frozen tags {list(result.used_tags)}  "
                    f"reprojection {result.reprojection_error_px:.1f}px"
                )
            elif result is None:
                pose_status = "Waiting for any workspace tag 12, 13, 14, or 15"
            else:
                pose_status = (
                    f"Tags {list(result.used_tags)}  "
                    f"reprojection {result.reprojection_error_px:.1f}px"
                )

            workspace_status = pose_status

            real_bgr = cv2.cvtColor(real_view, cv2.COLOR_RGB2BGR)
            sim_bgr = cv2.cvtColor(sim_view, cv2.COLOR_RGB2BGR)
            left = real_bgr
            _draw_tags(
                left,
                detections,
                source_width=args.camera_width,
                source_height=args.camera_height,
            )
            right = sim_bgr
            blended = cv2.addWeighted(
                real_bgr,
                overlay_alpha,
                sim_bgr,
                1.0 - overlay_alpha,
                0.0,
            )
            _draw_tags(
                blended,
                detections,
                source_width=args.camera_width,
                source_height=args.camera_height,
            )
            cv2.setWindowTitle(
                WINDOW_TITLE,
                f"{WINDOW_TITLE} — {workspace_status} — overlay {overlay_alpha:.0%}",
            )
            cv2.imshow(WINDOW_TITLE, np.hstack((left, right, blended)))
            if overhead_view_rgb is not None:
                renderer.update_scene(data, camera="overhead_camera")
                overhead_sim_rgb = renderer.render()
                if overhead_result is None:
                    overhead_pose_status = "Waiting for any workspace tag 12, 13, 14, or 15"
                else:
                    overhead_pose_status = (
                        f"Tags {list(overhead_result.used_tags)}  "
                        f"reprojection {overhead_result.reprojection_error_px:.1f}px"
                    )
                cube_status = "cube: no tags" if cube_pose is None else (
                    f"cube: {cube_pose.num_faces} face(s), reproj {cube_pose.reproj_px:.1f}px"
                )
                target_status = "target: not seen" if target is None else (
                    f"target: ({target.xy[0]:.3f}, {target.xy[1]:.3f})"
                )
                overhead_status = "\n".join(
                    (overhead_pose_status, cube_status, target_status)
                )
                overhead_real_bgr = cv2.cvtColor(overhead_view_rgb, cv2.COLOR_RGB2BGR)
                overhead_sim_bgr = cv2.cvtColor(overhead_sim_rgb, cv2.COLOR_RGB2BGR)
                overhead_left = overhead_real_bgr
                _draw_tags(
                    overhead_left,
                    overhead_detections,
                    source_width=args.camera_width,
                    source_height=args.camera_height,
                )
                if target is not None:
                    draw_paper_target(
                        overhead_left,
                        target,
                        args.width / args.camera_width,
                        args.height / args.camera_height,
                    )
                overhead_right = overhead_sim_bgr
                overhead_blended = cv2.addWeighted(
                    overhead_real_bgr,
                    overlay_alpha,
                    overhead_sim_bgr,
                    1.0 - overlay_alpha,
                    0.0,
                )
                _draw_tags(
                    overhead_blended,
                    overhead_detections,
                    source_width=args.camera_width,
                    source_height=args.camera_height,
                )
                if target is not None:
                    draw_paper_target(
                        overhead_blended,
                        target,
                        args.width / args.camera_width,
                        args.height / args.camera_height,
                    )
                cv2.setWindowTitle(
                    OVERHEAD_WINDOW_TITLE,
                    f"{OVERHEAD_WINDOW_TITLE} — {overhead_status.replace(chr(10), '; ')} "
                    f"— overlay {overlay_alpha:.0%}",
                )
                cv2.imshow(
                    OVERHEAD_WINDOW_TITLE,
                    np.hstack((overhead_left, overhead_right, overhead_blended)),
                )
            key = cv2.waitKey(0 if args.real_image is not None else 1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("f"):
                if workspace_pose_frozen:
                    workspace_pose_frozen = False
                    print("Workspace camera pose unfrozen")
                elif last_workspace_result is None:
                    print("Workspace camera pose cannot freeze before a successful solve")
                else:
                    workspace_pose_frozen = True
                    print("Workspace camera pose frozen")
            if key == ord("s"):
                try:
                    capture_path = _save_capture(
                        capture_dir=args.capture_dir,
                        cv2_module=cv2,
                        workspace_raw=raw_frame,
                        overhead_raw=overhead_raw,
                        model=model,
                        data=data,
                        workspace_camera_id=workspace_camera_id,
                        overhead_camera_id=overhead_camera_id,
                        workspace_intrinsics=intrinsics,
                        overhead_intrinsics=overhead_intrinsics,
                        cube_body_id=cube_body_id,
                        target=target,
                    )
                    print(f"Saved capture: {capture_path}")
                except (OSError, RuntimeError) as exc:
                    print(f"Could not save capture: {exc}")
            if key in (ord("+"), ord("=")):
                overlay_alpha = min(1.0, overlay_alpha + 0.05)
            elif key in (ord("-"), ord("_")):
                overlay_alpha = max(0.0, overlay_alpha - 0.05)
    finally:
        renderer.close()
        source_view.close()
        if overhead_view is not None:
            overhead_view.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
