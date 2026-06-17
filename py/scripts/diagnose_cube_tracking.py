#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Capture a burst of overhead frames and report what makes the cube pose jump.

With the cube held still this isolates the source of live jitter. It records, per
frame, how many cube faces are seen, their pixel size, the reprojection error,
whether the planar flip had to be disambiguated, and the recovered world pose;
then it reports the position/orientation spread, the largest frame-to-frame
jumps, and whether those jumps line up with the visible-face set changing or with
a single-face flip. Run it with the live viewer closed (one reader per camera):

    cd py
    python scripts/diagnose_cube_tracking.py --camera 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pick_and_place.cam_align_solve import parse_index_or_path
from pick_and_place.camera_compare import load_intrinsics
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.cube_detection import (
    cube_pose_to_world,
    detect_cube_faces,
    fuse_cube_faces,
    make_cube_detector,
)
from pick_and_place.scene import build_scene


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos_angle = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="OpenCV camera index or device path")
    parser.add_argument("--camera-name", default="overhead_camera")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="feed each pose back as the next frame's prior (the flip fix); off = raw per-frame",
    )
    args = parser.parse_args()

    spec = build_scene(wrist_camera=True, include_environment=True)
    model = spec.compile()
    data = mujoco.MjData(model)
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    mujoco.mj_forward(model, data)
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()

    intrinsics = args.intrinsics
    if intrinsics is None:
        candidate = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
        intrinsics = candidate if candidate.exists() else None
    if intrinsics is not None:
        camera_matrix, undistort_map = load_intrinsics(intrinsics, args.width, args.height, cv2)
    else:
        focal = (args.height / 2.0) / np.tan(np.radians(model.cam_fovy[camera_id]) / 2.0)
        camera_matrix = np.array(
            [[focal, 0, args.width / 2.0], [0, focal, args.height / 2.0], [0, 0, 1]], dtype=float
        )
        undistort_map = None
        print("Warning: no calibrated intrinsics; detecting on the raw frame")

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera!r} (is the viewer still running?)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = make_cube_detector()
    records = []
    prior_rotation = None
    try:
        for _ in range(args.frames):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if undistort_map is not None:
                rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
            detections = detect_cube_faces(rgb, detector)
            estimate = fuse_cube_faces(
                detections, camera_matrix, prior_rotation=prior_rotation if args.temporal else None
            )
            if estimate is None:
                records.append(None)
                continue
            if args.temporal:
                prior_rotation = estimate.rotation
            rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
            spans = [
                float(np.linalg.norm(np.asarray(d.corners).max(0) - np.asarray(d.corners).min(0)))
                for d in detections
            ]
            records.append(
                {
                    "ids": tuple(estimate.face_ids),
                    "faces": estimate.num_faces_used,
                    "span": min(spans) if spans else 0.0,
                    "reproj": estimate.reproj_px,
                    "candidates": estimate.num_candidates,
                    "position": position,
                    "rotation": rotation,
                }
            )
    finally:
        cap.release()

    poses = [r for r in records if r is not None]
    total = len(records)
    print(f"\nframes captured: {total}   with a pose: {len(poses)}   dropped: {total - len(poses)}")
    if len(poses) < 2:
        raise SystemExit("not enough detections to analyse; check camera/cube/lighting")

    face_sets = [r["ids"] for r in poses]
    from collections import Counter

    print("visible-face sets (id-tuple: count):")
    for ids, count in Counter(face_sets).most_common():
        print(f"  {ids}: {count}")
    set_changes = sum(1 for a, b in zip(face_sets, face_sets[1:]) if a != b)
    print(f"face-set changed between consecutive frames: {set_changes} time(s)")
    print(f"single-face frames (flip-prone): {sum(1 for r in poses if r['faces'] == 1)}")
    print(f"frames needing flip disambiguation (candidates==2): {sum(1 for r in poses if r['candidates'] == 2)}")
    print(f"min tag span: {min(r['span'] for r in poses):.0f}px   median reproj: {np.median([r['reproj'] for r in poses]):.2f}px")

    positions = np.array([r["position"] for r in poses])
    centre = positions.mean(0)
    pos_std_mm = positions.std(0) * 1000.0
    pos_rms_mm = float(np.sqrt((np.linalg.norm(positions - centre, axis=1) ** 2).mean()) * 1000.0)
    step_mm = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 1000.0
    print(
        f"\nposition: std(x,y,z)=({pos_std_mm[0]:.1f},{pos_std_mm[1]:.1f},{pos_std_mm[2]:.1f})mm "
        f"rms={pos_rms_mm:.1f}mm  max frame-to-frame step={step_mm.max():.1f}mm"
    )

    rotations = [r["rotation"] for r in poses]
    ref = rotations[len(rotations) // 2]
    devs = np.array([_angle_deg(ref, R) for R in rotations])
    steps = np.array([_angle_deg(a, b) for a, b in zip(rotations, rotations[1:])])
    print(
        f"orientation: spread(std)={devs.std():.1f}deg around median  "
        f"max frame-to-frame step={steps.max():.1f}deg  (>20deg steps = likely flips: {int((steps > 20).sum())})"
    )

    big = np.flatnonzero(steps > 20)
    if big.size:
        coincide = sum(1 for i in big if face_sets[i] != face_sets[i + 1] or poses[i + 1]["candidates"] == 2)
        print(f"  of {big.size} big orientation steps, {coincide} coincide with a face-set change or a flip candidate")

    print("\n--- read it like this ---")
    print("  dropped frames high            -> detection dropouts (lighting/exposure/tag size)")
    print("  many face-set changes + jumps  -> pose jumps when a face appears/disappears")
    print("  candidates==2 + >20deg steps   -> single-face flips (need a steadier disambiguation)")
    print("  small steps but high spread    -> corner-noise jitter (raise resolution / smoothing)")


if __name__ == "__main__":
    main()
