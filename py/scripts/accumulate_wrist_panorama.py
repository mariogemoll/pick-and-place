#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Accumulate a wrist-camera background panorama from an episode index (slow pass).

Reads the handoff ``.npz`` written by ``index_wrist_episodes.py`` and, for every
recorded wrist frame, recovers the camera's world pose from the logged joints via
``mj_forward``, casts a ray through each pixel, and accumulates the pixel color
into the equirectangular direction it points at. Averaging every frame that lands
on a given texel denoises and softens residual parallax (the wrist translates as
it looks around), giving a blurry, room-like panorama to drape on a background
dome in the sim.

Rays pointing downward hit the tabletop (textured separately) and the gripper, so
only rays above ``--min-elevation-deg`` are kept: the outward, near-horizon
directions the dome actually needs. A bottom strip of each frame is dropped too,
to shed the gripper jaws. Episodes are grouped into the index's brightness bins,
one panorama set per bin.

For each bin, three images are written beside ``--out``: the mean panorama, a
coverage map, and a hole-filled version for direct use as a texture.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.camera_compare import load_intrinsics
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.follower import ARM_JOINT_NAMES, JOINT_NAMES, real_frame_to_sim

WRIST_CAMERA = "wrist_camera"
WRIST_W, WRIST_H = 1280, 720


def _real_to_sim_vector(real_joints: np.ndarray) -> np.ndarray:
    arm_rad, gripper_rad = real_frame_to_sim(real_joints)
    return np.asarray([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad], dtype=float)


def _build_wrist_model() -> tuple[mujoco.MjModel, mujoco.MjData, int, dict[str, int]]:
    """A scene whose wrist-camera pose tracks the logged joints via ``mj_forward``.

    The cube and background environment are irrelevant here (we only read the
    camera pose), so the plain scene with the wrist camera attached is enough.
    """
    spec = build_scene(include_environment=False, robot_dynamics=False)
    model = spec.compile()
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
    qpos_addrs = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    }
    return model, data, cam_id, qpos_addrs


def _pixel_ray_grid(rect_matrix: np.ndarray, stride: int, bottom_crop: float):
    """Per-pixel ray directions in the MuJoCo camera frame (+X right, +Y up, -Z fwd).

    Built from the rectified pinhole intrinsics. Rows below ``bottom_crop`` of the
    frame are dropped so the gripper jaws never enter the panorama.
    """
    fx, fy = rect_matrix[0, 0], rect_matrix[1, 1]
    cx, cy = rect_matrix[0, 2], rect_matrix[1, 2]
    max_v = int(WRIST_H * (1.0 - bottom_crop))
    us = np.arange(0, WRIST_W, stride, dtype=np.float64)
    vs = np.arange(0, max_v, stride, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    # OpenCV pixel ray (+X right, +Y down, +Z forward) → MuJoCo camera frame.
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    rays = np.stack([x, -y, -z], axis=-1).reshape(-1, 3)
    return rays, us.astype(int), vs.astype(int)


def _accumulate_frame(
    frame_rgb: np.ndarray,
    cam_xmat: np.ndarray,
    rays_cam: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    min_elev: float,
    acc: np.ndarray,
    count: np.ndarray,
) -> None:
    pano_h, pano_w = count.shape
    world = rays_cam @ cam_xmat.T
    world /= np.linalg.norm(world, axis=1, keepdims=True)
    elev = np.arcsin(np.clip(world[:, 2], -1.0, 1.0))
    keep = elev > min_elev
    if not np.any(keep):
        return
    world = world[keep]
    elev = elev[keep]
    azim = np.arctan2(world[:, 1], world[:, 0])
    col = ((azim + math.pi) / (2.0 * math.pi) * pano_w).astype(int) % pano_w
    row = np.clip(((math.pi / 2.0 - elev) / math.pi * pano_h).astype(int), 0, pano_h - 1)
    colors = frame_rgb[np.ix_(vs, us)].reshape(-1, 3).astype(np.float64)[keep]
    flat = row * pano_w + col
    np.add.at(acc.reshape(-1, 3), flat, colors)
    np.add.at(count.reshape(-1), flat, 1)


def _fill_holes(mean: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Blur-inpaint texels no ray ever hit, so the texture has no black gaps."""
    mask = (count == 0).astype(np.uint8)
    if mask.any():
        return cv2.inpaint(mean.astype(np.uint8), mask, 8, cv2.INPAINT_TELEA)
    return mean.astype(np.uint8)


def _gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Blur out the sparse-sampling lattice; a soft background is what we want anyway."""
    if sigma_px <= 0:
        return image
    ksize = int(2 * np.ceil(3 * sigma_px) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma_px)


def _write_outputs(
    out: Path, acc: np.ndarray, count: np.ndarray, frames: int, blur_px: float = 0.0
) -> None:
    hit = count > 0
    mean = np.zeros_like(acc)
    mean[hit] = acc[hit] / count[hit, None]
    mean = mean.astype(np.uint8)
    filled = _gaussian_blur(_fill_holes(mean, count), blur_px)
    coverage = (count.astype(np.float64) / max(1, count.max()) * 255).astype(np.uint8)
    coverage = cv2.applyColorMap(coverage, cv2.COLORMAP_VIRIDIS)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(mean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out.with_name(out.stem + "_filled.png")), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out.with_name(out.stem + "_coverage.png")), coverage)
    frac = 100.0 * hit.sum() / hit.size
    print(f"  {out.name}: {frames} frames, {frac:.1f}% covered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="handoff .npz from index_wrist_episodes.py")
    parser.add_argument("--out", type=Path, required=True, help="output panorama PNG path")
    parser.add_argument("--pano-width", type=int, default=2048, help="panorama width (height = width/2)")
    parser.add_argument("--pixel-stride", type=int, default=4, help="subsample every Nth pixel")
    parser.add_argument("--frame-stride", type=int, default=4, help="use every Nth video frame")
    parser.add_argument("--max-frames", type=int, default=45000, help="cap on frames accumulated (split across bins)")
    parser.add_argument("--min-elevation-deg", type=float, default=-15.0, help="drop rays below this elevation")
    parser.add_argument("--bottom-crop", type=float, default=0.15, help="drop this fraction off each frame's bottom")
    parser.add_argument(
        "--blur",
        type=float,
        default=0.006,
        help="Gaussian blur sigma (fraction of panorama width) for the filled texture; 0 disables",
    )
    parser.add_argument(
        "--wrist-intrinsics",
        type=Path,
        default=Path(LOCAL_CAMERA_INTRINSICS_DIR) / "wrist_camera.json",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="treat the recorded wrist frames as already rectified",
    )
    args = parser.parse_args()

    index = np.load(args.index, allow_pickle=True)
    states_all = index["states"]
    video_paths = index["video_paths"]
    start_frames = index["start_frames"]
    bin_of = index["bin"]
    brightness = index["brightness"]
    n_bins = int(index["n_bins"])

    pano_w = args.pano_width
    pano_h = pano_w // 2
    min_elev = math.radians(args.min_elevation_deg)
    budget_per_bin = args.max_frames // n_bins

    rect_matrix, undistort_map = load_intrinsics(args.wrist_intrinsics, WRIST_W, WRIST_H, cv2)
    rays_cam, us, vs = _pixel_ray_grid(rect_matrix, args.pixel_stride, args.bottom_crop)
    model, data, cam_id, qpos_addrs = _build_wrist_model()

    accs = [np.zeros((pano_h, pano_w, 3), dtype=np.float64) for _ in range(n_bins)]
    counts = [np.zeros((pano_h, pano_w), dtype=np.int64) for _ in range(n_bins)]
    frames_per_bin = [0] * n_bins

    for states, video_path, start, b in zip(states_all, video_paths, start_frames, bin_of):
        b = int(b)
        if frames_per_bin[b] >= budget_per_bin:
            continue
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for i in range(len(states)):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if i % args.frame_stride != 0 or frames_per_bin[b] >= budget_per_bin:
                continue
            if not args.no_undistort:
                frame_bgr = cv2.remap(frame_bgr, *undistort_map, cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sim_joints = _real_to_sim_vector(states[i])
            for j, name in enumerate(JOINT_NAMES):
                data.qpos[qpos_addrs[name]] = sim_joints[j]
            mujoco.mj_forward(model, data)
            cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)
            _accumulate_frame(frame_rgb, cam_xmat, rays_cam, us, vs, min_elev, accs[b], counts[b])
            frames_per_bin[b] += 1
        cap.release()

    for b in range(n_bins):
        out = args.out if n_bins == 1 else args.out.with_name(f"{args.out.stem}_b{b}{args.out.suffix}")
        in_bin = brightness[bin_of == b]
        if in_bin.size:
            print(f"bin {b}: {in_bin.size} eps, brightness {in_bin.min():.0f}..{in_bin.max():.0f}")
        _write_outputs(out, accs[b], counts[b], frames_per_bin[b], blur_px=args.blur * pano_w)


if __name__ == "__main__":
    main()
