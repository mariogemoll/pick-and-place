#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Decompose the vertical hand-eye error from measurement JSONs.

After the shoulder-pan zero correction, the remaining systematic hand-eye
error is dominated by a vertical offset. Several physical causes produce
different signatures in the world-frame delta as a function of arm pose:

- a constant world-frame offset (wrong table height / cube size / uniform
  sag): the same delta everywhere;
- a wrist-camera mount translation error: a delta that is constant in the
  camera frame, i.e. rotates with the wrist in world coordinates;
- a wrist-camera mount rotation error (e.g. pitch): a delta proportional to
  the camera-to-cube distance, directed perpendicular to the axis;
- gravity sag of the arm: a vertical delta growing with horizontal arm
  extension.

This script loads measurement JSONs written by ``measure_hand_eye_offset``
(whose frames must carry world-frame deltas), reconstructs the per-frame
geometry from each episode's ``pairs.json`` (sim wrist camera pose and cube
pose), fits each candidate model by least squares, and reports the fitted
parameters and residuals so the models can be compared. The winning
correction is what the pair exporter should bake in.

The delta convention follows the measurement: ``real - sim`` in world
coordinates. If the real camera pose is the modeled pose plus a translation
``t`` and a small rotation ``omega`` (camera frame), the measured delta is
``-t - R (omega x p_cam)`` where ``p_cam`` is the cube in the sim camera
frame; the fits below report ``t``/``omega`` in that "real minus modeled"
sense.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pick_and_place.scene import PICK_CUBE_HALF_SIZE


@dataclass
class Frame:
    day: str
    episode: str
    delta: np.ndarray  # world, meters, real - sim
    delta_px: np.ndarray  # wrist image pixels, real - sim
    cam_pos: np.ndarray  # world
    cam_rot: np.ndarray  # cam -> world (MuJoCo convention)
    cube: np.ndarray  # sim cube center, world


def load_frames(measurement_paths: list[Path]) -> list[Frame]:
    frames: list[Frame] = []
    for path in measurement_paths:
        with path.open() as f:
            summary = json.load(f)
        for episode in summary["episodes"]:
            episode_dir = Path(episode["name"])
            if not episode_dir.is_absolute():
                for base in (Path.cwd(), path.parent):
                    if (base / episode_dir / "pairs.json").is_file():
                        episode_dir = (base / episode_dir).resolve()
                        break
            with (episode_dir / "pairs.json").open() as f:
                index = json.load(f)
            by_frame = {fr["frame"]: fr for fr in index["frames"]}
            for record in episode["frames"]:
                if "delta_world_mm" not in record:
                    continue
                pair = by_frame[record["frame"]]
                wrist_cam = pair.get("wrist_cam")
                if wrist_cam is None:
                    continue
                cube = pair["cube"]
                frames.append(
                    Frame(
                        day=episode_dir.parent.name,
                        episode=episode_dir.name,
                        delta=np.asarray(record["delta_world_mm"], dtype=float) / 1000.0,
                        delta_px=np.asarray(record["delta_px"], dtype=float),
                        cam_pos=np.asarray(wrist_cam["pos"], dtype=float),
                        cam_rot=np.asarray(wrist_cam["mat"], dtype=float).reshape(3, 3),
                        cube=np.array([cube["x"], cube["y"], PICK_CUBE_HALF_SIZE]),
                    )
                )
    return frames


def _design(frames: list[Frame], *, t_frame: str, rotation: bool) -> np.ndarray:
    """Design matrix mapping model parameters to predicted world deltas."""
    blocks = []
    for fr in frames:
        cols = []
        if t_frame == "world":
            cols.append(-np.eye(3))
        elif t_frame == "camera":
            cols.append(-fr.cam_rot)
        if rotation:
            rel = fr.cube - fr.cam_pos
            # -R (omega x p_cam) = -(R e_j) x rel per camera-frame axis e_j
            cols.append(np.stack([-np.cross(fr.cam_rot[:, j], rel) for j in range(3)], axis=1))
        blocks.append(np.concatenate(cols, axis=1))
    return np.concatenate(blocks, axis=0)


def _view_rays(frames: list[Frame]) -> np.ndarray:
    rays = np.stack([fr.cube - fr.cam_pos for fr in frames])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _fit(
    frames: list[Frame], *, t_frame: str, rotation: bool, transverse_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """LSQ fit; with ``transverse_only`` the depth (view-ray) component is
    projected out of both the data and the model, so only the image-plane
    evidence constrains the fit."""
    design = _design(frames, t_frame=t_frame, rotation=rotation)
    target = np.concatenate([fr.delta for fr in frames])
    if transverse_only:
        rays = _view_rays(frames)
        projector = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
        n_params = design.shape[1]
        design = (projector @ design.reshape(-1, 3, n_params)).reshape(-1, n_params)
        target = (projector @ target.reshape(-1, 3, 1)).reshape(-1)
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = (target - design @ params).reshape(-1, 3)
    return params, residual


def _report(name: str, params_desc: list[str], params: np.ndarray, residual: np.ndarray) -> None:
    rms = np.sqrt((residual**2).sum(axis=1).mean()) * 1000.0
    rms_z = np.sqrt((residual[:, 2] ** 2).mean()) * 1000.0
    desc = ", ".join(f"{d}={v:+.2f}" for d, v in zip(params_desc, params))
    print(f"  {name}: rms={rms:.1f}mm rms_z={rms_z:.1f}mm  {desc}")


MODELS = {
    "t_world (const world offset)": dict(t_frame="world", rotation=False),
    "t_camera (mount translation)": dict(t_frame="camera", rotation=False),
    "omega (mount rotation)": dict(t_frame="none", rotation=True),
    "t_world + omega": dict(t_frame="world", rotation=True),
    "t_camera + omega": dict(t_frame="camera", rotation=True),
}


def _param_names(t_frame: str, rotation: bool) -> list[str]:
    names = []
    if t_frame == "world":
        names += ["tx_mm", "ty_mm", "tz_mm"]
    elif t_frame == "camera":
        names += ["tcx_mm", "tcy_mm", "tcz_mm"]
    if rotation:
        names += ["wx_deg", "wy_deg", "wz_deg"]
    return names


def _scaled(params: np.ndarray, t_frame: str, rotation: bool) -> np.ndarray:
    scaled = np.array(params, dtype=float)
    i = 0
    if t_frame in ("world", "camera"):
        scaled[:3] *= 1000.0
        i = 3
    if rotation:
        scaled[i : i + 3] = np.degrees(scaled[i : i + 3])
    return scaled


def analyze(frames: list[Frame], label: str) -> None:
    deltas = np.stack([fr.delta for fr in frames]) * 1000.0
    ext = np.array([math.hypot(fr.cam_pos[0], fr.cam_pos[1]) for fr in frames])
    dist = np.array([np.linalg.norm(fr.cube - fr.cam_pos) for fr in frames])
    print(
        f"{label}: n={len(frames)} "
        f"median=({np.median(deltas[:, 0]):+.1f}, {np.median(deltas[:, 1]):+.1f}, "
        f"{np.median(deltas[:, 2]):+.1f})mm std=({deltas[:, 0].std():.1f}, "
        f"{deltas[:, 1].std():.1f}, {deltas[:, 2].std():.1f})mm"
    )
    slope_ext = np.polyfit(ext, deltas[:, 2], 1)
    slope_dist = np.polyfit(dist, deltas[:, 2], 1)
    print(
        f"  dz vs cam extension: {slope_ext[0]:+.1f} mm/m (intercept {slope_ext[1]:+.1f}mm, "
        f"ext range {ext.min():.2f}-{ext.max():.2f}m, r={np.corrcoef(ext, deltas[:, 2])[0, 1]:+.2f})"
    )
    print(
        f"  dz vs cam-cube dist: {slope_dist[0]:+.1f} mm/m (intercept {slope_dist[1]:+.1f}mm, "
        f"dist range {dist.min():.2f}-{dist.max():.2f}m, r={np.corrcoef(dist, deltas[:, 2])[0, 1]:+.2f})"
    )

    px = np.stack([fr.delta_px for fr in frames])
    print(
        f"  pixel delta median=({np.median(px[:, 0]):+.1f}, {np.median(px[:, 1]):+.1f})px "
        f"|median|={np.median(np.linalg.norm(px, axis=1)):.1f}px"
    )

    # Depth (radial) component: delta projected onto the camera->cube ray.
    # Tag-scale depth estimation errors live here; a slope against distance is
    # a real-vs-sim depth gain (e.g. tag size / intrinsics mismatch).
    rays = _view_rays(frames)
    radial = (np.stack([fr.delta for fr in frames]) * rays).sum(axis=1) * 1000.0
    gain, offset = np.polyfit(dist, radial, 1)
    print(
        f"  radial (depth) vs dist: gain={gain / 1000.0:+.3f} offset={offset:+.1f}mm "
        f"median={np.median(radial):+.1f}mm std={radial.std():.1f}mm "
        f"r={np.corrcoef(dist, radial)[0, 1]:+.2f}"
    )
    transverse = np.stack([fr.delta for fr in frames]) * 1000.0 - radial[:, None] * rays
    print(
        f"  transverse median=({np.median(transverse[:, 0]):+.1f}, "
        f"{np.median(transverse[:, 1]):+.1f}, {np.median(transverse[:, 2]):+.1f})mm "
        f"std=({transverse[:, 0].std():.1f}, {transverse[:, 1].std():.1f}, "
        f"{transverse[:, 2].std():.1f})mm"
    )
    for name, spec in MODELS.items():
        params, residual = _fit(frames, **spec)
        _report(name, _param_names(**spec), _scaled(params, **spec), residual)
    print("  transverse-only fits (depth component projected out):")
    for name, spec in MODELS.items():
        params, residual = _fit(frames, **spec, transverse_only=True)
        _report(name, _param_names(**spec), _scaled(params, **spec), residual)

    # Joint model: a constant world offset plus a radial measurement artifact
    # (depth gain and offset along the view ray). The view rays vary in
    # direction across frames, so a genuine world-frame z error is separable
    # from a pure depth-estimation bias.
    design = np.concatenate(
        [
            np.tile(np.eye(3), (len(frames), 1)),
            (rays * dist[:, None]).reshape(-1)[:, None],
            rays.reshape(-1)[:, None],
        ],
        axis=1,
    )
    target = np.concatenate([fr.delta for fr in frames])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = (target - design @ params).reshape(-1, 3)
    rms = np.sqrt((residual**2).sum(axis=1).mean()) * 1000.0
    print(
        f"  joint world offset + depth artifact: rms={rms:.1f}mm "
        f"b=({params[0] * 1000:+.1f}, {params[1] * 1000:+.1f}, {params[2] * 1000:+.1f})mm "
        f"depth gain={params[3]:+.3f} depth offset={params[4] * 1000:+.1f}mm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurements", type=Path, nargs="+", help="measure_hand_eye_offset --output JSON(s)"
    )
    args = parser.parse_args()

    frames = load_frames(args.measurements)
    if not frames:
        raise SystemExit("no frames with world deltas and camera poses found")

    days = sorted({fr.day for fr in frames})
    for day in days:
        analyze([fr for fr in frames if fr.day == day], day)
    if len(days) > 1:
        analyze(frames, "ALL")


if __name__ == "__main__":
    main()
