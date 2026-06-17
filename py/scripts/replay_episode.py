#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reconstruct a recorded episode from its logged ``qpos`` — no frames are stored.

``record_episodes.py`` logs the full per-frame ``qpos`` instead of camera images,
so the run can be replayed exactly afterwards. This rebuilds the identical scene
(the cube placed at the episode's recorded start pose), then plays the logged
``qpos`` back: either rendering a camera to an mp4 (``--video out.mp4``) or
stepping through it live in the MuJoCo viewer (the default).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics


def _rebuild_model(cube_start: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Recompile the scene with the cube at the episode's recorded start pose.

    Matches ``pick_and_place.episodes`` exactly so the ``qpos`` layout lines up;
    the cube's free joint is what makes the logged ``qpos`` 13-wide.
    """
    spec = build_scene(include_environment=True)
    
    # Apply local calibration if present
    apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    intrinsics = load_local_camera_intrinsics()
    for camera in spec.cameras:
        if camera.name in intrinsics and "fovy_deg" in intrinsics[camera.name]:
            camera.fovy = float(intrinsics[camera.name]["fovy_deg"])

    cube = spec.body("pick_cube")
    cube.pos = (float(cube_start[0]), float(cube_start[1]), float(cube_start[2]))
    half_yaw = float(cube_start[3]) / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()
    model = spec.compile()
    return model, mujoco.MjData(model)


def _render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    out_path: Path,
    camera: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    import imageio.v2 as imageio

    renderer = mujoco.Renderer(model, height=height, width=width)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    renderer.close()
    print(f"Wrote {len(qpos)} frames to {out_path}")


def _play_viewer(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, fps: float) -> None:
    import time

    import mujoco.viewer

    period = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame in qpos:
                if not viewer.is_running():
                    break
                start = time.time()
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                viewer.sync()
                remaining = period - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="path to an episode .npz")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="render to this mp4 instead of opening the viewer",
    )
    parser.add_argument("--camera", default="wrist_camera", help="camera to render (for --video)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="playback fps (default: the episode's recorded control_hz)",
    )
    args = parser.parse_args()

    record = np.load(args.episode, allow_pickle=True)
    qpos = record["qpos"]
    fps = args.fps if args.fps is not None else float(record["control_hz"])
    model, data = _rebuild_model(record["cube_start"])
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width {qpos.shape[1]} != model.nq {model.nq}; scene mismatch")

    if args.video is not None:
        _render_video(model, data, qpos, args.video, args.camera, fps, args.width, args.height)
    else:
        _play_viewer(model, data, qpos, fps)


if __name__ == "__main__":
    main()
