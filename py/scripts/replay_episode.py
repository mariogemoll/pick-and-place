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

from pick_and_place.sim.scene import build_scene
from pick_and_place.core.camera_calibration import load_local_camera_extrinsics
from pick_and_place.sim.camera_extrinsics import apply_camera_extrinsics_to_spec
from pick_and_place.core.camera_calibration import load_local_camera_intrinsics
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed, is_cube_pickup_allowed


def _add_target_marker(spec: mujoco.MjSpec, target: np.ndarray | None) -> None:
    """Add a visible, non-colliding floor marker for the requested drop target."""
    if target is None:
        return

    x, y = float(target[0]), float(target[1])
    body = spec.worldbody.add_body(name="replay_target_marker", pos=(x, y, 0.0))
    body.add_geom(
        name="replay_target_marker_square",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.002),
        size=(CUBE_HALF_SIZE, CUBE_HALF_SIZE, 0.001),
        rgba=(0.0, 0.95, 0.35, 0.65),
        contype=0,
        conaffinity=0,
    )
    body.add_geom(
        name="replay_target_marker_pin",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.04),
        size=(0.004, 0.04),
        rgba=(0.0, 0.45, 1.0, 0.85),
        contype=0,
        conaffinity=0,
    )


def _record_target(record: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "target" in record:
        return np.asarray(record["target"], dtype=float)
    if "cube_target" in record:
        return np.asarray(record["cube_target"], dtype=float)
    return None


def _add_collision_markers(
    spec: mujoco.MjSpec,
    events: np.ndarray | None,
    *,
    limit: int,
) -> None:
    """Add visible world-space markers for saved collision contact positions."""
    if events is None or len(events) == 0 or limit <= 0:
        return

    marker_count = min(limit, len(events))
    # Spread markers across the event list so long contacts show their path, and
    # add the deepest penetration as a larger yellow marker.
    indices = np.unique(np.linspace(0, len(events) - 1, marker_count, dtype=int))
    for marker_index, event_index in enumerate(indices):
        event = events[event_index]
        spec.worldbody.add_geom(
            name=f"replay_collision_marker_{marker_index:03d}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=(float(event["x"]), float(event["y"]), float(event["z"]) + 0.003),
            size=(0.004,),
            rgba=(1.0, 0.05, 0.0, 0.85),
            contype=0,
            conaffinity=0,
        )

    deepest = events[int(np.argmin(events["dist"]))]
    spec.worldbody.add_geom(
        name="replay_collision_marker_deepest",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(float(deepest["x"]), float(deepest["y"]), float(deepest["z"]) + 0.006),
        size=(0.007,),
        rgba=(1.0, 0.95, 0.0, 0.95),
        contype=0,
        conaffinity=0,
    )


def _rebuild_model(
    cube_start: np.ndarray,
    target: np.ndarray | None = None,
    collision_events: np.ndarray | None = None,
    collision_marker_limit: int = 64,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Recompile the scene with the cube at the episode's recorded start pose.

    Matches ``pick_and_place.runtime.episodes`` exactly so the ``qpos`` layout lines up;
    the cube's free joint is what makes the logged ``qpos`` 13-wide.
    """
    spec = build_scene(include_environment=True)
    _add_target_marker(spec, target)
    _add_collision_markers(spec, collision_events, limit=collision_marker_limit)
    
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
        viewer.opt.geomgroup[4] = 1
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
    parser.add_argument(
        "--collision-marker-limit",
        type=int,
        default=64,
        help="maximum saved collision contact points to mark in the replay scene",
    )
    args = parser.parse_args()

    record = np.load(args.episode, allow_pickle=True)
    qpos = record["qpos"]
    fps = args.fps if args.fps is not None else float(record["control_hz"])
    target = _record_target(record)
    if target is not None:
        x, y = float(target[0]), float(target[1])
        print(
            f"target=({x:.4f}, {y:.4f}) "
            f"vertical_grip_allowed={is_cube_pickup_allowed(x, y)} "
            f"drop_allowed={is_cube_drop_allowed(x, y)}"
        )
    collision_events = record["collision_events"] if "collision_events" in record else None
    if collision_events is not None and len(collision_events):
        deepest = collision_events[int(np.argmin(collision_events["dist"]))]
        print(
            f"collision_events={len(collision_events)} deepest="
            f"{-float(deepest['dist']) * 1000.0:.2f}mm "
            f"{deepest['geom1']} <-> {deepest['geom2']} "
            f"at t={float(deepest['time']):.3f}s"
        )
    model, data = _rebuild_model(
        record["cube_start"],
        target,
        collision_events,
        args.collision_marker_limit,
    )
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width {qpos.shape[1]} != model.nq {model.nq}; scene mismatch")

    if args.video is not None:
        _render_video(model, data, qpos, args.video, args.camera, fps, args.width, args.height)
    else:
        _play_viewer(model, data, qpos, fps)


if __name__ == "__main__":
    main()
