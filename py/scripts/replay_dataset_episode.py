#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Replay one LeRobotDataset episode in MuJoCo and overlay the sim render.

The dataset stores the real robot's measured joints as ``observation.state``,
the commanded set point as ``action``, the wrist/overhead videos, and
episode-level cube start/target metadata. This script rebuilds the calibrated
scene, seeds the cube at the recorded start pose, then writes comparison videos:

* ``--mode action`` drives the MuJoCo position servos from recorded actions and
  lets physics decide whether the cube is picked up.
* ``--mode state`` teleports the robot joints to the recorded measured states,
  useful as a visual calibration/timing ghost.
* ``--mode both`` writes both sets of overlays.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import mujoco
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pick_and_place import build_scene
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.executor import CONTROL_HZ, HARDWARE_SIMULATION_HZ
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    JOINT_NAMES,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    add_paper_target_marker,
    place_paper_target_marker,
)
from pick_and_place.robot_dynamics import set_actuator_activation
from pick_and_place.workspace_overlays import is_cube_drop_allowed

Mode = Literal["action", "state"]

CAMERA_TO_FEATURE = {
    "wrist_camera": "observation.images.wrist",
    "overhead_camera": "observation.images.overhead",
}


@dataclass(frozen=True)
class EpisodeRecord:
    index: int
    length: int
    data_path: Path
    video_paths: dict[str, Path]
    video_start_frames: dict[str, int]
    source_xy: tuple[float, float]
    source_z: float
    source_yaw: float
    target_xy: tuple[float, float] | None
    target_yaw: float
    states: np.ndarray
    actions: np.ndarray
    fps: float


def _finite(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _chunked_path(pattern: str, chunk_index: int, file_index: int) -> Path:
    return Path(pattern.format(chunk_index=chunk_index, file_index=file_index))


def _read_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def _read_episode_row(dataset_root: Path, episode_index: int) -> dict:
    episodes_dir = dataset_root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        mask = pc.equal(table["episode_index"], episode_index)
        filtered = table.filter(mask)
        if filtered.num_rows:
            return filtered.slice(0, 1).to_pylist()[0]
    raise ValueError(f"episode {episode_index} not found under {episodes_dir}")


def _read_episode_frames(data_path: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        data_path,
        columns=["episode_index", "observation.state", "action"],
    )
    table = table.filter(pc.equal(table["episode_index"], episode_index))
    if table.num_rows == 0:
        raise ValueError(f"episode {episode_index} has no rows in {data_path}")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    return states, actions


def _feature_video_path(
    dataset_root: Path,
    feature: str,
    episode_row: dict,
) -> tuple[Path, int]:
    prefix = f"videos/{feature}"
    chunk = int(episode_row[f"{prefix}/chunk_index"])
    file = int(episode_row[f"{prefix}/file_index"])
    from_timestamp = float(episode_row[f"{prefix}/from_timestamp"])
    # LeRobot video files use the same chunk/file pattern as data files.
    rel_path = Path("videos") / feature / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    return dataset_root / rel_path, round(from_timestamp * float(episode_row["_fps"]))


def _load_episode(dataset_root: Path, episode_index: int) -> EpisodeRecord:
    info = _read_info(dataset_root)
    row = _read_episode_row(dataset_root, episode_index)
    fps = float(info.get("fps", CONTROL_HZ))
    row["_fps"] = fps

    data_path = dataset_root / _chunked_path(
        info["data_path"],
        int(row["data/chunk_index"]),
        int(row["data/file_index"]),
    )
    states, actions = _read_episode_frames(data_path, episode_index)
    length = int(row["length"])
    if len(states) != length:
        raise ValueError(f"episode row says length={length}, but data has {len(states)} frame(s)")

    source_x = row.get("cube_start_x")
    source_y = row.get("cube_start_y")
    if not (_finite(source_x) and _finite(source_y)):
        raise ValueError(
            f"episode {episode_index} has no cube_start_x/y metadata; "
            "run the pose backfill first or pick an episode with source metadata"
        )
    source_z = float(row.get("cube_start_z") if _finite(row.get("cube_start_z")) else CUBE_HALF_SIZE)
    source_yaw = float(row.get("cube_start_yaw") if _finite(row.get("cube_start_yaw")) else 0.0)

    target_x = row.get("cube_target_x")
    target_y = row.get("cube_target_y")
    target_xy = (
        (float(target_x), float(target_y))
        if _finite(target_x) and _finite(target_y)
        else None
    )
    target_yaw = float(row.get("cube_target_yaw") if _finite(row.get("cube_target_yaw")) else 0.0)

    video_paths = {}
    video_start_frames = {}
    for camera, feature in CAMERA_TO_FEATURE.items():
        video_paths[camera], video_start_frames[camera] = _feature_video_path(
            dataset_root,
            feature,
            row,
        )

    return EpisodeRecord(
        index=episode_index,
        length=length,
        data_path=data_path,
        video_paths=video_paths,
        video_start_frames=video_start_frames,
        source_xy=(float(source_x), float(source_y)),
        source_z=source_z,
        source_yaw=source_yaw,
        target_xy=target_xy,
        target_yaw=target_yaw,
        states=states,
        actions=actions,
        fps=fps,
    )


def _joint_qpos_addrs(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: int(
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
        )
        for name in JOINT_NAMES
    }


def _actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in JOINT_NAMES
    }


def _cube_freejoint_addrs(model: mujoco.MjModel) -> tuple[int, int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    joint_id = int(model.body_jntadr[body_id])
    return int(model.jnt_qposadr[joint_id]), int(model.body_dofadr[body_id])


def _real_to_sim_vector(real_joints: np.ndarray) -> np.ndarray:
    arm_rad, gripper_rad = real_frame_to_sim(real_joints)
    return np.asarray([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad], dtype=float)


def _sim_to_real_vector(sim_joints: np.ndarray) -> np.ndarray:
    arm_rad = {name: float(sim_joints[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
    return sim_frame_to_real(arm_rad, float(sim_joints[-1]))


def _set_robot_qpos(
    data: mujoco.MjData,
    qpos_addrs: dict[str, int],
    sim_joints: np.ndarray,
) -> None:
    for i, name in enumerate(JOINT_NAMES):
        data.qpos[qpos_addrs[name]] = sim_joints[i]


def _set_ctrl(data: mujoco.MjData, actuator_ids: dict[str, int], sim_joints: np.ndarray) -> None:
    for i, name in enumerate(JOINT_NAMES):
        data.ctrl[actuator_ids[name]] = sim_joints[i]


def _seed_ctrl_and_activation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: dict[str, int],
    sim_joints: np.ndarray,
) -> None:
    _set_ctrl(data, actuator_ids, sim_joints)
    for i, name in enumerate(JOINT_NAMES):
        set_actuator_activation(model, data, actuator_ids[name], sim_joints[i])


def _get_robot_qpos(data: mujoco.MjData, qpos_addrs: dict[str, int]) -> np.ndarray:
    return np.asarray([data.qpos[qpos_addrs[name]] for name in JOINT_NAMES], dtype=float)


def _set_cube(
    data: mujoco.MjData,
    qadr: int,
    dadr: int,
    xy: tuple[float, float],
    z: float,
    yaw: float,
) -> None:
    half_yaw = yaw / 2.0
    data.qpos[qadr : qadr + 7] = (
        xy[0],
        xy[1],
        z,
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    data.qvel[dadr : dadr + 6] = 0.0


def _build_model(
    episode: EpisodeRecord,
    *,
    render_width: int,
    render_height: int,
    camera_calibration: bool,
    robot_dynamics: bool,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_scene(include_environment=True, robot_dynamics=robot_dynamics)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, render_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, render_height)
    add_paper_target_marker(spec)

    if camera_calibration:
        apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
        intrinsics = load_local_camera_intrinsics()
        for camera in spec.cameras:
            if camera.name in intrinsics and "fovy_deg" in intrinsics[camera.name]:
                camera.fovy = float(intrinsics[camera.name]["fovy_deg"])

    cube = spec.body("pick_cube")
    cube.pos = (episode.source_xy[0], episode.source_xy[1], episode.source_z)
    half_yaw = episode.source_yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()

    model = spec.compile()
    data = mujoco.MjData(model)
    if episode.target_xy is not None:
        place_paper_target_marker(
            model,
            episode.target_xy,
            episode.target_yaw,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*episode.target_xy),
            alpha=1.0,
        )
    return model, data


def _read_real_frames(
    video_path: Path,
    *,
    start_frame: int,
    count: int,
) -> list[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) != count:
        raise RuntimeError(
            f"read {len(frames)} frame(s) from {video_path}, expected {count} "
            f"starting at frame {start_frame}"
        )
    return frames


def _make_panel(real_bgr: np.ndarray, sim_bgr: np.ndarray, alpha: float) -> np.ndarray:
    blend = cv2.addWeighted(real_bgr, 1.0 - alpha, sim_bgr, alpha, 0.0)
    diff = cv2.absdiff(real_bgr, sim_bgr)
    return np.concatenate([real_bgr, sim_bgr, blend, diff], axis=1)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def _replay_mode(
    episode: EpisodeRecord,
    *,
    mode: Mode,
    cameras: list[str],
    output_dir: Path,
    alpha: float,
    panel: bool,
    max_frames: int | None,
    action_initial_state: str,
    camera_calibration: bool,
    robot_dynamics: bool,
) -> None:
    frame_count = episode.length if max_frames is None else min(episode.length, max_frames)
    first_camera_frames = _read_real_frames(
        episode.video_paths[cameras[0]],
        start_frame=episode.video_start_frames[cameras[0]],
        count=frame_count,
    )
    height, width = first_camera_frames[0].shape[:2]
    real_frames = {cameras[0]: first_camera_frames}
    for camera in cameras[1:]:
        real_frames[camera] = _read_real_frames(
            episode.video_paths[camera],
            start_frame=episode.video_start_frames[camera],
            count=frame_count,
        )
        if real_frames[camera][0].shape[:2] != (height, width):
            raise ValueError("all selected camera videos must have the same frame size")

    model, data = _build_model(
        episode,
        render_width=width,
        render_height=height,
        camera_calibration=camera_calibration,
        robot_dynamics=robot_dynamics,
    )
    renderer = mujoco.Renderer(model, width=width, height=height)
    qpos_addrs = _joint_qpos_addrs(model)
    actuator_ids = _actuator_ids(model)
    cube_qadr, cube_dadr = _cube_freejoint_addrs(model)
    _set_cube(data, cube_qadr, cube_dadr, episode.source_xy, episode.source_z, episode.source_yaw)

    first_state = _real_to_sim_vector(episode.states[0])
    first_action = _real_to_sim_vector(episode.actions[0])
    if mode == "action" and action_initial_state == "action":
        initial_qpos = first_action
    else:
        initial_qpos = first_state
    _set_robot_qpos(data, qpos_addrs, initial_qpos)
    _seed_ctrl_and_activation(model, data, actuator_ids, first_action)
    mujoco.mj_forward(model, data)

    out_size = (width * 4, height) if panel else (width, height)
    mode_label = mode
    if mode == "action" and robot_dynamics:
        mode_label = f"{mode}_mjcf_dynamics"
    writers = {
        camera: _open_writer(
            output_dir / f"episode_{episode.index:06d}_{camera}_{mode_label}_overlay.mp4",
            episode.fps,
            out_size,
        )
        for camera in cameras
    }

    simulation_steps_per_tick = round(HARDWARE_SIMULATION_HZ / episode.fps)
    state_errors = []
    try:
        for frame_index in range(frame_count):
            state = _real_to_sim_vector(episode.states[frame_index])
            action = _real_to_sim_vector(episode.actions[frame_index])
            if mode == "state":
                _set_robot_qpos(data, qpos_addrs, state)
                _set_ctrl(data, actuator_ids, state)
                mujoco.mj_forward(model, data)
            else:
                _set_ctrl(data, actuator_ids, action)
                mujoco.mj_step(model, data, nstep=simulation_steps_per_tick)
                sim_real = _sim_to_real_vector(_get_robot_qpos(data, qpos_addrs))
                state_errors.append(np.abs(sim_real - episode.states[frame_index]))

            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                sim_rgb = renderer.render()
                sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
                real_bgr = real_frames[camera][frame_index]
                frame = _make_panel(real_bgr, sim_bgr, alpha) if panel else cv2.addWeighted(
                    real_bgr,
                    1.0 - alpha,
                    sim_bgr,
                    alpha,
                    0.0,
                )
                writers[camera].write(frame)
    finally:
        for writer in writers.values():
            writer.release()
        renderer.close()

    if state_errors:
        errors = np.asarray(state_errors)
        mean = np.mean(errors, axis=0)
        peak = np.max(errors, axis=0)
        print(
            f"{mode} replay vs recorded state mean_abs={np.array2string(mean, precision=2)} "
            f"max_abs={np.array2string(peak, precision=2)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="LeRobotDataset root")
    parser.add_argument("episode_index", type=int, help="episode_index to replay")
    parser.add_argument(
        "--mode",
        choices=("action", "state", "both"),
        default="action",
        help="physics action replay, measured-state ghost, or both (default: action)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        choices=tuple(CAMERA_TO_FEATURE),
        help="camera to render; repeat for both (default: overhead_camera and wrist_camera)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("py/out/dataset_replay"))
    parser.add_argument("--alpha", type=float, default=0.45, help="sim opacity in blended output")
    parser.add_argument(
        "--panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write real | sim | blend | diff panels instead of only the blend",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="debug limit")
    parser.add_argument(
        "--action-initial-state",
        choices=("state", "action"),
        default="state",
        help=(
            "initial robot pose for --mode action: recorded first observation.state "
            "(default, matches the hardware start) or first action (stricter no-state seed)"
        ),
    )
    parser.add_argument(
        "--no-camera-calibration",
        action="store_true",
        help="render authored nominal cameras instead of applying local camera calibration",
    )
    parser.add_argument(
        "--no-robot-dynamics",
        action="store_true",
        help="use the raw upstream MuJoCo actuators instead of fitted MJCF actuator time constants",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")

    cameras = args.camera or ["overhead_camera", "wrist_camera"]
    modes: list[Mode] = ["action", "state"] if args.mode == "both" else [args.mode]
    episode = _load_episode(args.dataset_root, args.episode_index)

    print(
        f"episode={episode.index} frames={episode.length} fps={episode.fps:g} "
        f"cube_start=({episode.source_xy[0]:.4f}, {episode.source_xy[1]:.4f}, "
        f"yaw={episode.source_yaw:.3f})"
    )
    if episode.target_xy is not None:
        print(f"target=({episode.target_xy[0]:.4f}, {episode.target_xy[1]:.4f})")

    for mode in modes:
        _replay_mode(
            episode,
            mode=mode,
            cameras=cameras,
            output_dir=args.output_dir,
            alpha=args.alpha,
            panel=args.panel,
            max_frames=args.max_frames,
            action_initial_state=args.action_initial_state,
            camera_calibration=not args.no_camera_calibration,
            robot_dynamics=not args.no_robot_dynamics,
        )
    print(f"wrote overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
