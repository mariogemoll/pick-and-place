#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export aligned (sim render, real frame) image pairs from LeRobotDatasets.

For every frame of every episode, the real wrist/overhead video frames are
undistorted into the calibrated pinhole view and cropped/resized to the
requested output resolution, while the MuJoCo scene is posed kinematically from
the measured ``observation.state`` and rendered through the same calibrated
camera geometry. The cube is re-detected from the rectified overhead video
(AprilTag faces) and injected into the sim at the detected pose.

Raw single-tag detections carry a planar-pose ambiguity that makes the
estimated orientation flicker frame to frame. Since the cube is physically at
rest except while pushed or carried, detection runs as a first pass over the
whole episode and the track is then segmented into stable-position phases;
each phase gets one robust pose (median position, outlier-rejected circular
mean yaw preferring multi-face detections) that is held for its entire span.
Frames outside any stable phase (cube occluded or in the gripper) hold the
last known pose and are labelled ``hold`` in the index.

The default 960x720 output keeps the wrist camera's full rectified height, and
every crop the pipeline uses keeps the full frame height, so the policy input
domains (640x480 ACT, 512x512 SmolVLA, 96x96) are all center-crop + uniform
downscale derivations of these pairs.

Per episode the exporter writes::

    <out>/<dataset>/episode_XXXXXX/
        wrist_real/NNNNNN.jpg      wrist_sim/NNNNNN.jpg
        overhead_real/NNNNNN.jpg   overhead_sim/NNNNNN.jpg
        pairs.json                 per-frame cube pose, tracking label, gray-MAE
        montage_<camera>.mp4       real | sim | blend strips for eyeballing

The per-frame grayscale MAE between real and sim is a pair-quality signal: a
systematic bump while the arm moves indicates a time offset between the video
and the joint stream, and outlier frames can be filtered before training.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from pick_and_place import build_scene
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_spec,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.cube_detection import (
    cube_pose_to_world,
    detect_tags,
    estimate_cube_pose,
    make_cube_detector,
)
from pick_and_place.environment import WORKSPACE_FRAME_APRILTAG_PLATES
from pick_and_place.follower import ARM_JOINT_NAMES, JOINT_NAMES, real_frame_to_sim
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.image_rectify import (
    build_undistort_map,
    rectified_camera_matrix,
    transform_frame,
)
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    add_paper_target_marker,
    place_paper_target_marker,
)
from pick_and_place.workspace_alignment import (
    IDENTITY_ALIGNMENT,
    WorkspaceAlignment,
    fit_alignment,
    pixel_to_table_point,
)
from pick_and_place.workspace_overlays import is_cube_drop_allowed

CAMERA_TO_FEATURE = {
    "wrist_camera": "observation.images.wrist",
    "overhead_camera": "observation.images.overhead",
}
CAMERA_SHORT_NAMES = {"wrist_camera": "wrist", "overhead_camera": "overhead"}

# A resting phase of the cube track: detections whose positions agree within
# this radius belong to the same phase, and a phase needs this many detections
# before its pose is trusted over per-frame estimates.
STABLE_POSITION_TOLERANCE = 0.015
MIN_STABLE_DETECTIONS = 5
YAW_OUTLIER_DEG = 20.0


@dataclass(frozen=True)
class EpisodeRecord:
    index: int
    length: int
    fps: float
    states: np.ndarray
    video_paths: dict[str, Path]
    video_start_frames: dict[str, int]
    source_xy: tuple[float, float]
    source_yaw: float
    target_xy: tuple[float, float] | None


@dataclass(frozen=True)
class CubeDetection:
    frame: int
    x: float
    y: float
    yaw: float
    faces: int


def _finite(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _read_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def _read_episode_rows(dataset_root: Path) -> list[dict]:
    rows: list[dict] = []
    for parquet_path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(parquet_path).to_pylist())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def _read_states(dataset_root: Path, info: dict, row: dict) -> np.ndarray:
    data_path = dataset_root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    table = pq.read_table(data_path, columns=["episode_index", "observation.state"])
    table = table.filter(pc.equal(table["episode_index"], int(row["episode_index"])))
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)


def _load_episode(dataset_root: Path, info: dict, row: dict) -> EpisodeRecord:
    fps = float(info.get("fps", 30.0))
    index = int(row["episode_index"])

    source_x = row.get("cube_start_x")
    source_y = row.get("cube_start_y")
    if not (_finite(source_x) and _finite(source_y)):
        raise ValueError(f"episode {index} has no cube_start_x/y metadata")
    source_yaw = float(row["cube_start_yaw"]) if _finite(row.get("cube_start_yaw")) else 0.0

    target_x = row.get("target_x")
    target_y = row.get("target_y")
    target_xy = (
        (float(target_x), float(target_y)) if _finite(target_x) and _finite(target_y) else None
    )

    video_paths = {}
    video_start_frames = {}
    for camera, feature in CAMERA_TO_FEATURE.items():
        prefix = f"videos/{feature}"
        video_paths[camera] = (
            dataset_root
            / "videos"
            / feature
            / f"chunk-{int(row[f'{prefix}/chunk_index']):03d}"
            / f"file-{int(row[f'{prefix}/file_index']):03d}.mp4"
        )
        video_start_frames[camera] = round(float(row[f"{prefix}/from_timestamp"]) * fps)

    return EpisodeRecord(
        index=index,
        length=int(row["length"]),
        fps=fps,
        states=_read_states(dataset_root, info, row),
        video_paths=video_paths,
        video_start_frames=video_start_frames,
        source_xy=(float(source_x), float(source_y)),
        source_yaw=source_yaw,
        target_xy=target_xy,
    )


def _real_to_sim_vector(real_joints: np.ndarray) -> np.ndarray:
    arm_rad, gripper_rad = real_frame_to_sim(real_joints)
    return np.asarray([arm_rad[name] for name in ARM_JOINT_NAMES] + [gripper_rad], dtype=float)


def _output_fovy_deg(intrinsics: dict, out_w: int, out_h: int) -> float:
    """Vertical FOV of the rectified/cropped output, from its pinhole matrix."""
    f = rectified_camera_matrix(intrinsics, out_w, out_h)[1][1]
    return math.degrees(2.0 * math.atan(out_h / (2.0 * f)))


def _build_model(
    episode: EpisodeRecord,
    intrinsics: dict[str, dict],
    out_w: int,
    out_h: int,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_scene(include_environment=True)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, out_w)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, out_h)
    add_paper_target_marker(spec)
    apply_camera_extrinsics_to_spec(spec, load_local_camera_extrinsics())
    for camera in spec.cameras:
        if camera.name in intrinsics:
            camera.fovy = _output_fovy_deg(intrinsics[camera.name], out_w, out_h)

    cube = spec.body("pick_cube")
    cube.pos = (episode.source_xy[0], episode.source_xy[1], CUBE_HALF_SIZE)
    half_yaw = episode.source_yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()

    model = spec.compile()
    data = mujoco.MjData(model)
    if episode.target_xy is not None:
        place_paper_target_marker(
            model,
            episode.target_xy,
            0.0,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*episode.target_xy),
            alpha=1.0,
        )
    return model, data


def _set_cube(data: mujoco.MjData, qadr: int, dadr: int, x: float, y: float, yaw: float) -> None:
    half_yaw = yaw / 2.0
    data.qpos[qadr : qadr + 7] = (
        x,
        y,
        CUBE_HALF_SIZE,
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    data.qvel[dadr : dadr + 6] = 0.0


class _EpisodeVideoReader:
    """Sequential frame reader for one camera's slice of a chunked video file."""

    def __init__(self, path: Path, start_frame: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self._cap = cv2.VideoCapture(str(path))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def read(self) -> np.ndarray | None:
        """The next frame, or None once the video is exhausted."""
        ok, bgr = self._cap.read()
        return bgr if ok else None

    def close(self) -> None:
        self._cap.release()


def _native_camera_matrix(intrinsics: dict, native_w: int, native_h: int) -> np.ndarray:
    """Rectified pinhole matrix of the full-resolution undistorted frame."""
    matrix = np.array(intrinsics["camera_matrix"], dtype=float)
    matrix[0, :] *= native_w / float(intrinsics["width"])
    matrix[1, :] *= native_h / float(intrinsics["height"])
    fy = float(matrix[1, 1])
    return np.array(
        [[fy, 0.0, native_w / 2.0], [0.0, fy, native_h / 2.0], [0.0, 0.0, 1.0]], dtype=float
    )


ALIGNMENT_SAMPLE_STRIDE = 10


def _detect_cube_track(
    episode: EpisodeRecord,
    frame_count: int,
    undistort_map: tuple[np.ndarray, np.ndarray],
    camera_matrix: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    authored_tags: dict[int, tuple[float, float]] | None,
    tag_plane_z: float,
) -> tuple[list[CubeDetection], WorkspaceAlignment | None]:
    """First pass over the overhead video: per-frame cube detections, plus a
    workspace alignment fit from the frame tags when ``authored_tags`` is given."""
    detector = make_cube_detector()
    reader = _EpisodeVideoReader(
        episode.video_paths["overhead_camera"],
        episode.video_start_frames["overhead_camera"],
    )
    detections: list[CubeDetection] = []
    tag_points: dict[int, list[tuple[float, float]]] = {}
    try:
        for frame_index in range(frame_count):
            bgr = reader.read()
            if bgr is None:
                break
            rectified = cv2.remap(bgr, undistort_map[0], undistort_map[1], cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
            if authored_tags is not None and frame_index % ALIGNMENT_SAMPLE_STRIDE == 0:
                for detection in detect_tags(rgb, detector):
                    if detection.tag_id in authored_tags:
                        tag_points.setdefault(detection.tag_id, []).append(
                            pixel_to_table_point(
                                np.asarray(detection.center),
                                camera_matrix,
                                cam_pos,
                                cam_rot,
                                tag_plane_z,
                            )
                        )
            estimate = estimate_cube_pose(rgb, detector, camera_matrix)
            if estimate is None:
                continue
            rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
            yaw = float(Rotation.from_matrix(rotation).as_euler("xyz")[2])
            detections.append(
                CubeDetection(
                    frame=frame_index,
                    x=float(position[0]),
                    y=float(position[1]),
                    yaw=yaw,
                    faces=estimate.num_faces_used,
                )
            )
    finally:
        reader.close()

    alignment = None
    if authored_tags is not None:
        detected_tags = {
            tag_id: tuple(np.median(np.asarray(points), axis=0))
            for tag_id, points in tag_points.items()
        }
        alignment = fit_alignment(authored_tags, detected_tags)
    return detections, alignment


def _apply_alignment(
    detections: list[CubeDetection], alignment: WorkspaceAlignment
) -> list[CubeDetection]:
    corrected = []
    for d in detections:
        x, y = alignment.correct_point(d.x, d.y)
        corrected.append(
            CubeDetection(frame=d.frame, x=x, y=y, yaw=alignment.correct_yaw(d.yaw),
                          faces=d.faces)
        )
    return corrected


def _angular_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _robust_yaw(yaws: list[float]) -> float:
    """Circular median, then circular mean of the detections near it."""
    median = min(yaws, key=lambda a: sum(abs(_angular_difference(a, b)) for b in yaws))
    kept = [a for a in yaws if abs(_angular_difference(a, median)) <= math.radians(YAW_OUTLIER_DEG)]
    sin_sum = sum(math.sin(a) for a in kept)
    cos_sum = sum(math.cos(a) for a in kept)
    return math.atan2(sin_sum, cos_sum)


def _segment_track(detections: list[CubeDetection]) -> list[list[CubeDetection]]:
    """Group detections into runs whose positions agree with the run's median.

    Gaps in time do not break a run — only the cube actually moving does — so
    an occlusion phase over a resting cube stays one phase.
    """
    segments: list[list[CubeDetection]] = []
    for detection in detections:
        if segments:
            current = segments[-1]
            median_x = float(np.median([d.x for d in current]))
            median_y = float(np.median([d.y for d in current]))
            if math.hypot(detection.x - median_x, detection.y - median_y) \
                    <= STABLE_POSITION_TOLERANCE:
                current.append(detection)
                continue
        segments.append([detection])
    return segments


def _smooth_cube_track(
    detections: list[CubeDetection],
    frame_count: int,
    start_pose: tuple[float, float, float],
) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Per-frame cube pose and tracking label (``stable``/``detected``/``hold``)."""
    poses: list[tuple[float, float, float] | None] = [None] * frame_count
    labels: list[str | None] = [None] * frame_count

    for segment in _segment_track(detections):
        if len(segment) < MIN_STABLE_DETECTIONS:
            continue
        x = float(np.median([d.x for d in segment]))
        y = float(np.median([d.y for d in segment]))
        multi_face = [d.yaw for d in segment if d.faces >= 2]
        yaw = _robust_yaw(multi_face if len(multi_face) >= 3 else [d.yaw for d in segment])
        for frame in range(segment[0].frame, segment[-1].frame + 1):
            poses[frame] = (x, y, yaw)
            labels[frame] = "stable"

    for detection in detections:
        if labels[detection.frame] is None:
            poses[detection.frame] = (detection.x, detection.y, detection.yaw)
            labels[detection.frame] = "detected"

    last = start_pose
    for frame in range(frame_count):
        if poses[frame] is None:
            poses[frame] = last
            labels[frame] = "hold"
        else:
            last = poses[frame]
    return poses, labels  # type: ignore[return-value]


def _export_episode(
    dataset_root: Path,
    episode: EpisodeRecord,
    out_dir: Path,
    *,
    out_w: int,
    out_h: int,
    image_ext: str,
    montage: bool,
    max_frames: int | None,
    stable_only: bool,
    apply_alignment: bool,
    joint_offsets_deg: dict[str, float],
) -> None:
    intrinsics = load_local_camera_intrinsics()
    model, data = _build_model(episode, intrinsics, out_w, out_h)
    renderer = mujoco.Renderer(model, width=out_w, height=out_h)
    qpos_addrs = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    }
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    cube_joint = int(model.body_jntadr[cube_body])
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    cube_dadr = int(model.body_dofadr[cube_body])
    mujoco.mj_forward(model, data)

    frame_count = episode.length if max_frames is None else min(episode.length, max_frames)

    undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera in CAMERA_TO_FEATURE:
        probe = cv2.VideoCapture(str(episode.video_paths[camera]))
        native_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()
        undistort_maps[camera] = build_undistort_map(intrinsics[camera], native_w, native_h, cv2)

    overhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    probe = cv2.VideoCapture(str(episode.video_paths["overhead_camera"]))
    overhead_native = (
        int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    probe.release()
    frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workspace_frame_frame")
    frame_pos = data.xpos[frame_id]
    frame_mat = data.xmat[frame_id].reshape(3, 3)
    authored_tags = {}
    plate_z = []
    for tag_id, _corner, pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        world = frame_pos + frame_mat @ np.asarray(pos)
        authored_tags[tag_id] = (float(world[0]), float(world[1]))
        plate_z.append(float(world[2]) + 0.0025)
    tag_plane_z = float(np.mean(plate_z))

    detections, alignment = _detect_cube_track(
        episode,
        frame_count,
        undistort_maps["overhead_camera"],
        _native_camera_matrix(intrinsics["overhead_camera"], *overhead_native),
        data.cam_xpos[overhead_id].copy(),
        data.cam_xmat[overhead_id].reshape(3, 3).copy(),
        authored_tags,
        tag_plane_z,
    )
    if alignment is None:
        print(f"warning: episode {episode.index}: too few frame tags for alignment fit")
    effective = (alignment or IDENTITY_ALIGNMENT) if apply_alignment else IDENTITY_ALIGNMENT
    detections = _apply_alignment(detections, effective)
    start_xy = effective.correct_point(*episode.source_xy)
    cube_poses, cube_labels = _smooth_cube_track(
        detections,
        frame_count,
        (start_xy[0], start_xy[1], effective.correct_yaw(episode.source_yaw)),
    )

    readers: dict[str, _EpisodeVideoReader] = {}
    writers: dict[str, cv2.VideoWriter] = {}
    for camera in CAMERA_TO_FEATURE:
        readers[camera] = _EpisodeVideoReader(
            episode.video_paths[camera], episode.video_start_frames[camera]
        )
        short = CAMERA_SHORT_NAMES[camera]
        (out_dir / f"{short}_real").mkdir(parents=True, exist_ok=True)
        (out_dir / f"{short}_sim").mkdir(parents=True, exist_ok=True)
        if montage:
            writers[camera] = cv2.VideoWriter(
                str(out_dir / f"montage_{short}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                episode.fps,
                (out_w * 3, out_h),
            )

    frames_meta: list[dict] = []
    try:
        for frame_index in range(frame_count):
            reals = {camera: readers[camera].read() for camera in CAMERA_TO_FEATURE}
            if any(frame is None for frame in reals.values()):
                print(
                    f"warning: episode {episode.index} video ends at frame {frame_index}, "
                    f"expected {frame_count}"
                )
                break
            cube_x, cube_y, cube_yaw = cube_poses[frame_index]
            if stable_only and cube_labels[frame_index] != "stable":
                frames_meta.append(
                    {
                        "frame": frame_index,
                        "cube_tracking": cube_labels[frame_index],
                        "cube": {"x": cube_x, "y": cube_y, "yaw": cube_yaw},
                        "exported": False,
                    }
                )
                continue
            _set_cube(data, cube_qadr, cube_dadr, cube_x, cube_y, cube_yaw)
            sim_joints = _real_to_sim_vector(episode.states[frame_index])
            for joint_name, offset_deg in joint_offsets_deg.items():
                sim_joints[JOINT_NAMES.index(joint_name)] += math.radians(offset_deg)
            for i, name in enumerate(JOINT_NAMES):
                data.qpos[qpos_addrs[name]] = sim_joints[i]
            mujoco.mj_forward(model, data)

            frame_meta: dict = {
                "frame": frame_index,
                "cube_tracking": cube_labels[frame_index],
                "cube": {"x": cube_x, "y": cube_y, "yaw": cube_yaw},
                "wrist_cam": {
                    "pos": [float(v) for v in data.cam_xpos[wrist_id]],
                    "mat": [float(v) for v in data.cam_xmat[wrist_id]],
                },
            }
            for camera in CAMERA_TO_FEATURE:
                short = CAMERA_SHORT_NAMES[camera]
                real_out = transform_frame(
                    reals[camera], undistort_maps[camera], out_w, out_h, cv2
                )
                renderer.update_scene(data, camera=camera)
                sim_out = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                gray_real = cv2.cvtColor(real_out, cv2.COLOR_BGR2GRAY)
                gray_sim = cv2.cvtColor(sim_out, cv2.COLOR_BGR2GRAY)
                frame_meta[f"{short}_gray_mae"] = float(
                    np.mean(np.abs(gray_real.astype(np.float32) - gray_sim.astype(np.float32)))
                )
                cv2.imwrite(str(out_dir / f"{short}_real" / f"{frame_index:06d}{image_ext}"),
                            real_out)
                cv2.imwrite(str(out_dir / f"{short}_sim" / f"{frame_index:06d}{image_ext}"),
                            sim_out)
                if montage:
                    blend = cv2.addWeighted(real_out, 0.55, sim_out, 0.45, 0.0)
                    writers[camera].write(np.concatenate([real_out, sim_out, blend], axis=1))
            frames_meta.append(frame_meta)
    finally:
        renderer.close()
        for reader in readers.values():
            reader.close()
        for writer in writers.values():
            writer.release()

    exported = len(frames_meta)
    stable = sum(1 for meta in frames_meta if meta["cube_tracking"] == "stable")
    held = sum(1 for meta in frames_meta if meta["cube_tracking"] == "hold")
    index = {
        "dataset": str(dataset_root),
        "episode_index": episode.index,
        "fps": episode.fps,
        "width": out_w,
        "height": out_h,
        "joint_offsets_deg": joint_offsets_deg,
        "workspace_alignment": None
        if alignment is None
        else {
            "yaw_deg": math.degrees(alignment.yaw),
            "tx_mm": alignment.tx * 1000.0,
            "ty_mm": alignment.ty * 1000.0,
            "num_tags": alignment.num_tags,
            "residual_mm": alignment.residual_mm,
        },
        "frames": frames_meta,
    }
    with (out_dir / "pairs.json").open("w") as f:
        json.dump(index, f, indent=1)
    print(
        f"{dataset_root.name} episode {episode.index}: {exported} frame(s), "
        f"cube stable in {stable} ({100.0 * stable / max(exported, 1):.0f}%), held in {held}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_roots", type=Path, nargs="+", help="LeRobotDataset root(s)")
    parser.add_argument("--episode", type=int, default=None, help="export only this episode index")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/pairs"))
    parser.add_argument("--width", type=int, default=960, help="output width (default: 960)")
    parser.add_argument("--height", type=int, default=720, help="output height (default: 720)")
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="pair image format (default: jpg)",
    )
    parser.add_argument(
        "--montage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write real | sim | blend montage videos per episode",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="debug frame limit")
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="write images only for frames whose cube pose is confidently tracked",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="export at most this many episodes per dataset (per shard when sharded)",
    )
    parser.add_argument(
        "--apply-workspace-alignment",
        action="store_true",
        help=(
            "correct cube poses by the fitted workspace-frame alignment; the fit is "
            "always recorded in pairs.json, but wrist-camera evidence says the "
            "discrepancy is the physical frame's placement, not the camera, so the "
            "correction is off by default"
        ),
    )
    parser.add_argument(
        "--joint-offsets-deg",
        type=str,
        default="",
        metavar="NAME=DEG[,NAME=DEG...]",
        help=(
            "add these zero-offset corrections to the sim joints when posing, e.g. "
            "'shoulder_pan=3.75,elbow_flex=3.3' (fitted by fit_joint_zeros)"
        ),
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="I/N",
        help="process only episodes with episode_index %% N == I, for parallel runs",
    )
    args = parser.parse_args()

    joint_offsets_deg: dict[str, float] = {}
    if args.joint_offsets_deg:
        for item in args.joint_offsets_deg.split(","):
            name, _, value = item.partition("=")
            name = name.strip()
            if name not in JOINT_NAMES:
                parser.error(f"unknown joint {name!r} in --joint-offsets-deg")
            joint_offsets_deg[name] = float(value)

    shard_index, shard_count = 0, 1
    if args.shard is not None:
        try:
            shard_index, shard_count = (int(v) for v in args.shard.split("/"))
        except ValueError:
            parser.error("--shard must look like I/N, e.g. 0/3")
        if not 0 <= shard_index < shard_count:
            parser.error("--shard index must be in [0, N)")

    for dataset_root in args.dataset_roots:
        info = _read_info(dataset_root)
        exported_episodes = 0
        for row in _read_episode_rows(dataset_root):
            index = int(row["episode_index"])
            if args.episode is not None and index != args.episode:
                continue
            if index % shard_count != shard_index:
                continue
            if args.max_episodes is not None and exported_episodes >= args.max_episodes:
                break
            out_dir = args.output_root / dataset_root.name / f"episode_{index:06d}"
            if (out_dir / "pairs.json").is_file():
                print(f"{dataset_root.name} episode {index}: already exported, skipping")
                exported_episodes += 1
                continue
            exported_episodes += 1
            try:
                episode = _load_episode(dataset_root, info, row)
                _export_episode(
                    dataset_root,
                    episode,
                    out_dir,
                    out_w=args.width,
                    out_h=args.height,
                    image_ext=f".{args.image_format}",
                    montage=args.montage,
                    max_frames=args.max_frames,
                    stable_only=args.stable_only,
                    apply_alignment=args.apply_workspace_alignment,
                    joint_offsets_deg=joint_offsets_deg,
                )
            except Exception as exc:
                print(f"error: {dataset_root.name} episode {index}: {exc}")


if __name__ == "__main__":
    main()
