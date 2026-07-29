# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Replay a recorded ground-truth episode's scene and re-encode its videos.

A recorded simulation episode stores everything the renderer needs: the
hardware-frame joints per frame (``observation.state``), the true cube pose per
frame (``observation.environment_state``) and the drop-zone target pose and yaw
in episode metadata. Replaying those through the recording scene reproduces the
recorded frames, which makes the scene's *appearance* a free variable after the
fact — the property the DPPO retrain needs, because the pick cube has to be
rendered blue to be legible at 96x96 but has to be recorded with its AprilTag
faces for the descent visual servo to work.

Everything here is about reproducing the recording pipeline rather than merely
resembling it: the scene comes from
:func:`pick_and_place.sim_recorder.build_recording_scene` (the function the
recorder itself builds through), the images go through the same
:class:`~pick_and_place.sim_recorder.SimCameraRig` at the recorded resolution,
and the H.264 settings are read back out of the episode's own video rather than
assumed. What cannot be reproduced is flagged instead of papered over: see
:func:`assert_rerenderable`.

The states, actions and all other recorded columns are copied through
untouched; only pixels change.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from pick_and_place.camera_extrinsics import load_local_camera_extrinsics
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.camera_pose_envelope import apply_camera_jitter, camera_module_geoms
from pick_and_place.domain_randomization import ProceduralAppearance, write_procedural_textures
from pick_and_place.episodes import set_joint
from pick_and_place.follower import ARM_JOINT_NAMES, real_frame_to_sim
from pick_and_place.paper_detection import DROP_ZONE_HALF_SIZE, place_paper_target_marker
from pick_and_place.scene_appearance import SceneAppearance, SceneAppearanceOverride
from pick_and_place.sim_recorder import (
    OVERHEAD_CAMERA,
    WRIST_CAMERA,
    SimCameraRig,
    build_recording_scene,
)
from pick_and_place.workspace_overlays import is_cube_drop_allowed

#: Dataset feature name of each camera, and the MuJoCo camera it renders from.
CAMERA_FEATURES: dict[str, str] = {
    "observation.images.wrist": WRIST_CAMERA,
    "observation.images.overhead": OVERHEAD_CAMERA,
}

#: H.264 settings assumed when an episode's video carries no x264 options
#: string. The recorded datasets are written by LeRobot's streaming encoder at
#: these values; they are read back per episode rather than trusted blindly.
DEFAULT_X264_CRF = 30.0
DEFAULT_X264_KEYINT = 2

#: LeRobot samples every 6th pixel in each direction for image statistics,
#: which is what makes its recorded ``count`` exactly ``frames * 120 * 160``
#: for a 720x960 video. Matching the stride keeps recomputed statistics
#: comparable with the recorded ones.
STATS_PIXEL_STRIDE = 6


@dataclass(frozen=True)
class X264Settings:
    """The rate control an episode's videos were encoded with."""

    crf: float
    keyint: int

    def encoder_options(self) -> dict[str, str]:
        return {"crf": f"{self.crf:g}"}


@dataclass(frozen=True)
class FrameDiff:
    """Absolute per-pixel difference between two uint8 RGB images."""

    mean: float
    p99: float
    max: float

    @staticmethod
    def between(first: np.ndarray, second: np.ndarray) -> FrameDiff:
        if first.shape != second.shape:
            raise ValueError(f"cannot diff {first.shape} against {second.shape}")
        delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
        return FrameDiff(
            mean=float(delta.mean()),
            p99=float(np.percentile(delta, 99.0)),
            max=float(delta.max()),
        )


def episode_video_paths(episode_root: Path) -> dict[str, Path]:
    """Return each camera feature's single video file in a staged episode."""
    paths: dict[str, Path] = {}
    for feature in CAMERA_FEATURES:
        found = sorted((episode_root / "videos" / feature).glob("chunk-*/file-*.mp4"))
        if len(found) != 1:
            raise ValueError(
                f"{episode_root.name} must hold exactly one {feature} video, found {len(found)}"
            )
        paths[feature] = found[0]
    return paths


def episode_video_fps(episode_root: Path) -> int:
    """Frames per second the episode's videos were written at."""
    with (episode_root / "meta" / "info.json").open() as file:
        info = json.load(file)
    return int(info["fps"])


def x264_settings(video_path: Path, *, head_bytes: int = 512_000) -> X264Settings:
    """Read the rate control out of a video's embedded x264 options string.

    libx264 writes its full option list into the stream as user data, so an
    episode's own file states the CRF and keyframe interval it was produced
    with. Reading them back is what keeps a re-encode comparable to the
    recording even if the encoder's defaults change.
    """
    head = video_path.open("rb").read(head_bytes)
    crf = re.search(rb"[ -]crf=([0-9.]+)", head)
    keyint = re.search(rb"[ -]keyint=(\d+)", head)
    return X264Settings(
        crf=float(crf.group(1)) if crf else DEFAULT_X264_CRF,
        keyint=int(keyint.group(1)) if keyint else DEFAULT_X264_KEYINT,
    )


def decode_video(path: Path) -> Iterator[np.ndarray]:
    """Yield a video's frames as ``(H, W, 3)`` uint8 RGB arrays."""
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def video_frame_count(path: Path) -> int:
    """Frames the container reports, without decoding them."""
    import av

    with av.open(str(path)) as container:
        return container.streams.video[0].frames


class VideoWriter:
    """Write uint8 RGB frames as an H.264 MP4 matching a recorded episode's."""

    def __init__(
        self,
        target: Path | io.BytesIO,
        *,
        width: int,
        height: int,
        fps: int,
        settings: X264Settings,
    ) -> None:
        import av

        self._container = av.open(
            target if isinstance(target, io.BytesIO) else str(target),
            mode="w",
            format="mp4" if isinstance(target, io.BytesIO) else None,
        )
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.codec_context.time_base = Fraction(1, fps)
        self._stream.codec_context.gop_size = settings.keyint
        self._stream.codec_context.options = settings.encoder_options()
        self._frames = 0

    def write(self, rgb: np.ndarray) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = self._frames
        frame.time_base = self._stream.codec_context.time_base
        self._container.mux(self._stream.encode(frame))
        self._frames += 1

    @property
    def frames_written(self) -> int:
        return self._frames

    def close(self) -> None:
        self._container.mux(self._stream.encode(None))
        self._container.close()

    def __enter__(self) -> VideoWriter:  # noqa: PYI034 - a context manager, not a factory
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def encode_decode(
    frames: Sequence[np.ndarray], *, fps: int, settings: X264Settings
) -> list[np.ndarray]:
    """Round-trip frames through the recorded codec settings, in memory.

    The result minus the input is the codec's own noise floor: the part of a
    re-render's difference from a recorded video that no amount of rendering
    fidelity can remove, because the recording is itself lossy.
    """
    if not frames:
        return []
    buffer = io.BytesIO()
    height, width = frames[0].shape[:2]
    with VideoWriter(buffer, width=width, height=height, fps=fps, settings=settings) as writer:
        for frame in frames:
            writer.write(frame)
    buffer.seek(0)
    import av

    decoded: list[np.ndarray] = []
    with av.open(buffer) as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_ndarray(format="rgb24"))
    return decoded


@dataclass(frozen=True)
class CameraJitter:
    """One overhead-camera pose+focal draw, as a rigid transform from the authored pose.

    Camera pose and focal length are purely a rendering matter -- nothing reads
    the overhead image back into control during recording -- so a re-render can
    draw its own jitter independently of how (or whether) the source episode was
    recorded with one. See :func:`pick_and_place.camera_pose_envelope.draw_overhead_camera_jitter`.
    """

    position_m: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    focal_scale: float = 1.0


class EpisodeRenderer:
    """Re-render recorded ground truth through the recording camera pipeline."""

    def __init__(
        self,
        *,
        render_hw: tuple[int, int],
        image_hw: tuple[int, int],
        background_panorama: Path | str | np.ndarray | None = None,
        table_texture: Path | str | np.ndarray | None = None,
    ) -> None:
        render_height, render_width = render_hw
        image_height, image_width = image_hw
        self.model, self.data = build_recording_scene(
            render_width=render_width,
            render_height=render_height,
            background_panorama=background_panorama,
            table_texture=table_texture,
        )
        self.rig = SimCameraRig(
            self.model,
            load_local_camera_intrinsics(),
            width=image_width,
            height=image_height,
            render_width=render_width,
            render_height=render_height,
        )
        self.appearance = SceneAppearanceOverride(self.model)
        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[cube_body]])

        # Authored overhead camera pose/focal and its hardware geoms' base poses,
        # snapshotted once so a jittered episode can be restored to it exactly.
        self._overhead_camera = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, OVERHEAD_CAMERA
        )
        self._overhead_cam_pos = self.model.cam_pos[self._overhead_camera].copy()
        self._overhead_cam_quat = self.model.cam_quat[self._overhead_camera].copy()
        self._overhead_cam_fovy = float(self.model.cam_fovy[self._overhead_camera])
        self._overhead_geom_base = {
            geom: (self.model.geom_pos[geom].copy(), self.model.geom_quat[geom].copy())
            for geom in camera_module_geoms(self.model, self._overhead_camera)
        }

        # Present only when built with the finite-floor scene (background_panorama
        # or table_texture given); empty otherwise, making set_scene_texture a no-op.
        self._scene_texture_ids = tuple(
            ident
            for name in ("table_texture", "background_panorama")
            if (ident := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, name)) >= 0
        )
        self._scene_tex_data_base = self.model.tex_data.copy()

    def set_camera_jitter(self, jitter: CameraJitter | None) -> None:
        """Apply an overhead-camera pose+focal jitter, or restore the authored pose."""
        if jitter is None:
            self.model.cam_pos[self._overhead_camera] = self._overhead_cam_pos
            self.model.cam_quat[self._overhead_camera] = self._overhead_cam_quat
            self.model.cam_fovy[self._overhead_camera] = self._overhead_cam_fovy
            for geom, (geom_pos, geom_quat) in self._overhead_geom_base.items():
                self.model.geom_pos[geom] = geom_pos
                self.model.geom_quat[geom] = geom_quat
            return
        apply_camera_jitter(
            self.model,
            self._overhead_camera,
            self._overhead_cam_pos,
            self._overhead_cam_quat,
            self._overhead_geom_base,
            np.asarray(jitter.position_m, float),
            np.asarray(jitter.rotation_deg, float),
        )
        half = math.radians(self._overhead_cam_fovy) / 2.0
        self.model.cam_fovy[self._overhead_camera] = math.degrees(
            2.0 * math.atan(math.tan(half) / jitter.focal_scale)
        )

    def set_scene_texture(self, appearance: ProceduralAppearance | None) -> None:
        """Apply a background/table appearance, or restore the one built at construction.

        A no-op unless the renderer was built with ``background_panorama`` or
        ``table_texture`` (the finite-floor scene) -- there is nothing to vary
        under the infinite groundplane every plain recording has used so far.
        """
        if not self._scene_texture_ids:
            return
        if appearance is None:
            self.model.tex_data[:] = self._scene_tex_data_base
        else:
            write_procedural_textures(self.model, self._scene_texture_ids, appearance)
        self.rig.reload_textures(self._scene_texture_ids)

    def set_episode(
        self,
        target_xy: tuple[float, float],
        target_plate_yaw: float,
        *,
        camera_jitter: CameraJitter | None = None,
        scene_texture: ProceduralAppearance | None = None,
    ) -> None:
        """Place the drop-zone marker, overhead camera and scene texture for this episode."""
        place_paper_target_marker(
            self.model,
            target_xy,
            target_plate_yaw,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*target_xy),
            alpha=1.0,
        )
        self.set_camera_jitter(camera_jitter)
        self.set_scene_texture(scene_texture)
        mujoco.mj_forward(self.model, self.data)
        # The placement decides this episode's plate colour, which is what an
        # appearance leaving the target unset must restore.
        self.appearance.refresh_plate_baseline()

    def set_frame(self, state_real: np.ndarray, cube_pose: np.ndarray) -> None:
        """Pose the arm and cube at one recorded frame's ground truth."""
        arm_rad, gripper_rad = real_frame_to_sim(np.asarray(state_real, dtype=np.float64))
        for name in ARM_JOINT_NAMES:
            set_joint(self.model, self.data, name, arm_rad[name])
        set_joint(self.model, self.data, "gripper", gripper_rad)
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_pose
        mujoco.mj_forward(self.model, self.data)

    def capture(self, appearance: SceneAppearance) -> dict[str, np.ndarray]:
        """Render both cameras under ``appearance``, keyed by dataset feature."""
        self.appearance.apply(appearance)
        wrist, overhead = self.rig.capture(self.data)
        return {
            "observation.images.wrist": wrist,
            "observation.images.overhead": overhead,
        }

    def close(self) -> None:
        self.rig.close()


class ImageStatsAccumulator:
    """Accumulate LeRobot-shaped image statistics over re-rendered frames.

    Only every ``STATS_PIXEL_STRIDE``-th pixel is kept, both to match the
    recorded statistics' sample size and to keep a full episode's pixels in
    memory at a few tens of megabytes.
    """

    def __init__(self) -> None:
        self._samples: list[np.ndarray] = []

    def add(self, rgb: np.ndarray) -> None:
        stride = STATS_PIXEL_STRIDE
        self._samples.append(rgb[::stride, ::stride].reshape(-1, 3).copy())

    def result(self) -> dict[str, object]:
        if not self._samples:
            raise ValueError("no frames accumulated")
        sample = np.concatenate(self._samples).astype(np.float64) / 255.0

        def per_channel(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(value)]] for value in values]

        stats: dict[str, object] = {
            "min": per_channel(sample.min(axis=0)),
            "max": per_channel(sample.max(axis=0)),
            "mean": per_channel(sample.mean(axis=0)),
            "std": per_channel(sample.std(axis=0)),
            "count": [int(sample.shape[0])],
        }
        for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
            stats[f"q{int(quantile * 100):02d}"] = per_channel(
                np.quantile(sample, quantile, axis=0)
            )
        return stats


def rewrite_image_stats(stats_path: Path, stats_by_feature: dict[str, dict[str, object]]) -> None:
    """Replace the camera features' statistics; leave every other one alone.

    The recorded statistics describe the recorded pixels, so a re-render with a
    different appearance — a dark floor above all — invalidates exactly the
    image entries and nothing else.
    """
    with stats_path.open() as file:
        stats = json.load(file)
    for feature, feature_stats in stats_by_feature.items():
        if feature in stats:
            stats[feature] = feature_stats
    with stats_path.open("w") as file:
        json.dump(stats, file, indent=4)
        file.write("\n")


def assert_rerenderable(episode_root: Path) -> None:
    """Raise unless the episode's pixels are recoverable from what it stores.

    Two recorded conditions are not:

    * ``--miscalibration`` — ``observation.state`` is the servo-style readback,
      the *believed* joints, which differ from the joints physics and rendering
      used by the injected per-episode joint-zero offsets. The constant part of
      the draw is in episode metadata, but the within-session pan jitter is a
      random walk whose realization is not stored, so the true arm pose cannot
      be rebuilt frame by frame.
    * ``--domain-randomization`` — the appearance is drawn per episode. The draw
      *is* stored (``domain_sample_json``), so this is a missing feature rather
      than missing information: applying it needs the randomizer, its texture
      reloads, its image postprocess and its believed wrist-camera pose, none of
      which this replay does yet.
    """
    metadata_paths = sorted((episode_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if len(metadata_paths) != 1:
        raise ValueError(f"{episode_root.name} must contain one episode metadata parquet")
    columns = set(pq.read_schema(metadata_paths[0]).names)
    if any(column.startswith("injected_offset_") for column in columns):
        raise ValueError(
            f"{episode_root.name} was recorded with --miscalibration: observation.state holds "
            "the believed joints, not the true ones, and the pan jitter that separates them is "
            "not stored per frame, so its pixels cannot be reproduced. Re-rendering needs the "
            "recorder to store the true joints (or the per-frame offsets) alongside the cube pose."
        )
    if "domain_sample_json" in columns:
        raise ValueError(
            f"{episode_root.name} was recorded with --domain-randomization, whose per-episode "
            "appearance draw this replay does not apply (the draw is stored in "
            "domain_sample_json, so support is possible; it is simply not implemented)."
        )


def render_environment_fingerprint(
    *, render_hw: tuple[int, int], image_hw: tuple[int, int]
) -> dict[str, object]:
    """Identify everything outside the dataset that moves a rendered pixel.

    A re-render is only as trustworthy as the environment its verification ran
    in. The camera calibrations are machine-local (gitignored) files, and the
    OpenGL backend and MuJoCo version decide the shading, so a verification pass
    is only evidence for re-renders produced under the same fingerprint.
    """

    def digest(payload: object) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    extrinsics = load_local_camera_extrinsics()
    intrinsics = load_local_camera_intrinsics()
    return {
        "mujoco_version": mujoco.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "platform": f"{platform.system()}-{platform.machine()}",
        "camera_extrinsics": sorted(extrinsics),
        "camera_extrinsics_digest": digest(extrinsics),
        "camera_intrinsics": sorted(intrinsics),
        "camera_intrinsics_digest": digest(intrinsics),
        "render_hw": list(render_hw),
        "image_hw": list(image_hw),
    }


def copy_episode_scaffold(source: Path, destination: Path) -> None:
    """Copy an episode's parquet data and metadata; leave the videos to the caller."""
    destination.mkdir(parents=True)
    for name in ("data", "meta"):
        shutil.copytree(source / name, destination / name)
    for extra in source.iterdir():
        if extra.is_file():
            shutil.copy2(extra, destination / extra.name)


def directory_size(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def evenly_spaced(count: int, wanted: int) -> list[int]:
    """Indices of ``wanted`` evenly spaced items out of ``count`` (all if <= 0)."""
    if wanted <= 0 or wanted >= count:
        return list(range(count))
    return sorted({round(index) for index in np.linspace(0, count - 1, wanted)})


def mean_of(values: Iterable[float]) -> float:
    collected = list(values)
    return float(np.mean(collected)) if collected else float("nan")
