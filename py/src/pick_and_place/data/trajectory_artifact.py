# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One episode's behavior, recorded without any pixels.

This is the handoff between generating a trajectory and rendering it. Generating
one costs a physics run and cannot be redone cheaply; rendering one is a pure
function of the poses it stores, so a single artifact can be restyled as many
times as wanted. Keeping the two apart is only possible if the artifact holds
everything a renderer needs, which is what the schema below is for.

**Two frames, both required.** A servo commanded to 20 degrees with a drawn
+1.5 degree joint-zero offset physically rests at 21.5 and reports 20 back. The
20 is the training label — it is what a real servo reports and what a policy sees
at deployment — and the 21.5 is where the arm has to be put to reproduce the
picture. Storing only one of them loses the other for good: part of the offset is
a slow random walk that is not otherwise logged per frame. Both are stored here,
per frame, so that class of bug cannot recur.

The same split runs through the cube: the true pose is what physics had and what
the renderer needs, the believed pose is what the expert was steering by and is
kept because it is the only record of *why* the expert did what it did.

Nothing derived is stored. The final placement error is a function of the last
true cube pose and the target, so :meth:`TrajectoryArtifact.placement_error`
recomputes it rather than risk a stored copy drifting out of sync.

Size: about 37 floats per frame, so a 15-second episode at 30 Hz is around 65 KB
and a thousand episodes around 65 MB. The format is not the cost; the render pass
is. Replaying an artifact in frame order is cheap, which is what lets a variant
renderer restyle the scene once and then run the whole episode through it,
instead of paying a texture upload per variant per frame.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.core.camera_calibration import (
    load_local_camera_extrinsics,
    load_local_camera_intrinsics,
)
from pick_and_place.core.appearance import AppearanceDraw
from pick_and_place.core.geometry import CubePose, PlacementError
from pick_and_place.core.miscalibration import MiscalibrationDraw
from pick_and_place.core.task_phases import PhaseSpan, phase_spans_from_json, phase_spans_json
from pick_and_place.spec.robot import JOINT_NAMES
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

#: Bumped whenever the stored fields change meaning. A reader refuses anything
#: it was not written against rather than silently misinterpreting a column.
ARTIFACT_VERSION = 2

#: What the artifact is called inside a staged episode directory.
ARTIFACT_FILENAME = "trajectory.npz"

_JOINTS = len(JOINT_NAMES)
#: Position plus a MuJoCo ``w, x, y, z`` quaternion — the true cube pose, which
#: tumbles and so cannot be flattened to a yaw.
_QUAT_POSE = 7
#: Position plus intrinsic ZYX Euler angles — the shape a :class:`CubePose` has,
#: and the only shape the believed world is ever expressed in.
_EULER_POSE = 6

#: Column layout of every per-frame array, and the width each one must have.
FRAME_ARRAY_WIDTHS = {
    "true_state": _JOINTS,
    "believed_state": _JOINTS,
    "action": _JOINTS,
    "true_cube_pose": _QUAT_POSE,
    "believed_cube_pose": _EULER_POSE,
    "wrist_sighting": _EULER_POSE,
}

_METADATA_KEY = "episode"


def cube_pose_row(pose: CubePose) -> np.ndarray:
    """Flatten a :class:`CubePose` into its stored six columns."""
    return np.array(
        [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw], dtype=np.float32
    )


def cube_pose_from_row(row: np.ndarray) -> CubePose:
    """Rebuild a :class:`CubePose` from its stored six columns."""
    x, y, z, roll, pitch, yaw = (float(value) for value in row)
    return CubePose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)


@dataclasses.dataclass(frozen=True)
class TrajectoryFrames:
    """Everything that varies tick by tick, one row per control tick.

    ``true_state`` and ``believed_state`` are both real-frame six-vectors (arm
    joints in degrees, gripper 0-100), the same units as a hardware recording, so
    ``believed_state`` can be written straight out as ``observation.state`` and
    ``true_state`` can be fed straight back to a renderer. Without a
    miscalibration draw the two are equal and the loop degenerates to feedforward
    playback, which is exactly what should happen.

    ``wrist_sighting`` is the pose the descent servo solved out of the wrist
    camera that tick, or a row of NaN when nothing was seen. It drives nothing on
    replay; it is kept because a descent that misbehaves is otherwise
    unreconstructable after the fact.
    """

    true_state: np.ndarray
    believed_state: np.ndarray
    action: np.ndarray
    true_cube_pose: np.ndarray
    believed_cube_pose: np.ndarray
    wrist_sighting: np.ndarray

    def __len__(self) -> int:
        return int(self.true_state.shape[0])

    def __post_init__(self) -> None:
        for name, width in FRAME_ARRAY_WIDTHS.items():
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[1] != width:
                raise ValueError(f"{name} must be (n, {width}), got {array.shape}")
            if array.shape[0] != len(self):
                raise ValueError(
                    f"{name} holds {array.shape[0]} frames against "
                    f"{len(self)} in true_state"
                )

    @property
    def sighted(self) -> np.ndarray:
        """Per-frame mask of the ticks the wrist camera actually solved a pose on."""
        return ~np.isnan(self.wrist_sighting[:, 0])


@dataclasses.dataclass(frozen=True)
class MiscalibrationRecord:
    """The draw an episode ran under, as far as it is a fixed per-episode value.

    The pan jitter is deliberately absent: it is a random walk, so there is no
    per-episode number that describes it. Its realization is recoverable frame by
    frame as the difference between the true and believed states, which is the
    reason both are stored.
    """

    base_offsets_deg: dict[str, float]
    cube_belief_error: tuple[float, float, float, float]
    target_belief_error: tuple[float, float]

    @staticmethod
    def of(draw: MiscalibrationDraw | None) -> MiscalibrationRecord | None:
        if draw is None:
            return None
        return MiscalibrationRecord(
            base_offsets_deg={
                name: float(value) for name, value in draw.base_offsets_deg.items()
            },
            cube_belief_error=tuple(float(value) for value in draw.cube_belief_error),
            target_belief_error=tuple(float(value) for value in draw.target_belief_error),
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "base_offsets_deg": dict(self.base_offsets_deg),
            "cube_belief_error": list(self.cube_belief_error),
            "target_belief_error": list(self.target_belief_error),
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> MiscalibrationRecord:
        return MiscalibrationRecord(
            base_offsets_deg={
                str(name): float(value)
                for name, value in payload["base_offsets_deg"].items()
            },
            cube_belief_error=tuple(float(v) for v in payload["cube_belief_error"]),
            target_belief_error=tuple(float(v) for v in payload["target_belief_error"]),
        )


@dataclasses.dataclass(frozen=True)
class WristCameraMount:
    """Where the wrist camera physically sat, as an offset from its authored mount.

    Drawn when the episode was generated, because the expert had to servo through
    it — but recorded here because it also moves pixels: rebuild the picture with
    the camera on its nominal mount and every wrist frame is subtly wrong. It is
    the one L1 draw a renderer has to know about.
    """

    position_m: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]

    def as_json(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "rotation_deg": list(self.rotation_deg),
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> WristCameraMount:
        return WristCameraMount(
            position_m=tuple(float(v) for v in payload["position_m"]),
            rotation_deg=tuple(float(v) for v in payload["rotation_deg"]),
        )


@dataclasses.dataclass(frozen=True)
class EpisodeFacts:
    """What is true of the whole episode rather than of any one tick.

    ``target_xy`` and ``target_plate_yaw`` are here because the drop zone does not
    move: a renderer places the plate once and then runs every frame through it.
    ``seed``, ``miscalibration`` and ``wrist_camera_mount`` are what make the
    episode reproducible. ``recorded_appearance`` is the look its pixels were
    actually made with, kept so a re-render can reproduce the recording rather
    than only replace it — which is what lets a randomized recording be verified
    against its own video. A variant pass overrides it.

    ``verdict`` and the placement error derived from the last frame are what a
    consumer filters on.
    """

    target_xy: tuple[float, float]
    target_plate_yaw: float
    verdict: str
    phase_spans: tuple[PhaseSpan, ...]
    fingerprint: dict[str, Any]
    seed: int | None = None
    episode_index: int | None = None
    miscalibration: MiscalibrationRecord | None = None
    wrist_camera_mount: WristCameraMount | None = None
    recorded_appearance: AppearanceDraw | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "target_xy": [float(value) for value in self.target_xy],
            "target_plate_yaw": float(self.target_plate_yaw),
            "verdict": self.verdict,
            "phase_spans": phase_spans_json(self.phase_spans),
            "fingerprint": self.fingerprint,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "miscalibration": (
                None if self.miscalibration is None else self.miscalibration.as_json()
            ),
            "wrist_camera_mount": (
                None if self.wrist_camera_mount is None else self.wrist_camera_mount.as_json()
            ),
            "recorded_appearance": (
                None if self.recorded_appearance is None else self.recorded_appearance.as_json()
            ),
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> EpisodeFacts:
        version = int(payload["artifact_version"])
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"trajectory artifact version {version} cannot be read by this "
                f"build, which writes version {ARTIFACT_VERSION}"
            )
        target_x, target_y = payload["target_xy"]
        miscalibration = payload["miscalibration"]
        mount = payload["wrist_camera_mount"]
        appearance = payload["recorded_appearance"]
        return EpisodeFacts(
            target_xy=(float(target_x), float(target_y)),
            target_plate_yaw=float(payload["target_plate_yaw"]),
            verdict=str(payload["verdict"]),
            phase_spans=phase_spans_from_json(payload["phase_spans"]),
            fingerprint=dict(payload["fingerprint"]),
            seed=None if payload["seed"] is None else int(payload["seed"]),
            episode_index=(
                None if payload["episode_index"] is None else int(payload["episode_index"])
            ),
            miscalibration=(
                None
                if miscalibration is None
                else MiscalibrationRecord.from_json(miscalibration)
            ),
            wrist_camera_mount=(
                None if mount is None else WristCameraMount.from_json(mount)
            ),
            recorded_appearance=(
                None if appearance is None else AppearanceDraw.from_json(appearance)
            ),
        )


@dataclasses.dataclass(frozen=True)
class TrajectoryArtifact:
    """One episode: its frames, and the facts that hold across all of them."""

    frames: TrajectoryFrames
    facts: EpisodeFacts

    def placement_error(self) -> PlacementError:
        """Where the cube ended up, relative to where it was meant to go.

        Recomputed from the last frame rather than stored, so it cannot disagree
        with the trajectory it summarizes.
        """
        if not len(self.frames):
            raise ValueError("an episode with no frames has no placement error")
        cube = self.frames.true_cube_pose[-1]
        cube_xyz = (float(cube[0]), float(cube[1]), float(cube[2]))
        target_xyz = (*self.facts.target_xy, float(CUBE_HALF_SIZE))
        dx = cube_xyz[0] - target_xyz[0]
        dy = cube_xyz[1] - target_xyz[1]
        dz = cube_xyz[2] - target_xyz[2]
        return PlacementError(
            cube_xyz=cube_xyz,
            target_xyz=target_xyz,
            dx=dx,
            dy=dy,
            dz=dz,
            xy=math.hypot(dx, dy),
        )


class TrajectoryWriter:
    """Collects an episode's frames as it is generated.

    Owns the phase spans too, because it is the one object that sees every tick's
    phase name — which means an episode has exact spans whether or not anything
    is capturing images from it.
    """

    def __init__(self) -> None:
        self._rows: dict[str, list[np.ndarray]] = {name: [] for name in FRAME_ARRAY_WIDTHS}
        self._spans: list[PhaseSpan] = []

    def __len__(self) -> int:
        return len(self._rows["true_state"])

    @property
    def spans(self) -> tuple[PhaseSpan, ...]:
        return tuple(self._spans)

    def record(
        self,
        *,
        phase_name: str,
        true_state: np.ndarray,
        believed_state: np.ndarray,
        action: np.ndarray,
        true_cube_pose: np.ndarray,
        believed_cube_pose: CubePose,
        wrist_sighting: CubePose | None,
    ) -> None:
        """Append one control tick."""
        if not self._spans or self._spans[-1].name != phase_name:
            self._spans.append(PhaseSpan(name=phase_name, start_frame=len(self)))
        nothing_seen = np.full(_EULER_POSE, np.nan, dtype=np.float32)
        for name, row in (
            ("true_state", true_state),
            ("believed_state", believed_state),
            ("action", action),
            ("true_cube_pose", true_cube_pose),
            ("believed_cube_pose", cube_pose_row(believed_cube_pose)),
            (
                "wrist_sighting",
                nothing_seen if wrist_sighting is None else cube_pose_row(wrist_sighting),
            ),
        ):
            self._rows[name].append(np.asarray(row, dtype=np.float32).copy())

    def frames(self) -> TrajectoryFrames:
        """The accumulated rows, stacked."""
        stacked = {
            name: (
                np.stack(rows)
                if rows
                else np.zeros((0, FRAME_ARRAY_WIDTHS[name]), dtype=np.float32)
            )
            for name, rows in self._rows.items()
        }
        return TrajectoryFrames(**stacked)


def save_trajectory(path: Path, artifact: TrajectoryArtifact) -> None:
    """Write one episode's artifact as an uncompressed NPZ.

    Uncompressed because the arrays are small, the episode metadata rides along
    as one JSON member, and a stored archive can be memory-mapped later without
    reading it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(artifact.frames, name) for name in FRAME_ARRAY_WIDTHS}
    payload[_METADATA_KEY] = np.frombuffer(
        json.dumps(artifact.facts.as_json(), sort_keys=True).encode(), dtype=np.uint8
    )
    with path.open("wb") as file:
        np.savez(file, **payload)


def load_trajectory(path: Path) -> TrajectoryArtifact:
    """Read back an artifact written by :func:`save_trajectory`."""
    with np.load(path) as archive:
        facts = EpisodeFacts.from_json(
            json.loads(bytes(archive[_METADATA_KEY]).decode())
        )
        frames = TrajectoryFrames(
            **{name: np.asarray(archive[name]) for name in FRAME_ARRAY_WIDTHS}
        )
    return TrajectoryArtifact(frames=frames, facts=facts)


def render_environment_fingerprint(
    *, render_hw: tuple[int, int] | None = None, image_hw: tuple[int, int] | None = None
) -> dict[str, Any]:
    """Identify everything outside the artifact that moves a rendered pixel.

    An artifact is portable; the pixels made from it are not. The camera
    calibrations are machine-local (gitignored) files, and the OpenGL backend and
    MuJoCo version decide the shading, so two variant sets generated on different
    machines will not match. Without the fingerprint that surfaces as a confusing
    training result rather than as an error.

    ``render_hw``/``image_hw`` belong to a render pass rather than to an episode,
    so they are recorded only when a caller supplies them.
    """
    # Imported here rather than at module scope: MuJoCo is the simulator's
    # dependency, and reading an artifact must not require it.
    import mujoco

    def digest(payload: object) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    extrinsics = load_local_camera_extrinsics()
    intrinsics = load_local_camera_intrinsics()
    fingerprint: dict[str, Any] = {
        "mujoco_version": mujoco.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "platform": f"{platform.system()}-{platform.machine()}",
        "camera_extrinsics": sorted(extrinsics),
        "camera_extrinsics_digest": digest(extrinsics),
        "camera_intrinsics": sorted(intrinsics),
        "camera_intrinsics_digest": digest(intrinsics),
    }
    if render_hw is not None:
        fingerprint["render_hw"] = list(render_hw)
    if image_hw is not None:
        fingerprint["image_hw"] = list(image_hw)
    return fingerprint
