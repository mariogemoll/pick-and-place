# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Data loading and metrics for the DPPO chunk-imitation audit."""

from __future__ import annotations

import json
import struct
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import pyarrow.parquet as pq

from pick_and_place.diffusion_policy_dataset import CAMERA_FEATURES
from pick_and_place.follower import ARM_JOINT_NAMES, JOINT_NAMES
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    PolicyObservation,
)
from pick_and_place.sim_recorder import resize_and_center_crop

SOURCE_HZ = 30
POLICY_HZ = 10
FRAME_STRIDE = SOURCE_HZ // POLICY_HZ
HISTORY_STEPS = 2
HORIZON_STEPS = 16
SUMMARY_STEPS = (0, 1, 3, 7, 15)
HELD_OUT_FIRST_EPISODE = 1017
HELD_OUT_LAST_EPISODE = 1117
FAILED_HELD_OUT_EPISODE = 1110


@dataclass(frozen=True)
class ExampleRef:
    """A valid chunk start, addressed in the 10 Hz episode timeline."""

    episode: str
    index: int


@dataclass(frozen=True)
class AuditExample:
    """One explicit observation history and its demonstrated action chunk."""

    ref: ExampleRef
    observations: tuple[PolicyObservation, PolicyObservation]
    target_actions: np.ndarray
    source_frame_indices: tuple[int, int] | None = None


def held_out_episode_names() -> tuple[str, ...]:
    """Return the audit's exact 100 successful standalone episode names."""
    return tuple(
        f"ep{episode:06d}"
        for episode in range(HELD_OUT_FIRST_EPISODE, HELD_OUT_LAST_EPISODE + 1)
        if episode != FAILED_HELD_OUT_EPISODE
    )


def sample_refs(refs: list[ExampleRef], count: int, *, seed: int) -> list[ExampleRef]:
    """Select examples without replacement in a reproducible order."""
    if count < 1:
        raise ValueError("sample count must be positive")
    if not refs:
        raise ValueError("split has no complete examples")
    if count > len(refs):
        raise ValueError(f"requested {count} examples, but split has only {len(refs)}")
    selected = np.random.default_rng(seed).choice(len(refs), size=count, replace=False)
    return [refs[int(index)] for index in selected]


def _valid_refs(episode_lengths: Iterable[tuple[str, int]]) -> list[ExampleRef]:
    refs = []
    for episode, length in episode_lengths:
        # t=0 lacks the required preceding observation. The final valid start
        # satisfies t + HORIZON_STEPS <= length.
        refs.extend(ExampleRef(episode, index) for index in range(1, length - HORIZON_STEPS + 1))
    return refs


class StoredNpz:
    """Memory-map arrays from the exporter's uncompressed NPZ container.

    ``numpy.load`` materializes an entire member. That is unacceptable for the
    5.35 GB image member, while the exporter deliberately stores NPY members
    without ZIP compression. Mapping the NPY payload in place gives efficient
    random access without changing the canonical training artifact.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def array(self, name: str) -> np.memmap:
        member_name = f"{name}.npy"
        with zipfile.ZipFile(self.path) as archive:
            info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member_name} must be stored without ZIP compression")

        with self.path.open("rb") as file:
            file.seek(info.header_offset)
            local_header = file.read(30)
            if len(local_header) != 30:
                raise ValueError(f"truncated ZIP header for {member_name}")
            fields = struct.unpack("<4s5H3I2H", local_header)
            if fields[0] != b"PK\x03\x04":
                raise ValueError(f"invalid ZIP header for {member_name}")
            filename_length, extra_length = fields[-2:]
            file.seek(info.header_offset + 30 + filename_length + extra_length)
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
            else:
                raise ValueError(f"unsupported NPY version {version} for {member_name}")
            payload_offset = file.tell()

        return np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=payload_offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )


def _unnormalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return (values + 1.0) / 2.0 * (maximum - minimum + 1e-6) + minimum


class TrainingArtifactSplit:
    """Random-access view of the exact stitched 10 Hz training artifact."""

    def __init__(self, dataset_path: Path, normalization_path: Path) -> None:
        archive = StoredNpz(dataset_path)
        self.states = archive.array("states")
        self.actions = archive.array("actions")
        self.images = archive.array("images")
        self.traj_lengths = archive.array("traj_lengths")
        with np.load(normalization_path, allow_pickle=False) as bounds:
            self.obs_min = bounds["obs_min"].astype(np.float32)
            self.obs_max = bounds["obs_max"].astype(np.float32)
            self.action_min = bounds["action_min"].astype(np.float32)
            self.action_max = bounds["action_max"].astype(np.float32)

        total = int(self.traj_lengths.sum())
        expected_shapes = {
            "states": (total, len(JOINT_NAMES)),
            "actions": (total, len(JOINT_NAMES)),
            "images": (total, 6, 96, 96),
        }
        for name, expected in expected_shapes.items():
            actual = getattr(self, name).shape
            if actual != expected:
                raise ValueError(f"training {name} must have shape {expected}, got {actual}")
        self.offsets = np.concatenate(([0], np.cumsum(self.traj_lengths, dtype=np.int64)))

    def refs(self) -> list[ExampleRef]:
        return _valid_refs(
            (str(episode), int(length)) for episode, length in enumerate(self.traj_lengths)
        )

    def load_examples(self, refs: Iterable[ExampleRef]) -> list[AuditExample]:
        examples = []
        for ref in refs:
            episode = int(ref.episode)
            start = int(self.offsets[episode])
            history_slice = slice(start + ref.index - 1, start + ref.index + 1)
            target_slice = slice(
                start + ref.index,
                start + ref.index + HORIZON_STEPS,
            )
            states = _unnormalize(
                np.asarray(self.states[history_slice]), self.obs_min, self.obs_max
            ).astype(np.float32)
            targets = _unnormalize(
                np.asarray(self.actions[target_slice]), self.action_min, self.action_max
            ).astype(np.float32)
            images = np.asarray(self.images[history_slice])
            observations = tuple(
                {
                    STATE_FEATURE: states[history_index].copy(),
                    OVERHEAD_FEATURE: np.moveaxis(images[history_index, :3], 0, -1).copy(),
                    WRIST_FEATURE: np.moveaxis(images[history_index, 3:], 0, -1).copy(),
                }
                for history_index in range(HISTORY_STEPS)
            )
            examples.append(
                AuditExample(
                    ref=ref,
                    observations=observations,  # type: ignore[arg-type]
                    target_actions=targets,
                )
            )
        return examples


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _single_parquet_row(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"{path} must contain exactly one episode row")
    return table.to_pylist()[0]


def _format_path(pattern: str, *, row: dict[str, Any], video_key: str | None = None) -> Path:
    values = {
        "chunk_index": int(row["data/chunk_index"]),
        "file_index": int(row["data/file_index"]),
    }
    if video_key is not None:
        values = {
            "video_key": video_key,
            "chunk_index": int(row[f"videos/{video_key}/chunk_index"]),
            "file_index": int(row[f"videos/{video_key}/file_index"]),
        }
    return Path(pattern.format(**values))


@dataclass
class _HeldOutEpisode:
    name: str
    root: Path
    info: dict[str, Any]
    row: dict[str, Any]
    states: np.ndarray
    actions: np.ndarray

    @property
    def length_10hz(self) -> int:
        return (len(self.states) + FRAME_STRIDE - 1) // FRAME_STRIDE


class HeldOutSplit:
    """Standalone LeRobot episodes, downsampled with the artifact's contract.

    Defaults to the audit's fixed 100 standalone XPS episodes; pass
    ``episode_names`` to evaluate a different held-out recording batch with the
    same loading contract.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_episodes: int | None = None,
        episode_names: tuple[str, ...] | None = None,
    ) -> None:
        names = held_out_episode_names() if episode_names is None else episode_names
        if not names:
            raise ValueError("episode_names must be nonempty")
        if max_episodes is not None:
            if not 1 <= max_episodes <= len(names):
                raise ValueError(f"max_episodes must be in [1, {len(names)}]")
            names = names[:max_episodes]
        self.root = root.resolve()
        missing = [name for name in names if not (self.root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"held-out root is missing {len(missing)} episode(s), starting with {missing[0]}"
            )
        self.episodes = {name: self._load_episode(name) for name in names}

    def _load_episode(self, name: str) -> _HeldOutEpisode:
        root = self.root / name
        info = _read_json(root / "meta" / "info.json")
        source_hz = int(info.get("fps", 0))
        if source_hz != SOURCE_HZ:
            raise ValueError(f"{name} must be recorded at {SOURCE_HZ} Hz, got {source_hz}")
        episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if len(episode_paths) != 1:
            raise ValueError(f"{name} must contain one episode metadata parquet")
        row = _single_parquet_row(episode_paths[0])
        data_path = root / _format_path(info["data_path"], row=row)
        table = pq.read_table(
            data_path,
            columns=["frame_index", "observation.state", "action"],
        ).sort_by("frame_index")
        frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
        if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
            raise ValueError(f"{name} frame indices must be contiguous and start at zero")
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.shape != actions.shape or states.shape[1:] != (len(JOINT_NAMES),):
            raise ValueError(f"{name} has malformed state/action arrays")
        if int(row["length"]) != len(states):
            raise ValueError(f"{name} data length does not match episode metadata")
        return _HeldOutEpisode(name, root, info, row, states, actions)

    def refs(self) -> list[ExampleRef]:
        return _valid_refs(
            (episode.name, episode.length_10hz) for episode in self.episodes.values()
        )

    @staticmethod
    def _decode_frames(
        episode: _HeldOutEpisode,
        feature: str,
        source_indices: set[int],
    ) -> dict[int, np.ndarray]:
        relative_path = _format_path(episode.info["video_path"], row=episode.row, video_key=feature)
        video_path = episode.root / relative_path
        start = round(float(episode.row[f"videos/{feature}/from_timestamp"]) * SOURCE_HZ)
        absolute_indices = {start + index: index for index in source_indices}
        decoded = {}
        with av.open(str(video_path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                source_index = absolute_indices.get(frame_index)
                if source_index is not None:
                    image = frame.to_ndarray(format="rgb24")
                    decoded[source_index] = resize_and_center_crop(image, 96, 96)
                    if len(decoded) == len(source_indices):
                        break
        missing = source_indices.difference(decoded)
        if missing:
            raise ValueError(f"decoded video {video_path} is missing source frame {min(missing)}")
        return decoded

    def load_examples(self, refs: Iterable[ExampleRef]) -> list[AuditExample]:
        refs = list(refs)
        refs_by_episode: dict[str, list[ExampleRef]] = defaultdict(list)
        for ref in refs:
            refs_by_episode[ref.episode].append(ref)

        images_by_episode: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for name, episode_refs in refs_by_episode.items():
            source_indices = {
                FRAME_STRIDE * history_index
                for ref in episode_refs
                for history_index in (ref.index - 1, ref.index)
            }
            episode = self.episodes[name]
            images_by_episode[name] = {
                feature: self._decode_frames(episode, feature, source_indices)
                for feature in CAMERA_FEATURES
            }

        examples = []
        for ref in refs:
            episode = self.episodes[ref.episode]
            history_indices = (ref.index - 1, ref.index)
            source_indices = tuple(index * FRAME_STRIDE for index in history_indices)
            observations = tuple(
                {
                    STATE_FEATURE: episode.states[source_index].copy(),
                    OVERHEAD_FEATURE: images_by_episode[ref.episode][CAMERA_FEATURES[0]][
                        source_index
                    ].copy(),
                    WRIST_FEATURE: images_by_episode[ref.episode][CAMERA_FEATURES[1]][
                        source_index
                    ].copy(),
                }
                for source_index in source_indices
            )
            target_source_indices = FRAME_STRIDE * np.arange(ref.index, ref.index + HORIZON_STEPS)
            targets = episode.actions[target_source_indices].copy()
            examples.append(
                AuditExample(
                    ref=ref,
                    observations=observations,  # type: ignore[arg-type]
                    target_actions=targets,
                    source_frame_indices=source_indices,
                )
            )
        return examples


def distribution(values: np.ndarray) -> dict[str, float]:
    """Summarize one physical-unit error distribution."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_errors(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Aggregate requested per-joint MAE and five-joint vector L2 errors."""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    expected_tail = (HORIZON_STEPS, len(JOINT_NAMES))
    if predictions.shape != targets.shape or predictions.shape[1:] != expected_tail:
        raise ValueError(
            f"predictions and targets must both have shape (N, {HORIZON_STEPS}, "
            f"{len(JOINT_NAMES)}), got {predictions.shape} and {targets.shape}"
        )
    absolute = np.abs(predictions - targets)
    arm_l2 = np.linalg.norm(
        predictions[..., : len(ARM_JOINT_NAMES)] - targets[..., : len(ARM_JOINT_NAMES)],
        axis=-1,
    )

    def summarize_selection(step: int | None) -> dict[str, Any]:
        selected_absolute = absolute if step is None else absolute[:, step]
        selected_l2 = arm_l2 if step is None else arm_l2[:, step]
        return {
            "joint_absolute_error": {
                name: distribution(selected_absolute[..., index])
                for index, name in enumerate(JOINT_NAMES)
            },
            "arm_vector_l2": distribution(selected_l2),
        }

    per_step = {str(step): summarize_selection(step) for step in range(HORIZON_STEPS)}
    return {
        "num_examples": int(len(predictions)),
        "horizon_steps": HORIZON_STEPS,
        "per_step": per_step,
        "selected_steps": {str(step): per_step[str(step)] for step in SUMMARY_STEPS},
        "all_steps": summarize_selection(None),
    }


def example_record(
    split: str,
    example: AuditExample,
    prediction: np.ndarray,
    *,
    sampler: str,
    sampling_seed: int,
    panel_path: str | None,
) -> dict[str, Any]:
    """Build the JSONL record for one evaluated chunk."""
    prediction = np.asarray(prediction, dtype=np.float32)
    error = prediction - example.target_actions
    return {
        "split": split,
        "episode": example.ref.episode,
        "index_10hz": example.ref.index,
        "history_indices_10hz": [example.ref.index - 1, example.ref.index],
        "history_indices_30hz": (
            list(example.source_frame_indices) if example.source_frame_indices is not None else None
        ),
        "sampler": sampler,
        "sampling_seed": sampling_seed,
        "target_actions": example.target_actions.tolist(),
        "predicted_actions": prediction.tolist(),
        "absolute_error": np.abs(error).tolist(),
        "arm_vector_l2": np.linalg.norm(error[:, : len(ARM_JOINT_NAMES)], axis=-1).tolist(),
        "panel": panel_path,
    }


def save_panel(
    path: Path,
    split: str,
    example: AuditExample,
    prediction: np.ndarray,
) -> None:
    """Save both input frames and a physical-unit action comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 4)
    camera_labels = ("overhead", "wrist")
    features = (OVERHEAD_FEATURE, WRIST_FEATURE)
    for history_index, observation in enumerate(example.observations):
        for camera_index, (label, feature) in enumerate(zip(camera_labels, features, strict=True)):
            axis = figure.add_subplot(grid[history_index, 2 * camera_index])
            axis.imshow(observation[feature])
            axis.set_title(f"t{history_index - 1:+d} {label}")
            axis.axis("off")
            if camera_index == 0:
                state_axis = figure.add_subplot(grid[history_index, 1])
                state_axis.axis("off")
                state = observation[STATE_FEATURE]
                text = "\n".join(
                    f"{name}: {state[index]:7.2f}" for index, name in enumerate(JOINT_NAMES)
                )
                state_axis.text(0.0, 0.95, text, va="top", family="monospace")
            else:
                error_axis = figure.add_subplot(grid[history_index, 3])
                error_axis.axis("off")
                if history_index == 0:
                    error = np.abs(prediction - example.target_actions)
                    error_axis.text(
                        0.0,
                        0.95,
                        f"mean |error|: {error.mean():.3f}\n"
                        f"step-0 arm L2: {np.linalg.norm(error[0, :5]):.3f}",
                        va="top",
                    )

    action_grid = grid[2, :].subgridspec(2, 3)
    steps = np.arange(HORIZON_STEPS)
    for index, name in enumerate(JOINT_NAMES):
        axis = figure.add_subplot(action_grid[index // 3, index % 3])
        axis.plot(steps, example.target_actions[:, index], label="demonstrated")
        axis.plot(steps, prediction[:, index], label="predicted")
        axis.set_title(name)
        axis.set_xlabel("10 Hz horizon step")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend()
    figure.suptitle(
        f"{split}: episode {example.ref.episode}, index {example.ref.index}",
        fontsize=14,
    )
    figure.savefig(path, dpi=120)
    plt.close(figure)
