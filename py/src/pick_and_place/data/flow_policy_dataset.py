# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export privileged simulation demonstrations as generic CondOT examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pick_and_place.spec.robot import JOINT_NAMES

STATE_FEATURE = "observation.state"
ENVIRONMENT_FEATURE = "observation.environment_state"
ACTION_FEATURE = "action"
FORMAT_VERSION = "flow-policy-state-v1"

OBSERVATION_NAMES = (*JOINT_NAMES, "cube_x", "cube_y", "cube_z", "target_x", "target_y")


@dataclass(frozen=True)
class EpisodeValues:
    """Aligned observations and endpoints for one decimated episode."""

    episode_index: int
    observations: np.ndarray
    endpoints: np.ndarray


def normalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Map columns to approximately ``[-1, 1]`` using fixed bounds."""
    values = np.asarray(values, dtype=np.float32)
    span = maximum - minimum
    normalized = 2.0 * (values - minimum) / np.where(span > 1e-6, span, 1.0) - 1.0
    return np.where(span > 1e-6, normalized, 0.0).astype(np.float32)


def make_examples(
    observations: np.ndarray,
    endpoints: np.ndarray,
    *,
    observation_steps: int,
    prediction_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one episode into padded, flattened conditional examples.

    The example at tick ``t`` repeats the first observation when its history
    reaches before the episode and repeats the final endpoint when its prediction
    reaches beyond it. Consequently every recorded tick contributes exactly one
    example and no example can cross an episode boundary.
    """
    observations = np.asarray(observations, dtype=np.float32)
    endpoints = np.asarray(endpoints, dtype=np.float32)
    if observations.ndim != 2 or endpoints.ndim != 2:
        raise ValueError("observations and endpoints must both be rank-2 arrays")
    if len(observations) != len(endpoints) or not len(observations):
        raise ValueError("observations and endpoints must have the same nonzero length")
    if observation_steps < 1 or prediction_steps < 1:
        raise ValueError("observation_steps and prediction_steps must be positive")

    ticks = np.arange(len(observations))[:, None]
    history_offsets = np.arange(observation_steps - 1, -1, -1)[None, :]
    history_indices = np.maximum(ticks - history_offsets, 0)
    prediction_offsets = np.arange(prediction_steps)[None, :]
    prediction_indices = np.minimum(ticks + prediction_offsets, len(endpoints) - 1)
    conditions = observations[history_indices].reshape(len(observations), -1)
    flattened_endpoints = endpoints[prediction_indices].reshape(len(endpoints), -1)
    return conditions.astype(np.float32), flattened_endpoints.astype(np.float32)


def split_episode_indices(
    episode_indices: list[int], *, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split whole episodes deterministically, retaining source order in each split."""
    if not episode_indices:
        raise ValueError("at least one episode is required")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode indices must be unique")
    if validation_fraction == 0.0:
        return list(episode_indices), []
    if len(episode_indices) < 2:
        raise ValueError("at least two episodes are required for a validation split")

    validation_count = min(
        len(episode_indices) - 1,
        max(1, round(len(episode_indices) * validation_fraction)),
    )
    rng = np.random.default_rng(seed)
    validation = set(
        int(value)
        for value in rng.choice(episode_indices, size=validation_count, replace=False)
    )
    training_indices = [index for index in episode_indices if index not in validation]
    validation_indices = [index for index in episode_indices if index in validation]
    return training_indices, validation_indices


def prepare_splits(
    episodes: list[EpisodeValues],
    *,
    observation_steps: int,
    prediction_steps: int,
    validation_fraction: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[int], list[int]]:
    """Normalize from training episodes and construct generic examples."""
    if not episodes:
        raise ValueError("at least one episode is required")
    observation_dim = episodes[0].observations.shape[1]
    endpoint_dim = episodes[0].endpoints.shape[1]
    for episode in episodes:
        if episode.observations.ndim != 2 or episode.observations.shape[1] != observation_dim:
            raise ValueError("all episode observations must have the same rank-2 shape")
        if episode.endpoints.ndim != 2 or episode.endpoints.shape[1] != endpoint_dim:
            raise ValueError("all episode endpoints must have the same rank-2 shape")
        if len(episode.observations) != len(episode.endpoints) or not len(episode.observations):
            raise ValueError("each episode must contain aligned nonempty arrays")

    training_indices, validation_indices = split_episode_indices(
        [episode.episode_index for episode in episodes],
        validation_fraction=validation_fraction,
        seed=seed,
    )
    training_set = set(training_indices)
    training_episodes = [episode for episode in episodes if episode.episode_index in training_set]
    training_observations = np.concatenate(
        [episode.observations for episode in training_episodes]
    ).astype(np.float32)
    training_endpoints = np.concatenate([episode.endpoints for episode in training_episodes]).astype(
        np.float32
    )
    bounds = {
        "observation_min": training_observations.min(axis=0),
        "observation_max": training_observations.max(axis=0),
        "endpoint_min": training_endpoints.min(axis=0),
        "endpoint_max": training_endpoints.max(axis=0),
    }

    def build(indices: list[int]) -> dict[str, np.ndarray]:
        selected = set(indices)
        pairs = []
        for episode in episodes:
            if episode.episode_index not in selected:
                continue
            normalized_observations = normalize(
                episode.observations,
                bounds["observation_min"],
                bounds["observation_max"],
            )
            normalized_endpoints = normalize(
                episode.endpoints,
                bounds["endpoint_min"],
                bounds["endpoint_max"],
            )
            pairs.append(
                make_examples(
                    normalized_observations,
                    normalized_endpoints,
                    observation_steps=observation_steps,
                    prediction_steps=prediction_steps,
                )
            )
        if not pairs:
            return {
                "observations": np.empty((0, observation_steps * observation_dim), np.float32),
                "data": np.empty((0, prediction_steps * endpoint_dim), np.float32),
            }
        return {
            "observations": np.concatenate([pair[0] for pair in pairs]),
            "data": np.concatenate([pair[1] for pair in pairs]),
        }

    return build(training_indices), build(validation_indices), bounds, training_indices, validation_indices


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _episode_rows(dataset_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode metadata found under {dataset_root}")
    rows = sorted(
        pa.concat_tables([pq.read_table(path) for path in paths]).to_pylist(),
        key=lambda row: int(row["episode_index"]),
    )
    return rows, paths


def _data_path(dataset_root: Path, info: dict[str, Any], row: dict[str, Any]) -> Path:
    return dataset_root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )


def _load_episodes(
    dataset_root: Path,
    info: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    frame_stride: int,
) -> tuple[list[EpisodeValues], list[Path]]:
    paths = sorted({_data_path(dataset_root, info, row) for row in rows})
    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=["index", "episode_index", STATE_FEATURE, ENVIRONMENT_FEATURE, ACTION_FEATURE],
            )
            for path in paths
        ]
    )
    selected_indices = pa.array([int(row["episode_index"]) for row in rows], type=pa.int64())
    table = table.filter(pc.is_in(table["episode_index"], value_set=selected_indices)).sort_by(
        "index"
    )
    actual_indices = table["episode_index"].to_numpy(zero_copy_only=False)
    expected_indices = np.concatenate(
        [np.full(int(row["length"]), int(row["episode_index"]), dtype=np.int64) for row in rows]
    )
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("data rows do not match the selected episode metadata")

    states = np.asarray(table[STATE_FEATURE].to_pylist(), dtype=np.float32)
    environment = np.asarray(table[ENVIRONMENT_FEATURE].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[ACTION_FEATURE].to_pylist(), dtype=np.float32)
    episodes = []
    start = 0
    for row in rows:
        length = int(row["length"])
        keep = start + np.arange(0, length, frame_stride, dtype=np.int64)
        target = np.array([row["target_x"], row["target_y"]], dtype=np.float32)
        targets = np.broadcast_to(target, (len(keep), 2))
        observations = np.concatenate((states[keep], environment[keep, :3], targets), axis=1)
        episodes.append(
            EpisodeValues(
                episode_index=int(row["episode_index"]),
                observations=observations.astype(np.float32),
                endpoints=actions[keep].astype(np.float32),
            )
        )
        start += length
    return episodes, paths


def _fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def export_flow_policy_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    policy_hz: int = 10,
    observation_steps: int = 2,
    prediction_steps: int = 16,
    validation_fraction: float = 0.1,
    seed: int = 0,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Export a LeRobot simulation dataset as semantics-free CondOT pairs."""
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    building_dir = output_dir.with_name(f"{output_dir.name}.building")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if building_dir.exists():
        raise FileExistsError(f"incomplete export already exists: {building_dir}")
    if min(policy_hz, observation_steps, prediction_steps) < 1:
        raise ValueError("policy_hz, observation_steps, and prediction_steps must be positive")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")

    info_path = dataset_root / "meta" / "info.json"
    info = _read_json(info_path)
    required_features = {STATE_FEATURE, ENVIRONMENT_FEATURE, ACTION_FEATURE}
    features = info.get("features", {})
    if not required_features <= set(features):
        raise ValueError(f"dataset is missing features: {sorted(required_features - set(features))}")
    source_fps = int(info.get("fps", 0))
    if source_fps < 1 or source_fps % policy_hz:
        raise ValueError(f"source fps {source_fps} must be a positive multiple of policy_hz")
    if tuple(features[STATE_FEATURE].get("names", ())) != JOINT_NAMES:
        raise ValueError("observation.state names do not match the project's joint order")
    if tuple(features[ACTION_FEATURE].get("names", ())) != JOINT_NAMES:
        raise ValueError("action names do not match the project's joint order")
    environment_names = tuple(features[ENVIRONMENT_FEATURE].get("names", ()))
    if environment_names[:3] != ("cube_x", "cube_y", "cube_z"):
        raise ValueError("environment state must begin with cube_x, cube_y, cube_z")

    rows, metadata_paths = _episode_rows(dataset_root)
    if max_episodes is not None:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("no episodes selected")
    episodes, data_paths = _load_episodes(
        dataset_root,
        info,
        rows,
        frame_stride=source_fps // policy_hz,
    )
    training, validation, bounds, training_indices, validation_indices = prepare_splits(
        episodes,
        observation_steps=observation_steps,
        prediction_steps=prediction_steps,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    building_dir.mkdir(parents=True)
    np.savez_compressed(building_dir / "train.npz", **training)
    np.savez_compressed(building_dir / "validation.npz", **validation)
    np.savez_compressed(building_dir / "normalization.npz", **bounds)
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_dataset": str(dataset_root),
        "source_sha256": _fingerprint([info_path, *metadata_paths, *data_paths], dataset_root),
        "source_fps": source_fps,
        "policy_hz": policy_hz,
        "frame_stride": source_fps // policy_hz,
        "observation_steps": observation_steps,
        "prediction_steps": prediction_steps,
        "observation_dim": len(OBSERVATION_NAMES),
        "endpoint_dim": len(JOINT_NAMES),
        "condition_dim": int(training["observations"].shape[1]),
        "output_dim": int(training["data"].shape[1]),
        "observation_names": list(OBSERVATION_NAMES),
        "endpoint_names": list(JOINT_NAMES),
        "endpoint_semantics": "absolute joint command",
        "normalization": "per-dimension training-split min-max to [-1, 1]",
        "padding": "repeat first observation and final endpoint within each episode",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "training_episode_indices": training_indices,
        "validation_episode_indices": validation_indices,
        "training_examples": int(len(training["observations"])),
        "validation_examples": int(len(validation["observations"])),
    }
    with (building_dir / "export.json").open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    building_dir.rename(output_dir)
    return manifest
