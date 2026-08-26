# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export LeRobot demonstrations as stitched Diffusion Policy arrays."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap
from tqdm import tqdm

from pick_and_place.core.image_ops import resize_and_center_crop
from pick_and_place.spec.action_encoding import (
    ACTION_ENCODING_KEY,
    ActionEncoding,
    encode_actions,
)

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
CAMERA_FEATURES = (
    "observation.images.overhead",
    "observation.images.wrist",
)
FORMAT_VERSION = "diffusion-policy-stitched-v1"
DEFAULT_POLICY_HZ = 10
#: ``goal_source`` value for the goal slot fed by each episode's target point.
GOAL_TARGET_XY = "episode_target_xy"
_GOAL_TARGET_XY_COLUMNS = ("target_x", "target_y")
_CAMERA_PROGRESS_BATCH = 128
_camera_progress_counter: Any | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode metadata found under {dataset_root}")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    rows = sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
    indices = [int(row["episode_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("episode metadata contains duplicate episode indices")
    return rows


def _chunked_path(pattern: str, *, chunk_index: int, file_index: int, **values: Any) -> Path:
    return Path(
        pattern.format(
            chunk_index=chunk_index,
            file_index=file_index,
            **values,
        )
    )


def _data_path(dataset_root: Path, info: dict[str, Any], row: dict[str, Any]) -> Path:
    return dataset_root / _chunked_path(
        info["data_path"],
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )


def _video_path(
    dataset_root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
    feature: str,
) -> Path:
    return dataset_root / _chunked_path(
        info["video_path"],
        video_key=feature,
        chunk_index=int(row[f"videos/{feature}/chunk_index"]),
        file_index=int(row[f"videos/{feature}/file_index"]),
    )


def _load_low_dimensional_arrays(
    dataset_root: Path,
    info: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted({_data_path(dataset_root, info, row) for row in rows})
    tables = [
        pq.read_table(path, columns=["index", "episode_index", STATE_FEATURE, ACTION_FEATURE])
        for path in paths
    ]
    table = pa.concat_tables(tables)
    selected_indices = pa.array([int(row["episode_index"]) for row in rows], type=pa.int64())
    table = table.filter(pc.is_in(table["episode_index"], value_set=selected_indices))
    table = table.sort_by("index")

    expected_episode_indices = np.concatenate(
        [np.full(int(row["length"]), int(row["episode_index"]), dtype=np.int64) for row in rows]
    )
    actual_episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
    if not np.array_equal(actual_episode_indices, expected_episode_indices):
        raise ValueError("data rows do not match the selected episode metadata")

    states = np.asarray(table[STATE_FEATURE].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[ACTION_FEATURE].to_pylist(), dtype=np.float32)
    return states, actions


def episode_goal_vectors(
    rows: list[dict[str, Any]], source_lengths: list[int]
) -> np.ndarray:
    """Per-frame goal slot, repeated from each episode's recorded target point.

    The goal is an episode-level fact, so every frame of an episode carries the
    same value. It is appended to the state vector *before* normalization on
    purpose: the goal then gets min-max bounds fitted alongside every other
    state dimension, and a rollout normalizes a live target through the same
    ``normalization.npz`` it already loads rather than through a second path
    that could drift out of sync with it.
    """
    missing = [name for name in _GOAL_TARGET_XY_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(
            f"episode metadata has no {missing[0]!r}; this dataset was recorded "
            f"without a target point, so it cannot be exported with goal={GOAL_TARGET_XY!r}"
        )
    goals = np.asarray(
        [[float(row[name]) for name in _GOAL_TARGET_XY_COLUMNS] for row in rows],
        dtype=np.float32,
    )
    if not np.isfinite(goals).all():
        raise ValueError("episode target points contain non-finite values")
    return np.repeat(goals, source_lengths, axis=0)


def _check_goal_varies(goals: np.ndarray) -> None:
    """Refuse a goal slot that is constant across the whole export.

    A constant column min-max normalizes to a constant, so the policy would be
    conditioned on a dimension carrying no information, and a rollout asking for
    any other target would be extrapolating off the fitted bounds. Both failures
    are silent at training time, which is why this is an error and not a warning.
    """
    if len(goals) and np.allclose(goals.min(axis=0), goals.max(axis=0)):
        raise ValueError(
            "every episode shares one target point, so the goal slot would carry "
            "no information; export a dataset whose targets vary, or pass goal=None"
        )


def normalize_min_max(
    values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize columns to the policy's ``[-1, 1]`` convention.

    ``bounds`` squashes into a scale someone else already fixed instead of the
    one this data implies. Fitting fresh bounds is right for a dataset a policy
    will be trained on from scratch, and wrong for one that continues training a
    checkpoint: the weights learned what a normalized unit means under the
    original bounds, so re-fitting moves the input and action scales out from
    under them, and what should measure adaptation partly measures recovery from
    a rescaling.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected a non-empty rank-2 array, got {values.shape}")
    if bounds is None:
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
    else:
        minimum, maximum = (np.asarray(bound, dtype=np.float32) for bound in bounds)
        if minimum.shape != values.shape[1:] or maximum.shape != values.shape[1:]:
            raise ValueError(
                f"supplied bounds have shape {minimum.shape}/{maximum.shape}, "
                f"but the data has {values.shape[1]} columns"
            )
        if np.any(maximum < minimum):
            raise ValueError("supplied bounds are inverted")
    normalized = 2.0 * (values - minimum) / (maximum - minimum + 1e-6) - 1.0
    return normalized.astype(np.float32), minimum, maximum


def decimated_length(length: int, frame_stride: int) -> int:
    """Frames kept from a ``length``-frame episode when taking every Nth frame."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    return (length + frame_stride - 1) // frame_stride


def _decimated_indices(lengths: list[int], frame_stride: int) -> np.ndarray:
    """Return stitched-array indices sampled relative to each episode start."""
    indices: list[np.ndarray] = []
    source_start = 0
    for length in lengths:
        indices.append(source_start + np.arange(0, length, frame_stride, dtype=np.int64))
        source_start += length
    if not indices:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(indices)


def _video_segments(
    dataset_root: Path,
    info: dict[str, Any],
    rows: list[dict[str, Any]],
    feature: str,
    frame_stride: int,
) -> dict[Path, list[tuple[int, int, int]]]:
    fps = float(info["fps"])
    output_start = 0
    segments: dict[Path, list[tuple[int, int, int]]] = defaultdict(list)
    for row in rows:
        length = int(row["length"])
        start_frame = round(float(row[f"videos/{feature}/from_timestamp"]) * fps)
        path = _video_path(dataset_root, info, row, feature)
        segments[path].append((start_frame, length, output_start))
        output_start += decimated_length(length, frame_stride)
    return dict(segments)


def _write_camera_images(
    destination: np.memmap,
    *,
    channel_offset: int,
    dataset_root: Path,
    info: dict[str, Any],
    rows: list[dict[str, Any]],
    feature: str,
    image_size: int,
    frame_stride: int,
    progress_position: int = 0,
) -> list[Path]:
    segments_by_path = _video_segments(dataset_root, info, rows, feature, frame_stride)
    expected = sum(decimated_length(int(row["length"]), frame_stride) for row in rows)
    written = 0
    camera_name = feature.rsplit(".", maxsplit=1)[-1]
    with tqdm(
        total=expected,
        desc=f"Export {camera_name}",
        unit="frame",
        dynamic_ncols=True,
        position=progress_position,
    ) as progress:
        for path, segments in segments_by_path.items():
            written += _write_camera_video(
                destination,
                channel_offset=channel_offset,
                path=path,
                segments=segments,
                feature=feature,
                image_size=image_size,
                frame_stride=frame_stride,
                report_progress=progress.update,
            )

    if written != expected:
        raise ValueError(f"decoded {written} {feature} frames; expected {expected}")
    return sorted(segments_by_path)


def _write_camera_video(
    destination: np.memmap,
    *,
    channel_offset: int,
    path: Path,
    segments: list[tuple[int, int, int]],
    feature: str,
    image_size: int,
    frame_stride: int,
    report_progress,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    segments.sort()
    expected = sum(decimated_length(length, frame_stride) for _, length, _ in segments)
    written = 0
    unreported = 0
    segment_index = 0
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            while (
                segment_index < len(segments)
                and frame_index >= segments[segment_index][0] + segments[segment_index][1]
            ):
                segment_index += 1
            if segment_index == len(segments):
                break
            start, length, output_start = segments[segment_index]
            if frame_index < start:
                continue
            offset = frame_index - start
            if offset >= length or offset % frame_stride:
                continue
            image = frame.to_ndarray(format="rgb24")
            image = resize_and_center_crop(image, image_size, image_size)
            destination[
                output_start + offset // frame_stride, channel_offset : channel_offset + 3
            ] = np.moveaxis(image, -1, 0)
            written += 1
            unreported += 1
            if unreported == _CAMERA_PROGRESS_BATCH:
                report_progress(unreported)
                unreported = 0
    if unreported:
        report_progress(unreported)
    if written != expected:
        raise ValueError(
            f"decoded {written} selected {feature} frames from {path}; expected {expected}"
        )
    return written


def _initialize_camera_worker(progress_counter: Any) -> None:
    global _camera_progress_counter
    _camera_progress_counter = progress_counter


def _report_camera_worker_progress(count: int) -> None:
    if _camera_progress_counter is None:
        return
    with _camera_progress_counter.get_lock():
        _camera_progress_counter.value += count


def _write_camera_video_worker(
    images_path: Path,
    *,
    channel_offset: int,
    path: Path,
    segments: list[tuple[int, int, int]],
    feature: str,
    image_size: int,
    frame_stride: int,
) -> tuple[Path, int]:
    images = np.load(images_path, mmap_mode="r+")
    try:
        written = _write_camera_video(
            images,
            channel_offset=channel_offset,
            path=path,
            segments=segments,
            feature=feature,
            image_size=image_size,
            frame_stride=frame_stride,
            report_progress=_report_camera_worker_progress,
        )
        return path, written
    finally:
        images.flush()
        del images


def _write_all_camera_images(
    images_path: Path,
    *,
    dataset_root: Path,
    info: dict[str, Any],
    rows: list[dict[str, Any]],
    image_size: int,
    frame_stride: int,
    workers: int,
) -> set[Path]:
    if workers == 1:
        images = np.load(images_path, mmap_mode="r+")
        try:
            video_paths: set[Path] = set()
            for camera_index, feature in enumerate(CAMERA_FEATURES):
                video_paths.update(
                    _write_camera_images(
                        images,
                        channel_offset=3 * camera_index,
                        dataset_root=dataset_root,
                        info=info,
                        rows=rows,
                        feature=feature,
                        image_size=image_size,
                        frame_stride=frame_stride,
                    )
                )
            return video_paths
        finally:
            images.flush()
            del images

    tasks = [
        (3 * camera_index, path, segments, feature)
        for camera_index, feature in enumerate(CAMERA_FEATURES)
        for path, segments in _video_segments(
            dataset_root, info, rows, feature, frame_stride
        ).items()
    ]
    expected = sum(
        decimated_length(length, frame_stride)
        for _, _, segments, _ in tasks
        for _, length, _ in segments
    )
    context = multiprocessing.get_context("spawn")
    progress_counter = context.Value("q", 0)
    video_paths: set[Path] = set()
    with (
        tqdm(total=expected, desc="Export cameras", unit="frame", dynamic_ncols=True) as progress,
        ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=context,
            initializer=_initialize_camera_worker,
            initargs=(progress_counter,),
        ) as executor,
    ):
        pending = {
            executor.submit(
                _write_camera_video_worker,
                images_path,
                channel_offset=channel_offset,
                path=path,
                segments=segments,
                feature=feature,
                image_size=image_size,
                frame_stride=frame_stride,
            )
            for channel_offset, path, segments, feature in tasks
        }
        while pending:
            completed, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            with progress_counter.get_lock():
                completed_frames = int(progress_counter.value)
            progress.update(completed_frames - int(progress.n))
            for future in completed:
                path, _ = future.result()
                video_paths.add(path)
        with progress_counter.get_lock():
            completed_frames = int(progress_counter.value)
        progress.update(completed_frames - int(progress.n))
    if completed_frames != expected:
        raise ValueError(f"exported {completed_frames} camera frames; expected {expected}")
    return video_paths


def _write_stored_npz(path: Path, arrays_dir: Path, names: tuple[str, ...]) -> None:
    """Package existing NPY files without making a second in-memory image copy."""
    array_paths = [arrays_dir / f"{name}.npy" for name in names]
    total_bytes = sum(array_path.stat().st_size for array_path in array_paths)
    with (
        tqdm(
            total=total_bytes,
            desc="Package train.npz",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress,
        zipfile.ZipFile(path, mode="w", allowZip64=True) as archive,
    ):
        for name, array_path in zip(names, array_paths, strict=True):
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = 0o600 << 16
            with array_path.open("rb") as source:
                with archive.open(entry, mode="w", force_zip64=True) as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
                        progress.update(len(chunk))


def _sha256(path: Path, progress: tqdm | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
            if progress is not None:
                progress.update(len(chunk))
    return digest.hexdigest()


def _source_fingerprint(dataset_root: Path, paths: set[Path]) -> tuple[str, dict[str, str]]:
    sorted_paths = sorted(paths)
    total_bytes = sum(path.stat().st_size for path in sorted_paths)
    with tqdm(
        total=total_bytes,
        desc="Fingerprint source",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as progress:
        file_hashes = {
            path.relative_to(dataset_root).as_posix(): _sha256(path, progress)
            for path in sorted_paths
        }
    digest = hashlib.sha256()
    for relative_path, file_hash in file_hashes.items():
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
    return digest.hexdigest(), file_hashes


def _supplied_bounds(
    bounds_from: Path | None,
    action_encoding: ActionEncoding,
) -> dict[str, np.ndarray] | None:
    """Read the normalization bounds of an earlier export, to reuse them here.

    Refuses an export whose action encoding differs: absolute and delta bounds
    describe different quantities, so reusing one for the other would squash
    joint commands into a scale fitted to one tick's motion.
    """
    if bounds_from is None:
        return None
    path = Path(bounds_from)
    if path.is_dir():
        path = path / "normalization.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = ("obs_min", "obs_max", "action_min", "action_max")
        missing = [name for name in names if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing normalization bounds: {missing}")
        earlier = archive[ACTION_ENCODING_KEY] if ACTION_ENCODING_KEY in archive else None
        if earlier is not None and str(earlier) != action_encoding.value:
            raise ValueError(
                f"{path} was exported with action encoding {str(earlier)!r}, "
                f"but this export uses {action_encoding.value!r}; its action bounds "
                "describe a different quantity"
            )
        return {name: np.asarray(archive[name], dtype=np.float32) for name in names}


def export_diffusion_policy_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    image_size: int = 96,
    policy_hz: int = DEFAULT_POLICY_HZ,
    max_episodes: int | None = None,
    workers: int = 1,
    action_encoding: ActionEncoding = ActionEncoding.ABSOLUTE,
    bounds_from: Path | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """Export Diffusion Policy arrays without modifying the LeRobot source.

    The source is recorded at the rig's control rate, but the policy runs at
    ``policy_hz``; every episode is decimated to episode-relative indices
    ``0, stride, 2 * stride, ...`` so the action chunks a model learns span the
    same wall-clock time the controller replays them over.

    ``action_encoding`` chooses what the policy is asked to predict: the joint
    command itself, or its offset from the joints measured on the same tick.
    Deltas are fitted their own min-max bounds, which is the point of them --
    a normalized unit then spans one tick's motion rather than a joint's whole
    range. The choice is recorded in both the manifest and the normalization
    archive, because every rollout path has to decode what was encoded here.
    """
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    building_dir = output_dir.with_name(f"{output_dir.name}.building")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if building_dir.exists():
        raise FileExistsError(f"incomplete export already exists: {building_dir}")
    if image_size < 8 or image_size % 8:
        raise ValueError("image_size must be a positive multiple of 8")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if goal is not None and goal != GOAL_TARGET_XY:
        raise ValueError(f"unknown goal source {goal!r}; expected {GOAL_TARGET_XY!r} or None")

    info_path = dataset_root / "meta" / "info.json"
    info = _read_json(info_path)
    features = info.get("features", {})
    required_features = {STATE_FEATURE, ACTION_FEATURE, *CAMERA_FEATURES}
    if not required_features <= set(features):
        raise ValueError(
            f"dataset is missing features: {sorted(required_features - set(features))}"
        )
    source_fps = int(info.get("fps", 0))
    if source_fps <= 0:
        raise ValueError("dataset fps must be positive")
    # The resolution the videos were written at, which every camera shares. A
    # live rollout has to downsample through it to land on this export's images.
    video_shapes = {tuple(features[feature]["shape"][:2]) for feature in CAMERA_FEATURES}
    if len(video_shapes) != 1:
        raise ValueError(f"dataset cameras must share one resolution, got {video_shapes}")
    source_video_hw = [int(value) for value in next(iter(video_shapes))]
    if policy_hz < 1:
        raise ValueError("policy_hz must be positive")
    if source_fps % policy_hz:
        raise ValueError(
            f"source fps {source_fps} is not an integer multiple of policy_hz {policy_hz}"
        )
    frame_stride = source_fps // policy_hz

    rows = _load_episode_rows(dataset_root)
    if max_episodes is not None:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("no episodes selected")

    source_lengths = [int(row["length"]) for row in rows]
    source_frames = sum(source_lengths)
    traj_lengths = np.asarray(
        [decimated_length(length, frame_stride) for length in source_lengths], dtype=np.int64
    )
    total_frames = int(traj_lengths.sum())
    states_raw, actions_raw = _load_low_dimensional_arrays(dataset_root, info, rows)
    if len(states_raw) != source_frames or len(actions_raw) != source_frames:
        raise ValueError("low-dimensional arrays do not match trajectory lengths")
    keep = _decimated_indices(source_lengths, frame_stride)
    if len(keep) != total_frames:
        raise ValueError("decimated index count does not match trajectory lengths")
    joints_raw = states_raw[keep]
    actions_raw = actions_raw[keep]
    goal_dim = 0
    observation_raw = joints_raw
    if goal is not None:
        goals = episode_goal_vectors(rows, source_lengths)[keep]
        _check_goal_varies(goals)
        goal_dim = int(goals.shape[1])
        # Appended, never prepended: the leading columns stay the joints an
        # older export put there, so a state vector remains readable by index.
        observation_raw = np.hstack((joints_raw, goals))
    supplied = _supplied_bounds(bounds_from, action_encoding)
    states, obs_min, obs_max = normalize_min_max(
        observation_raw, None if supplied is None else (supplied["obs_min"], supplied["obs_max"])
    )
    # Deltas are defined against the measured *joints*, so the goal columns must
    # not reach this: a command minus a target point is not a quantity.
    actions, action_min, action_max = normalize_min_max(
        encode_actions(action_encoding, actions_raw, joints_raw),
        None if supplied is None else (supplied["action_min"], supplied["action_max"]),
    )

    building_dir.mkdir(parents=True)
    arrays_dir = building_dir / "arrays"
    arrays_dir.mkdir()
    np.save(arrays_dir / "states.npy", states, allow_pickle=False)
    np.save(arrays_dir / "actions.npy", actions, allow_pickle=False)
    np.save(arrays_dir / "traj_lengths.npy", traj_lengths, allow_pickle=False)
    images_path = arrays_dir / "images.npy"
    images = open_memmap(
        images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, 3 * len(CAMERA_FEATURES), image_size, image_size),
    )
    images.flush()
    del images
    video_paths = _write_all_camera_images(
        images_path,
        dataset_root=dataset_root,
        info=info,
        rows=rows,
        image_size=image_size,
        frame_stride=frame_stride,
        workers=workers,
    )

    _write_stored_npz(
        building_dir / "train.npz",
        arrays_dir,
        ("states", "actions", "images", "traj_lengths"),
    )
    np.savez_compressed(
        building_dir / "normalization.npz",
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
        # Beside the bounds rather than only in the manifest: the bounds are
        # what every rollout path loads, and reading the encoding from the same
        # file is what stops a delta checkpoint being commanded as absolute
        # joint positions.
        **{ACTION_ENCODING_KEY: action_encoding.value},
    )
    shutil.rmtree(arrays_dir)

    metadata_paths = set((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    data_paths = {_data_path(dataset_root, info, row) for row in rows}
    source_hash, source_files = _source_fingerprint(
        dataset_root,
        {info_path, *metadata_paths, *data_paths, *video_paths},
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_dataset": str(dataset_root),
        "source_sha256": source_hash,
        "source_files": source_files,
        "episode_indices": [int(row["episode_index"]) for row in rows],
        "num_episodes": len(rows),
        "num_frames": total_frames,
        "fps": policy_hz,
        "source_fps": source_fps,
        "frame_stride": frame_stride,
        "state_feature": STATE_FEATURE,
        "action_feature": ACTION_FEATURE,
        "camera_features": list(CAMERA_FEATURES),
        "image_layout": "NCHW; RGB cameras concatenated in camera_features order",
        "image_dtype": "uint8",
        "image_size": [image_size, image_size],
        "source_video_hw": source_video_hw,
        "image_transform": "aspect-fill resize followed by center crop",
        "state_action_normalization": "per-dimension min-max to [-1, 1]",
        ACTION_ENCODING_KEY: action_encoding.value,
        "action_semantics": (
            "absolute joint command"
            if action_encoding is ActionEncoding.ABSOLUTE
            else "joint command minus the joints measured on the same control tick"
        ),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "goal_dim": goal_dim,
        "goal_source": goal,
    }
    with (building_dir / "export.json").open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    building_dir.rename(output_dir)
    return manifest
