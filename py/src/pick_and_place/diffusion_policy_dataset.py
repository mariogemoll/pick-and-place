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

from pick_and_place.sim_recorder import resize_and_center_crop

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
CAMERA_FEATURES = (
    "observation.images.overhead",
    "observation.images.wrist",
)
FORMAT_VERSION = "diffusion-policy-stitched-v1"
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


def normalize_min_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize columns to the policy's ``[-1, 1]`` convention."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected a non-empty rank-2 array, got {values.shape}")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    normalized = 2.0 * (values - minimum) / (maximum - minimum + 1e-6) - 1.0
    return normalized.astype(np.float32), minimum, maximum


def _video_segments(
    dataset_root: Path,
    info: dict[str, Any],
    rows: list[dict[str, Any]],
    feature: str,
) -> dict[Path, list[tuple[int, int, int]]]:
    fps = float(info["fps"])
    output_start = 0
    segments: dict[Path, list[tuple[int, int, int]]] = defaultdict(list)
    for row in rows:
        length = int(row["length"])
        start_frame = round(float(row[f"videos/{feature}/from_timestamp"]) * fps)
        path = _video_path(dataset_root, info, row, feature)
        segments[path].append((start_frame, length, output_start))
        output_start += length
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
    progress_position: int = 0,
) -> list[Path]:
    segments_by_path = _video_segments(dataset_root, info, rows, feature)
    expected = sum(int(row["length"]) for row in rows)
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
    report_progress,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    segments.sort()
    expected = sum(length for _, length, _ in segments)
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
            if offset >= length:
                continue
            image = frame.to_ndarray(format="rgb24")
            image = resize_and_center_crop(image, image_size, image_size)
            destination[output_start + offset, channel_offset : channel_offset + 3] = np.moveaxis(
                image, -1, 0
            )
            written += 1
            unreported += 1
            if unreported == _CAMERA_PROGRESS_BATCH:
                report_progress(unreported)
                unreported = 0
    if unreported:
        report_progress(unreported)
    if written != expected:
        raise ValueError(f"decoded {written} selected {feature} frames from {path}; expected {expected}")
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
                    )
                )
            return video_paths
        finally:
            images.flush()
            del images

    tasks = [
        (3 * camera_index, path, segments, feature)
        for camera_index, feature in enumerate(CAMERA_FEATURES)
        for path, segments in _video_segments(dataset_root, info, rows, feature).items()
    ]
    expected = sum(length for _, _, segments, _ in tasks for _, length, _ in segments)
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


def export_diffusion_policy_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    image_size: int = 96,
    max_episodes: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Export Diffusion Policy arrays without modifying the LeRobot source."""
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

    info_path = dataset_root / "meta" / "info.json"
    info = _read_json(info_path)
    features = info.get("features", {})
    required_features = {STATE_FEATURE, ACTION_FEATURE, *CAMERA_FEATURES}
    if not required_features <= set(features):
        raise ValueError(
            f"dataset is missing features: {sorted(required_features - set(features))}"
        )
    if int(info.get("fps", 0)) <= 0:
        raise ValueError("dataset fps must be positive")

    rows = _load_episode_rows(dataset_root)
    if max_episodes is not None:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("no episodes selected")

    traj_lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    total_frames = int(traj_lengths.sum())
    states_raw, actions_raw = _load_low_dimensional_arrays(dataset_root, info, rows)
    if len(states_raw) != total_frames or len(actions_raw) != total_frames:
        raise ValueError("low-dimensional arrays do not match trajectory lengths")
    states, obs_min, obs_max = normalize_min_max(states_raw)
    actions, action_min, action_max = normalize_min_max(actions_raw)

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
        "fps": int(info["fps"]),
        "state_feature": STATE_FEATURE,
        "action_feature": ACTION_FEATURE,
        "camera_features": list(CAMERA_FEATURES),
        "image_layout": "NCHW; RGB cameras concatenated in camera_features order",
        "image_dtype": "uint8",
        "image_size": [image_size, image_size],
        "image_transform": "aspect-fill resize followed by center crop",
        "state_action_normalization": "per-dimension min-max to [-1, 1]",
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
    }
    with (building_dir / "export.json").open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    building_dir.rename(output_dir)
    return manifest
