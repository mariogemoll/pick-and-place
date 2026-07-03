# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Write a losslessly-reindexed subset of a LeRobotDataset's episodes.

Unlike ``lerobot.datasets.dataset_tools.delete_episodes``, ``write_subset_dataset``
never re-encodes video files. LeRobot video files can pack multiple episodes
together, so dropping a scattered subset of episodes (a success filter, a
train/val split, ...) can make video files mix kept and dropped episodes.
``delete_episodes`` handles that by decoding and re-encoding video to
physically remove unwanted frames, which is lossy for the frames that remain.
This module instead copies every referenced video file byte-for-byte and only
rewrites the *data* rows and episode metadata to point at (and iterate over)
the kept episodes. Frames belonging to excluded episodes stay physically
present in the copied video files but are never referenced by the subset
dataset's index, so nothing that reads it via ``LeRobotDataset`` ever sees
them.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Episode-metadata columns that are LeRobot bookkeeping (file layout, video
# spans, per-feature stats) rather than project data. Every other column is
# treated as project metadata (success, cube pose, pickup/placement checks,
# ...) and carried through to the subset dataset unchanged, mirroring
# convert_dataset_resolution.py's approach.
BOOKKEEPING_COLUMNS = {"episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"}
BOOKKEEPING_PREFIXES = ("data/", "videos/", "stats/", "meta/episodes/")


def load_all_episodes(root: Path) -> pd.DataFrame:
    return pd.concat(
        pd.read_parquet(p) for p in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    ).sort_values("episode_index")


def project_metadata_columns(episodes: pd.DataFrame) -> list[str]:
    return [
        c for c in episodes.columns if c not in BOOKKEEPING_COLUMNS and not c.startswith(BOOKKEEPING_PREFIXES)
    ]


def _copy_videos_unfiltered(src_dataset, dst_meta, episode_mapping: dict[int, int]) -> dict[int, dict]:
    """Video metadata for kept episodes, copying each referenced video file as-is.

    Every video file that at least one kept episode points into is copied
    once, whole, byte-for-byte. Kept episodes reuse their original
    chunk/file/timestamp pointers unchanged, since the video content at those
    coordinates hasn't moved. No decoding, filtering, or re-encoding happens.
    """
    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}
    for video_key in src_dataset.meta.video_keys:
        copied_files: set[tuple[int, int]] = set()
        for old_idx, new_idx in episode_mapping.items():
            src_ep = src_dataset.meta.episodes[old_idx]
            chunk_idx = src_ep[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep[f"videos/{video_key}/file_index"]
            file_key = (chunk_idx, file_idx)
            if file_key not in copied_files:
                src_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                dst_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, dst_path)
                copied_files.add(file_key)

            episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = file_idx
            episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                f"videos/{video_key}/from_timestamp"
            ]
            episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                f"videos/{video_key}/to_timestamp"
            ]
    return episodes_video_metadata


def _write_episodes_metadata(
    dataset,
    new_meta,
    episode_mapping: dict[int, int],
    data_metadata: dict[int, dict],
    video_metadata: dict[int, dict] | None,
    project_metadata_by_episode: dict[int, dict[str, Any]],
) -> None:
    """Like ``dataset_tools._copy_and_reindex_episodes_metadata``, but also carries
    through project-specific episode columns (success, cube pose, pickup/placement
    checks, ...) that the upstream helper doesn't know about and would otherwise drop."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.dataset_tools import _load_episode_with_stats
    from lerobot.datasets.io_utils import write_info, write_stats
    from lerobot.datasets.utils import flatten_dict

    all_stats = []
    total_frames = 0

    for old_idx, new_idx in sorted(episode_mapping.items(), key=lambda x: x[1]):
        src_episode_full = _load_episode_with_stats(dataset, old_idx)
        src_episode = dataset.meta.episodes[old_idx]

        episode_meta = data_metadata[new_idx].copy()
        if video_metadata and new_idx in video_metadata:
            episode_meta.update(video_metadata[new_idx])

        episode_stats: dict[str, dict] = {}
        for key in src_episode_full:
            if not key.startswith("stats/"):
                continue
            feature_name, stat_name = key.replace("stats/", "").split("/")
            episode_stats.setdefault(feature_name, {})

            value = src_episode_full[key]
            if dataset.meta.features.get(feature_name, {}).get("dtype") in ("image", "video"):
                if stat_name != "count" and isinstance(value, np.ndarray) and value.dtype == object:
                    flat_values = []
                    for item in value:
                        while isinstance(item, np.ndarray):
                            item = item.flatten()[0]
                        flat_values.append(item)
                    value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                elif stat_name != "count" and isinstance(value, np.ndarray) and value.shape == (3,):
                    value = value.reshape(3, 1, 1)
            episode_stats[feature_name][stat_name] = value
        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],
            "length": src_episode["length"],
        }
        episode_dict.update(episode_meta)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        episode_dict.update(project_metadata_by_episode[old_idx])
        new_meta._save_episode_metadata(episode_dict)

        total_frames += src_episode["length"]

    new_meta.finalize()
    new_meta.info.update(
        {
            "total_episodes": len(episode_mapping),
            "total_frames": total_frames,
            "total_tasks": len(new_meta.tasks) if new_meta.tasks is not None else 0,
            "splits": {"train": f"0:{len(episode_mapping)}"},
        }
    )
    write_info(new_meta.info, new_meta.root)

    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in new_meta.features}
    write_stats(filtered_stats, new_meta.root)


def write_subset_dataset(
    src_root: Path,
    dst_root: Path,
    repo_id: str,
    kept_indices: list[int],
    episodes: pd.DataFrame | None = None,
) -> None:
    """Write the episodes in ``kept_indices`` (source ``episode_index`` values, in the
    given order) from ``src_root`` to a new LeRobotDataset at ``dst_root``, without
    re-encoding any video. ``episodes`` may be passed in already loaded (from
    ``load_all_episodes``) to avoid re-reading it when the caller already has it."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import _copy_and_reindex_data
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episodes is None:
        episodes = load_all_episodes(src_root)

    dataset = LeRobotDataset(repo_id=f"local/{src_root.name}", root=src_root)
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}

    metadata_columns = project_metadata_columns(episodes)
    episodes = episodes.copy()
    # A batch of buffered episode rows that mixes real strings with pandas'
    # float NaN infers a pyarrow type that conflicts across batches (see the
    # same fix in convert_dataset_resolution.py); filling with "" keeps every
    # value a real str so the column always infers consistently.
    for col in metadata_columns:
        if pd.api.types.is_string_dtype(episodes[col]):
            episodes[col] = episodes[col].astype(object).where(episodes[col].notna(), "")
    project_metadata_by_episode = {
        int(row["episode_index"]): {col: row[col] for col in metadata_columns}
        for row in episodes.to_dict("records")
    }

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=f"local/{repo_id}",
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=dst_root,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    video_metadata = (
        _copy_videos_unfiltered(dataset, new_meta, episode_mapping) if dataset.meta.video_keys else None
    )
    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)
    _write_episodes_metadata(
        dataset, new_meta, episode_mapping, data_metadata, video_metadata, project_metadata_by_episode
    )
