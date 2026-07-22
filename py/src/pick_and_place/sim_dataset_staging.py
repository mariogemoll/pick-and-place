# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Manage finalized per-episode simulation datasets before aggregation."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pick_and_place.dataset_subset import (
    SUCCESS_XY_TOLERANCE_M,
    load_all_episodes,
    successful_episode_mask,
)


_EPISODE_DIRECTORY = re.compile(r"ep(\d+)")
COLLECTION_CONFIG_FILENAME = "collection.json"


def episode_staging_root(dataset_root: Path) -> Path:
    """Return the sibling directory holding individually finalized episodes."""
    return dataset_root.with_name(f"{dataset_root.name}_episodes")


def episode_index(path: Path) -> int:
    """Return the global episode index encoded in an ``epNNNNNN`` directory."""
    match = _EPISODE_DIRECTORY.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid staged episode directory name: {path.name}")
    return int(match.group(1))


def staged_episode_dirs(episodes_root: Path) -> list[Path]:
    """Return all staged episode directories, including incomplete ones."""
    if not episodes_root.exists():
        return []
    return sorted(
        (
            path
            for path in episodes_root.iterdir()
            if path.is_dir() and _EPISODE_DIRECTORY.fullmatch(path.name)
        ),
        key=episode_index,
    )


def find_episode_datasets(episodes_root: Path) -> list[Path]:
    """Return complete staged episode datasets in global-index order."""
    return [
        path
        for path in staged_episode_dirs(episodes_root)
        if (path / "meta" / "info.json").is_file()
    ]


def next_episode_index(episodes_root: Path) -> int:
    """Return an unused global index after every complete or partial episode."""
    staged = staged_episode_dirs(episodes_root)
    return episode_index(staged[-1]) + 1 if staged else 0


def ensure_collection_config(episodes_root: Path, config: dict[str, Any]) -> None:
    """Create or validate the immutable configuration shared by top-up runs."""
    normalized = json.loads(json.dumps(config, sort_keys=True))
    path = episodes_root / COLLECTION_CONFIG_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != normalized:
            differing = sorted(
                key
                for key in existing.keys() | normalized.keys()
                if existing.get(key) != normalized.get(key)
            )
            raise ValueError(
                "top-up configuration differs from the staged collection in: "
                + ", ".join(differing)
            )
        return
    if staged_episode_dirs(episodes_root):
        raise ValueError(
            f"{episodes_root} contains episodes but no {COLLECTION_CONFIG_FILENAME}; "
            "use a new dataset root"
        )
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")


def successful_episode_datasets(
    episode_roots: list[Path],
    xy_tolerance: float = SUCCESS_XY_TOLERANCE_M,
) -> list[Path]:
    """Return staged episodes whose final placement passes the success test."""
    successful: list[Path] = []
    for root in episode_roots:
        episodes = load_all_episodes(root)
        if len(episodes) != 1:
            raise ValueError(f"expected one episode in {root}, found {len(episodes)}")
        if bool(successful_episode_mask(episodes, xy_tolerance).iloc[0]):
            successful.append(root)
    return successful


def merge_episodes(
    episode_roots: list[Path],
    *,
    output_root: Path,
    output_repo_id: str,
    keep_episodes: bool,
) -> None:
    """Losslessly aggregate staged episodes into one training dataset."""
    from lerobot.datasets.aggregate import aggregate_datasets

    if not episode_roots:
        raise ValueError("cannot merge an empty episode selection")

    print(f"Merging {len(episode_roots)} episode(s) into {output_root}...")
    repo_ids = [f"{output_repo_id}-{root.name}" for root in episode_roots]
    aggregate_datasets(
        # LeRobot shows progress while copying, but loading every source's
        # metadata happens first and can take several minutes for large staged
        # collections. It consumes repo_ids and roots together, so wrapping one
        # side of that zip reports each completed metadata load.
        repo_ids=tqdm(repo_ids, desc="Load metadata", unit="episode", dynamic_ncols=True),
        aggr_repo_id=output_repo_id,
        roots=episode_roots,
        aggr_root=output_root,
    )
    print(f"Merged dataset -> {output_root}")
    if not keep_episodes:
        for root in episode_roots:
            shutil.rmtree(root)
        print(f"Removed {len(episode_roots)} merged episode dir(s).")
