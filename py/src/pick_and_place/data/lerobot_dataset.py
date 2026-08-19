# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reading the parts of a LeRobotDataset that several commands need.

A dataset keeps its shape in ``meta/info.json``, one row per episode across
``meta/episodes/chunk-*/file-*.parquet``, and the frames themselves in the data
files those rows point at. Reading that layout is not specific to any one
policy or export, so it lives here rather than being open-coded per command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pick_and_place.spec.controller import STATE_FEATURE


def read_info(dataset_root: Path) -> dict[str, Any]:
    """Read a dataset's ``meta/info.json``."""
    with (dataset_root / "meta" / "info.json").open() as file:
        return json.load(file)


def read_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """Read every episode-metadata row, ordered by ``episode_index``."""
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def read_episode_states(
    dataset_root: Path, info: dict[str, Any], row: dict[str, Any]
) -> np.ndarray:
    """Read one episode's recorded joint states, in the order they were recorded.

    The row names the data file its frames live in, which holds several
    episodes, so the file is filtered back down to this episode's index.
    """
    data_path = dataset_root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    table = pq.read_table(data_path, columns=["episode_index", STATE_FEATURE])
    table = table.filter(pc.equal(table["episode_index"], int(row["episode_index"])))
    return np.asarray(table[STATE_FEATURE].to_pylist(), dtype=np.float32)
