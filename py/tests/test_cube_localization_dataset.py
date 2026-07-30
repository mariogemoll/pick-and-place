# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pick_and_place.cube_localization_dataset import episode_frame_split, load_cube_targets


def _write_staged_episode(
    root: Path,
    *,
    length: int,
    phase_spans: list[tuple[str, int]],
    cube_xyz_start: tuple[float, float, float],
) -> None:
    """A one-episode staged directory, matching the layout under a
    ``sim-200_episodes``-style prefix (one LeRobot episode per directory,
    before finalization)."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    episode = {
        "episode_index": [0],
        "length": [length],
        "phase_spans": [json.dumps(phase_spans)],
    }
    pq.write_table(
        pa.table(episode),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    # The cube drifts by one unit per frame on x, so decimated positions are
    # easy to check by hand; quaternion columns are constant and unused.
    x0, y0, z0 = cube_xyz_start
    environment_state = [
        [x0 + frame, y0, z0, 1.0, 0.0, 0.0, 0.0] for frame in range(length)
    ]
    pq.write_table(
        pa.table(
            {
                "index": list(range(length)),
                "episode_index": [0] * length,
                "observation.environment_state": environment_state,
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )


def test_load_cube_targets_decimates_and_concatenates_in_episode_order(tmp_path):
    episodes_root = tmp_path / "sim-200_episodes"
    _write_staged_episode(
        episodes_root / "ep000000",
        length=5,
        phase_spans=[("approach", 0), ("grasp", 3)],
        cube_xyz_start=(0.0, 1.0, 0.02),
    )
    _write_staged_episode(
        episodes_root / "ep000001",
        length=4,
        phase_spans=[("approach", 0), ("lift", 2)],
        cube_xyz_start=(10.0, 2.0, 0.03),
    )

    positions, phases, traj_lengths = load_cube_targets(
        episodes_root, ["ep000000", "ep000001"], frame_stride=2
    )

    # Episode 0 keeps frames 0, 2, 4 (ceil(5/2) = 3); episode 1 keeps 0, 2.
    np.testing.assert_array_equal(traj_lengths, [3, 2])
    np.testing.assert_allclose(
        positions,
        [
            [0.0, 1.0, 0.02],
            [2.0, 1.0, 0.02],
            [4.0, 1.0, 0.02],
            [10.0, 2.0, 0.03],
            [12.0, 2.0, 0.03],
        ],
    )
    # Episode 0's grasp starts at frame 3, so decimated frame 4 (kept index 2)
    # is "grasp". Episode 1's lift starts at frame 2, so its decimated frame 0
    # is still "acquisition" and decimated frame 2 is already "transport".
    assert list(phases) == ["acquisition", "acquisition", "grasp", "acquisition", "transport"]


def test_load_cube_targets_requires_matching_frame_count(tmp_path):
    episodes_root = tmp_path / "sim-200_episodes"
    root = episodes_root / "ep000000"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "length": [4],
                "phase_spans": [json.dumps([["approach", 0]])],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "index": [0, 1, 2],
                "episode_index": [0, 0, 0],
                "observation.environment_state": [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )

    with pytest.raises(ValueError, match="expected 4"):
        load_cube_targets(episodes_root, ["ep000000"], frame_stride=1)


def test_load_cube_targets_rejects_nonpositive_frame_stride(tmp_path):
    with pytest.raises(ValueError, match="frame_stride"):
        load_cube_targets(tmp_path, ["ep000000"], frame_stride=0)


def test_episode_frame_split_holds_out_whole_episodes():
    traj_lengths = np.array([5, 3, 4, 2, 6, 1, 7, 2, 3, 4])

    train_frames, held_out_frames = episode_frame_split(
        traj_lengths, held_out_fraction=0.3, seed=0
    )

    total_frames = int(traj_lengths.sum())
    assert len(train_frames) + len(held_out_frames) == total_frames
    assert set(train_frames.tolist()).isdisjoint(held_out_frames.tolist())
    # 30% of 10 episodes rounds to 3 held-out episodes.
    ends = np.cumsum(traj_lengths)
    starts = ends - traj_lengths
    held_out_episode_count = sum(
        1
        for start, end in zip(starts, ends)
        if any(start <= frame < end for frame in held_out_frames)
    )
    assert held_out_episode_count == 3


def test_episode_frame_split_is_deterministic_given_a_seed():
    traj_lengths = np.array([5, 3, 4, 2, 6])

    first = episode_frame_split(traj_lengths, held_out_fraction=0.4, seed=7)
    second = episode_frame_split(traj_lengths, held_out_fraction=0.4, seed=7)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_episode_frame_split_rejects_out_of_range_fraction():
    traj_lengths = np.array([5, 3])
    with pytest.raises(ValueError, match="held_out_fraction"):
        episode_frame_split(traj_lengths, held_out_fraction=0.0, seed=0)
    with pytest.raises(ValueError, match="held_out_fraction"):
        episode_frame_split(traj_lengths, held_out_fraction=1.0, seed=0)
