# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pick_and_place import dppo_imitation_audit
from pick_and_place.dppo_imitation_audit import (
    HORIZON_STEPS,
    HeldOutSplit,
    TrainingArtifactSplit,
    ExampleRef,
    StoredNpz,
    held_out_episode_names,
    sample_refs,
    summarize_errors,
)
from pick_and_place.follower import JOINT_NAMES


def test_held_out_episode_contract_has_100_successes() -> None:
    names = held_out_episode_names()

    assert len(names) == 100
    assert names[0] == "ep001017"
    assert names[-1] == "ep001117"
    assert "ep001110" not in names


def test_sample_refs_is_deterministic_and_without_replacement() -> None:
    refs = [ExampleRef("0", index) for index in range(20)]

    first = sample_refs(refs, 5, seed=42)
    second = sample_refs(refs, 5, seed=42)

    assert first == second
    assert len(set(first)) == 5
    with pytest.raises(ValueError, match="only 20"):
        sample_refs(refs, 21, seed=42)


def test_stored_npz_maps_members_without_materializing_archive(tmp_path: Path) -> None:
    path = tmp_path / "train.npz"
    arrays = {
        "states": np.arange(24, dtype=np.float32).reshape(4, 6),
        "images": np.arange(4 * 6 * 8 * 8, dtype=np.uint8).reshape(4, 6, 8, 8),
    }
    np.savez(path, **arrays)

    archive = StoredNpz(path)

    for name, expected in arrays.items():
        actual = archive.array(name)
        assert isinstance(actual, np.memmap)
        np.testing.assert_array_equal(actual, expected)


def test_training_artifact_uses_t_minus_1_t_history_and_actions_from_t(tmp_path: Path) -> None:
    length = HORIZON_STEPS + 2
    states = np.repeat(np.arange(length, dtype=np.float32)[:, None], 6, axis=1)
    actions = states + 20
    images = np.zeros((length, 6, 96, 96), dtype=np.uint8)
    images[:, :3] = np.arange(length, dtype=np.uint8)[:, None, None, None]
    images[:, 3:] = 100 + np.arange(length, dtype=np.uint8)[:, None, None, None]
    dataset_path = tmp_path / "train.npz"
    normalization_path = tmp_path / "normalization.npz"
    np.savez(
        dataset_path,
        states=states,
        actions=actions,
        images=images,
        traj_lengths=np.asarray([length], dtype=np.int64),
    )
    np.savez(
        normalization_path,
        obs_min=np.zeros(6, dtype=np.float32),
        obs_max=np.ones(6, dtype=np.float32) * 2,
        action_min=np.zeros(6, dtype=np.float32),
        action_max=np.ones(6, dtype=np.float32) * 2,
    )

    split = TrainingArtifactSplit(dataset_path, normalization_path)
    example = split.load_examples([ExampleRef("0", 1)])[0]

    # The synthetic arrays are interpreted as normalized values. Their exact
    # physical value is unimportant; the selected temporal indices are not.
    assert example.observations[0]["observation.state"][0] == pytest.approx(1.0)
    assert example.observations[1]["observation.state"][0] == pytest.approx(2.0)
    assert example.target_actions[:, 0].tolist() == pytest.approx(
        [value + 21 for value in range(1, 1 + HORIZON_STEPS)]
    )
    np.testing.assert_array_equal(example.observations[0]["observation.images.overhead"], 0)
    np.testing.assert_array_equal(example.observations[1]["observation.images.overhead"], 1)
    np.testing.assert_array_equal(example.observations[0]["observation.images.wrist"], 100)
    np.testing.assert_array_equal(example.observations[1]["observation.images.wrist"], 101)


def test_held_out_split_applies_episode_relative_stride_to_every_signal(
    tmp_path: Path, monkeypatch
) -> None:
    episode_root = tmp_path / "ep001017"
    (episode_root / "meta/episodes/chunk-000").mkdir(parents=True)
    (episode_root / "data/chunk-000").mkdir(parents=True)
    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (episode_root / "meta/info.json").write_text(json.dumps(info))
    length = 3 * HORIZON_STEPS + 3
    row = {
        "episode_index": [0],
        "length": [length],
        "data/chunk_index": [0],
        "data/file_index": [0],
    }
    for feature in dppo_imitation_audit.CAMERA_FEATURES:
        row.update(
            {
                f"videos/{feature}/chunk_index": [0],
                f"videos/{feature}/file_index": [0],
                f"videos/{feature}/from_timestamp": [0.0],
            }
        )
        video_path = episode_root / "videos" / feature / "chunk-000/file-000.mp4"
        video_path.parent.mkdir(parents=True)
        video_path.touch()
    pq.write_table(pa.table(row), episode_root / "meta/episodes/chunk-000/file-000.parquet")
    source = np.arange(length, dtype=np.float32)
    pq.write_table(
        pa.table(
            {
                "frame_index": np.arange(length),
                "observation.state": np.repeat(source[:, None], 6, axis=1).tolist(),
                "action": np.repeat((100 + source)[:, None], 6, axis=1).tolist(),
            }
        ),
        episode_root / "data/chunk-000/file-000.parquet",
    )

    class FakeFrame:
        def __init__(self, value: int) -> None:
            self.value = value

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            return np.full((96, 96, 3), self.value, dtype=np.uint8)

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def decode(self, *, video: int):
            assert video == 0
            return (FakeFrame(index) for index in range(length))

    monkeypatch.setattr(dppo_imitation_audit.av, "open", lambda path: FakeContainer())

    split = HeldOutSplit(tmp_path, max_episodes=1)
    example = split.load_examples([ExampleRef("ep001017", 1)])[0]

    assert example.source_frame_indices == (0, 3)
    assert example.observations[0]["observation.state"][0] == 0
    assert example.observations[1]["observation.state"][0] == 3
    np.testing.assert_array_equal(example.observations[0]["observation.images.overhead"], 0)
    np.testing.assert_array_equal(example.observations[1]["observation.images.overhead"], 3)
    assert example.target_actions[:, 0].tolist() == list(100 + 3 * np.arange(1, 1 + HORIZON_STEPS))


def test_summarize_errors_keeps_splits_steps_and_distributions_explicit() -> None:
    targets = np.zeros((2, HORIZON_STEPS, len(JOINT_NAMES)), dtype=np.float32)
    predictions = np.ones_like(targets)
    predictions[1] *= 3

    summary = summarize_errors(predictions, targets)

    assert summary["num_examples"] == 2
    assert set(summary["per_step"]) == {str(step) for step in range(HORIZON_STEPS)}
    assert set(summary["selected_steps"]) == {"0", "1", "3", "7", "15"}
    shoulder = summary["selected_steps"]["0"]["joint_absolute_error"]["shoulder_pan"]
    assert shoulder == pytest.approx({"mean": 2.0, "median": 2.0, "p90": 2.8, "p95": 2.9})
    assert summary["all_steps"]["arm_vector_l2"]["mean"] == pytest.approx(2 * np.sqrt(5))
