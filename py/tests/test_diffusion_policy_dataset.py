# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pick_and_place.data import diffusion_policy_dataset
from pick_and_place.data.diffusion_policy_dataset import (
    CAMERA_FEATURES,
    _decimated_indices,
    decimated_length,
    export_diffusion_policy_dataset,
    normalize_min_max,
)


def test_normalize_min_max_uses_policy_range_and_preserves_constant_columns():
    values = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)

    normalized, minimum, maximum = normalize_min_max(values)

    np.testing.assert_allclose(normalized[:, 0], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(normalized[:, 1], [-1.0, -1.0], atol=1e-6)
    np.testing.assert_array_equal(minimum, [1.0, 5.0])
    np.testing.assert_array_equal(maximum, [3.0, 5.0])


def test_decimation_restarts_at_each_episode_boundary():
    assert decimated_length(5, 3) == 2
    np.testing.assert_array_equal(_decimated_indices([5, 4], 3), [0, 3, 5, 8])


def _write_tiny_dataset(root: Path) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    for feature in CAMERA_FEATURES:
        video_path = root / "videos" / feature / "chunk-000" / "file-000.mp4"
        video_path.parent.mkdir(parents=True)
        video_path.touch()

    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"shape": [2]},
            "action": {"shape": [2]},
            **{feature: {"shape": [4, 6, 3]} for feature in CAMERA_FEATURES},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    episode = {
        "episode_index": [0],
        "length": [4],
        "data/chunk_index": [0],
        "data/file_index": [0],
    }
    for feature in CAMERA_FEATURES:
        episode.update(
            {
                f"videos/{feature}/chunk_index": [0],
                f"videos/{feature}/file_index": [0],
                f"videos/{feature}/from_timestamp": [0.0],
            }
        )
    pq.write_table(
        pa.table(episode),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "index": [0, 1, 2, 3],
                "episode_index": [0, 0, 0, 0],
                "observation.state": [
                    [0.0, 10.0],
                    [100.0, 100.0],
                    [200.0, 200.0],
                    [2.0, 14.0],
                ],
                "action": [[-2.0, 1.0], [100.0, 100.0], [200.0, 200.0], [2.0, 5.0]],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )


def test_export_writes_policy_arrays_normalization_and_camera_order(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "diffusion-policy"
    second_output = tmp_path / "diffusion-policy-second"
    _write_tiny_dataset(source)

    def fake_write_images(destination, *, channel_offset, rows, frame_stride, **kwargs):
        del kwargs
        value = 20 if channel_offset == 0 else 40
        destination[:, channel_offset : channel_offset + 3] = value
        assert sum(decimated_length(int(row["length"]), frame_stride) for row in rows) == len(
            destination
        )
        feature = CAMERA_FEATURES[channel_offset // 3]
        return [source / "videos" / feature / "chunk-000" / "file-000.mp4"]

    monkeypatch.setattr(diffusion_policy_dataset, "_write_camera_images", fake_write_images)

    manifest = export_diffusion_policy_dataset(source, output, image_size=8)

    with np.load(output / "train.npz", allow_pickle=False) as dataset:
        assert set(dataset.files) == {"states", "actions", "images", "traj_lengths"}
        assert dataset["images"].shape == (2, 6, 8, 8)
        np.testing.assert_array_equal(dataset["images"][:, :3], 20)
        np.testing.assert_array_equal(dataset["images"][:, 3:], 40)
        np.testing.assert_allclose(dataset["states"], [[-1.0, -1.0], [1.0, 1.0]], atol=2e-6)
        np.testing.assert_allclose(dataset["actions"], [[-1.0, -1.0], [1.0, 1.0]], atol=2e-6)
        np.testing.assert_array_equal(dataset["traj_lengths"], [2])
    with np.load(output / "normalization.npz", allow_pickle=False) as normalization:
        np.testing.assert_array_equal(normalization["obs_min"], [0.0, 10.0])
        np.testing.assert_array_equal(normalization["obs_max"], [2.0, 14.0])
        np.testing.assert_array_equal(normalization["action_min"], [-2.0, 1.0])
        np.testing.assert_array_equal(normalization["action_max"], [2.0, 5.0])
    assert manifest["camera_features"] == list(CAMERA_FEATURES)
    assert manifest["fps"] == 10
    assert manifest["source_fps"] == 30
    assert manifest["frame_stride"] == 3
    assert json.loads((output / "export.json").read_text()) == manifest
    assert not output.with_name("diffusion-policy.building").exists()

    export_diffusion_policy_dataset(source, second_output, image_size=8)
    assert (second_output / "train.npz").read_bytes() == (output / "train.npz").read_bytes()


def test_export_rejects_nonpositive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="workers must be positive"):
        export_diffusion_policy_dataset(tmp_path / "source", tmp_path / "output", workers=0)


def test_export_rejects_policy_rate_that_does_not_divide_source_fps(tmp_path):
    source = tmp_path / "source"
    _write_tiny_dataset(source)

    with pytest.raises(ValueError, match="not an integer multiple"):
        export_diffusion_policy_dataset(source, tmp_path / "output", policy_hz=11)
