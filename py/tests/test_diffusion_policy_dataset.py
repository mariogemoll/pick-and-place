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
from pick_and_place.spec.action_encoding import ActionEncoding, read_action_encoding


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
        assert read_action_encoding(normalization) is ActionEncoding.ABSOLUTE
    assert manifest["action_encoding"] == "absolute"
    assert manifest["camera_features"] == list(CAMERA_FEATURES)
    assert manifest["fps"] == 10
    assert manifest["source_fps"] == 30
    assert manifest["frame_stride"] == 3
    assert json.loads((output / "export.json").read_text()) == manifest
    assert not output.with_name("diffusion-policy.building").exists()

    export_diffusion_policy_dataset(source, second_output, image_size=8)
    assert (second_output / "train.npz").read_bytes() == (output / "train.npz").read_bytes()


def test_delta_export_fits_its_own_bounds_and_declares_the_encoding(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "diffusion-policy-delta"
    _write_tiny_dataset(source)
    monkeypatch.setattr(
        diffusion_policy_dataset,
        "_write_camera_images",
        lambda destination, *, channel_offset, **kwargs: [
            source
            / "videos"
            / CAMERA_FEATURES[channel_offset // 3]
            / "chunk-000"
            / "file-000.mp4"
        ],
    )

    manifest = export_diffusion_policy_dataset(
        source, output, image_size=8, action_encoding=ActionEncoding.DELTA
    )

    # The two kept frames are (state, action) = ([0, 10], [-2, 1]) and
    # ([2, 14], [2, 5]), so the deltas are [-2, -9] and [0, -9].
    with np.load(output / "normalization.npz", allow_pickle=False) as normalization:
        np.testing.assert_array_equal(normalization["action_min"], [-2.0, -9.0])
        np.testing.assert_array_equal(normalization["action_max"], [0.0, -9.0])
        # The states are untouched: only what the policy predicts changes.
        np.testing.assert_array_equal(normalization["obs_min"], [0.0, 10.0])
        np.testing.assert_array_equal(normalization["obs_max"], [2.0, 14.0])
        assert read_action_encoding(normalization) is ActionEncoding.DELTA
    with np.load(output / "train.npz", allow_pickle=False) as dataset:
        # The whole point: a delta spans one tick's motion, so the normalized
        # range is filled by that rather than by a joint's whole travel.
        np.testing.assert_allclose(dataset["actions"], [[-1.0, -1.0], [1.0, -1.0]], atol=2e-6)
    assert manifest["action_encoding"] == "delta"
    assert "measured on the same control tick" in manifest["action_semantics"]


def test_export_rejects_nonpositive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="workers must be positive"):
        export_diffusion_policy_dataset(tmp_path / "source", tmp_path / "output", workers=0)


def test_export_rejects_policy_rate_that_does_not_divide_source_fps(tmp_path):
    source = tmp_path / "source"
    _write_tiny_dataset(source)

    with pytest.raises(ValueError, match="not an integer multiple"):
        export_diffusion_policy_dataset(source, tmp_path / "output", policy_hz=11)


def test_supplied_bounds_are_used_instead_of_refitting():
    """Continuing training on a fresh export must not rescale the weights' world.

    A checkpoint learned what a normalized unit means under the bounds its own
    export fitted. Re-fitting on the fine-tune data moves the input and action
    scales, so what should measure adaptation partly measures recovery from a
    rescaling.
    """
    from pick_and_place.data.diffusion_policy_dataset import normalize_min_max

    values = np.array([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    fitted, low, high = normalize_min_max(values)
    np.testing.assert_allclose(low, [0.0, -1.0])
    np.testing.assert_allclose(high, [1.0, 1.0])
    np.testing.assert_allclose(fitted, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-5)

    wider = (np.array([-1.0, -2.0], np.float32), np.array([3.0, 2.0], np.float32))
    reused, low, high = normalize_min_max(values, wider)
    np.testing.assert_allclose(low, wider[0])
    np.testing.assert_allclose(high, wider[1])
    # Same data, someone else's scale: no longer spanning [-1, 1].
    np.testing.assert_allclose(reused, [[-0.5, -0.5], [0.0, 0.5]], atol=1e-5)


def test_supplied_bounds_must_match_the_column_count():
    from pick_and_place.data.diffusion_policy_dataset import normalize_min_max

    values = np.zeros((4, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="supplied bounds have shape"):
        normalize_min_max(values, (np.zeros(3, np.float32), np.ones(3, np.float32)))


def test_reusing_bounds_across_action_encodings_is_refused(tmp_path):
    """Absolute and delta bounds describe different quantities."""
    from pick_and_place.data.diffusion_policy_dataset import (
        ACTION_ENCODING_KEY,
        ActionEncoding,
        _supplied_bounds,
    )

    path = tmp_path / "normalization.npz"
    np.savez(
        path,
        obs_min=np.zeros(6, np.float32),
        obs_max=np.ones(6, np.float32),
        action_min=np.zeros(6, np.float32),
        action_max=np.ones(6, np.float32),
        **{ACTION_ENCODING_KEY: ActionEncoding.DELTA.value},
    )
    with pytest.raises(ValueError, match="different quantity"):
        _supplied_bounds(path, ActionEncoding.ABSOLUTE)
    assert _supplied_bounds(path, ActionEncoding.DELTA) is not None
