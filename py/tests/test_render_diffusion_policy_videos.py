# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reading an export back as frames: the mapping, slicing and camera layout."""

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.data.stored_npz import episode_bounds, memmap_stored_npz

RENDER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_diffusion_policy_videos.py"


def _load_render_module():
    spec = importlib.util.spec_from_file_location("render_diffusion_policy_videos", RENDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_stored_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", mode="w") as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def test_memmap_stored_npz_returns_the_written_arrays(tmp_path):
    arrays = {
        "traj_lengths": np.array([2, 3], dtype=np.int64),
        "images": np.arange(5 * 6 * 4 * 4, dtype=np.uint8).reshape(5, 6, 4, 4),
    }
    _write_stored_npz(tmp_path / "train.npz", arrays)

    mapped = memmap_stored_npz(tmp_path / "train.npz")

    assert set(mapped) == {"traj_lengths", "images"}
    for name, array in arrays.items():
        np.testing.assert_array_equal(mapped[name], array)


def test_memmap_stored_npz_rejects_a_compressed_archive(tmp_path):
    path = tmp_path / "train.npz"
    np.savez_compressed(path, states=np.zeros(4, dtype=np.float32))

    with pytest.raises(ValueError, match="compressed"):
        memmap_stored_npz(path)


def test_episode_bounds_covers_every_frame_exactly_once():
    np.testing.assert_array_equal(episode_bounds(np.array([2, 3, 1])), [[0, 2], [2, 5], [5, 6]])


def test_split_cameras_lays_stacked_channels_out_left_to_right():
    module = _load_render_module()
    overhead = np.full((1, 3, 2, 2), 10, dtype=np.uint8)
    wrist = np.full((1, 3, 2, 2), 20, dtype=np.uint8)

    strip = module.split_cameras(np.concatenate([overhead, wrist], axis=1), 2)

    assert strip.shape == (1, 2, 4, 3)
    assert np.all(strip[0, :, :2] == 10)
    assert np.all(strip[0, :, 2:] == 20)


def test_split_cameras_keeps_pixels_together_across_channels():
    module = _load_render_module()
    frame = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(1, 3, 2, 2)

    strip = module.split_cameras(frame, 1)

    np.testing.assert_array_equal(strip[0, 1, 0], frame[0, :, 1, 0])


def test_split_cameras_rejects_a_channel_count_that_is_not_rgb_per_camera():
    module = _load_render_module()

    with pytest.raises(ValueError, match="channels"):
        module.split_cameras(np.zeros((1, 5, 2, 2), dtype=np.uint8), 2)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [(None, [0, 1, 2]), ("1", [1]), ("0-1,2", [0, 1, 2])],
)
def test_parse_episode_selection_expands_ranges(selection, expected):
    module = _load_render_module()

    assert module.parse_episode_selection(selection, 3) == expected


def test_parse_episode_selection_rejects_an_episode_that_does_not_exist():
    module = _load_render_module()

    with pytest.raises(ValueError, match="out of range"):
        module.parse_episode_selection("0-3", 3)
