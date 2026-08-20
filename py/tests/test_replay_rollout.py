# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The PPRL bytes have a reader in TypeScript, so the layout is a contract."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from pick_and_place.runtime.replay_rollout import write_rollout

HEADER_BYTES = 4 + 4 * 4 + 2 * 4


def _read(path):
    """Unpack a PPRL file the way ``episode-replay/rollout.ts`` does."""
    blob = path.read_bytes()
    assert blob[:4] == b"PPRL"
    version, fps, nframes, nq = struct.unpack_from("<IIII", blob, 4)
    target_x, target_y = struct.unpack_from("<ff", blob, 20)
    qpos = np.frombuffer(blob, dtype="<f4", offset=HEADER_BYTES, count=nframes * nq)
    return version, fps, nframes, nq, (target_x, target_y), qpos.reshape(nframes, nq)


def test_round_trips_through_the_typescript_layout(tmp_path):
    qpos = np.arange(5 * 13, dtype=np.float32).reshape(5, 13)
    path = tmp_path / "episode.bin"
    write_rollout(path, qpos, 10.0, (0.21, -0.13))

    version, fps, nframes, nq, target, frames = _read(path)
    assert (version, fps, nframes, nq) == (1, 10, 5, 13)
    assert target == pytest.approx((0.21, -0.13), abs=1e-6)
    np.testing.assert_array_equal(frames, qpos)
    assert path.stat().st_size == HEADER_BYTES + 5 * 13 * 4


def test_fps_is_rounded_to_the_integer_the_header_holds(tmp_path):
    path = tmp_path / "episode.bin"
    write_rollout(path, np.zeros((2, 13), dtype=np.float32), 29.6, (0.0, 0.0))
    assert _read(path)[1] == 30


def test_creates_the_directory_it_is_handed(tmp_path):
    path = tmp_path / "nested" / "episode.bin"
    write_rollout(path, np.zeros((1, 13), dtype=np.float32), 10.0, (0.0, 0.0))
    assert path.exists()


def test_refuses_frames_that_are_not_a_matrix(tmp_path):
    with pytest.raises(ValueError, match="nframes, nq"):
        write_rollout(tmp_path / "e.bin", np.zeros(13, dtype=np.float32), 10.0, (0.0, 0.0))
