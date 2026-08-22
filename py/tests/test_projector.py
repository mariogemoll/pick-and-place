# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pathlib import Path

import numpy as np
import pytest

from pick_and_place.hardware.projector import FramebufferInfo, pack_xrgb8888

INFO = FramebufferInfo(
    device=Path("/dev/fb0"), width=4, height=2, bits_per_pixel=32, stride=16
)


def test_pack_orders_bytes_as_bgrx():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)  # BGR

    packed = np.frombuffer(pack_xrgb8888(frame, INFO), dtype=np.uint8)

    # XRGB8888 little-endian lands in memory as B, G, R, X.
    assert list(packed[:4]) == [10, 20, 30, 255]


def test_pack_respects_a_stride_wider_than_the_row():
    padded = FramebufferInfo(
        device=Path("/dev/fb0"), width=4, height=2, bits_per_pixel=32, stride=24
    )
    frame = np.full((2, 4, 3), 7, dtype=np.uint8)

    packed = pack_xrgb8888(frame, padded)

    assert len(packed) == 24 * 2
    row = np.frombuffer(packed, dtype=np.uint8)[:24]
    assert list(row[16:]) == [0] * 8  # the pad is not pixel data


def test_pack_rejects_a_frame_of_the_wrong_size():
    """Silently scaling would corrupt a calibration while looking like it worked."""
    with pytest.raises(ValueError, match="Nothing here scales"):
        pack_xrgb8888(np.zeros((3, 4, 3), dtype=np.uint8), INFO)


def test_pack_rejects_a_non_uint8_frame():
    with pytest.raises(ValueError, match="uint8"):
        pack_xrgb8888(np.zeros((2, 4, 3), dtype=np.float32), INFO)


def test_pack_rejects_a_stride_shorter_than_one_row():
    narrow = FramebufferInfo(
        device=Path("/dev/fb0"), width=4, height=2, bits_per_pixel=32, stride=8
    )
    with pytest.raises(ValueError, match="shorter than one"):
        pack_xrgb8888(np.zeros((2, 4, 3), dtype=np.uint8), narrow)


def test_size_reports_width_then_height():
    assert INFO.size == (4, 2)
