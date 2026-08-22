# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pick_and_place.cli.render_apriltag_textures import render_texture


def test_workspace_frame_textures_are_reproducible():
    texture_dir = Path(__file__).resolve().parents[2] / "assets" / "apriltags" / "textures"

    for tag_id in range(12, 16):
        expected = cv2.cvtColor(
            render_texture(tag_id, sticker_mm=60.0, tag_mm=40.0, px_per_cell=32),
            cv2.COLOR_RGB2BGR,
        )
        actual = cv2.imread(
            str(texture_dir / f"tagStandard41h12_{tag_id:05d}_60x60mm_tag40mm.png"),
            cv2.IMREAD_COLOR,
        )

        assert actual is not None
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
