# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


def load_texture_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_apriltag_textures.py"
    spec = importlib.util.spec_from_file_location("render_apriltag_textures", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workspace_frame_textures_are_reproducible():
    module = load_texture_script()
    texture_dir = Path(__file__).resolve().parents[2] / "assets" / "apriltags" / "textures"

    for tag_id in range(12, 16):
        expected = cv2.cvtColor(
            module.render_texture(tag_id, sticker_mm=60.0, tag_mm=40.0, px_per_cell=32),
            cv2.COLOR_RGB2BGR,
        )
        actual = cv2.imread(
            str(texture_dir / f"tagStandard41h12_{tag_id:05d}_60x60mm_tag40mm.png"),
            cv2.IMREAD_COLOR,
        )

        assert actual is not None
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
