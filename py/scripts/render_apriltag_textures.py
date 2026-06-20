#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Render AprilTag PNG textures for the MuJoCo scene.

The output mirrors the physical sticker layout: a white square sticker with the
black/white tag centered at the requested tag/sticker size ratio. The workspace
frame uses 60 mm stickers with 40 mm tag graphics, matching the real plates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from generate_apriltags import TAG_41H12_BITS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "assets" / "apriltags" / "textures"


@dataclass(frozen=True)
class TextureSpec:
    name: str
    ids: tuple[int, ...]
    sticker_mm: float
    tag_mm: float


DEFAULT_SPECS = (
    TextureSpec("cube_30mm", tuple(range(0, 6)), 30.0, 20.0),
    TextureSpec("drop_box_100mm", tuple(range(8, 12)), 100.0, 60.0),
    TextureSpec("workspace_frame_60mm", tuple(range(12, 16)), 60.0, 40.0),
)


def parse_ids(value: str) -> tuple[int, ...]:
    """Parse ID ranges like ``0-5,8,12-15``."""
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(v) for v in part.split("-", 1))
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(part))
    return tuple(ids)


def tag_image(tag_id: int, px_per_cell: int) -> np.ndarray:
    """Return an upscaled RGB tag bitmap without sticker margin."""
    if tag_id not in TAG_41H12_BITS:
        raise ValueError(f"tagStandard41h12 id {tag_id} is not in the local table")
    grid = np.array(TAG_41H12_BITS[tag_id], dtype=np.uint8)
    cells = grid.shape[0]
    tag = np.full((cells, cells), 255, dtype=np.uint8)
    tag[grid == 1] = 0
    return np.kron(tag, np.ones((px_per_cell, px_per_cell), dtype=np.uint8))


def render_texture(
    tag_id: int,
    *,
    sticker_mm: float,
    tag_mm: float,
    px_per_cell: int,
) -> np.ndarray:
    """Render one sticker texture as an RGB image."""
    tag = tag_image(tag_id, px_per_cell)
    tag_px = tag.shape[0]
    margin_px = round(tag_px * (sticker_mm - tag_mm) / (2.0 * tag_mm))
    canvas_px = tag_px + 2 * margin_px
    canvas = np.full((canvas_px, canvas_px), 255, dtype=np.uint8)
    canvas[margin_px : margin_px + tag_px, margin_px : margin_px + tag_px] = tag
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def texture_stem(tag_id: int, sticker_mm: float, tag_mm: float) -> str:
    """Return the canonical texture filename stem."""
    return (
        f"tagStandard41h12_{tag_id:05d}_"
        f"{sticker_mm:g}x{sticker_mm:g}mm_tag{tag_mm:g}mm"
    )


def render_specs(specs: tuple[TextureSpec, ...], out_dir: Path, px_per_cell: int) -> None:
    """Render all textures for ``specs`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for tag_id in spec.ids:
            image = render_texture(
                tag_id,
                sticker_mm=spec.sticker_mm,
                tag_mm=spec.tag_mm,
                px_per_cell=px_per_cell,
            )
            output = out_dir / f"{texture_stem(tag_id, spec.sticker_mm, spec.tag_mm)}.png"
            cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"{spec.name}: wrote {len(spec.ids)} PNG textures to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--px-per-cell", type=int, default=32)
    parser.add_argument("--ids", type=parse_ids, default=None, help="IDs like 12-15 or 0-5,8-11.")
    parser.add_argument("--sticker-mm", type=float)
    parser.add_argument("--tag-mm", type=float)
    parser.add_argument(
        "--all-defaults",
        action="store_true",
        help="render cube, drop-box, and workspace-frame texture presets (the default)",
    )
    args = parser.parse_args()

    custom_requested = any(
        value is not None for value in (args.ids, args.sticker_mm, args.tag_mm)
    )
    if args.all_defaults or not custom_requested:
        specs = DEFAULT_SPECS
    else:
        ids = args.ids if args.ids is not None else tuple(range(12, 16))
        sticker_mm = args.sticker_mm if args.sticker_mm is not None else 60.0
        tag_mm = args.tag_mm if args.tag_mm is not None else 40.0
        specs = (TextureSpec("custom", ids, sticker_mm, tag_mm),)

    render_specs(specs, args.out_dir.resolve(), args.px_per_cell)


if __name__ == "__main__":
    main()
