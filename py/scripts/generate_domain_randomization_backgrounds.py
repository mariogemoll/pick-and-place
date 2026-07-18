#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate a deterministic preview bank of procedural sim appearances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    generate_procedural_appearance,
)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def _contact_sheet(images: list[np.ndarray], columns: int, width: int) -> np.ndarray:
    resized = []
    for image in images:
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        resized.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    cell_height = max(image.shape[0] for image in resized)
    rows = (len(resized) + columns - 1) // columns
    sheet = np.zeros((rows * cell_height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(resized):
        row, column = divmod(index, columns)
        sheet[row * cell_height : row * cell_height + image.shape[0], column * width : (column + 1) * width] = image
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "config"
        / "domain_randomization"
        / "act_mild_v1.json",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/procedural_appearances"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    if args.count < 1 or args.columns < 1:
        parser.error("--count and --columns must be positive")

    preset = DomainRandomizationPreset.load(args.preset)
    backgrounds = []
    tables = []
    manifest = []
    for index in range(args.count):
        episode_seed = int(
            np.random.default_rng(np.random.SeedSequence([args.seed, index])).integers(2**63)
        )
        sample = preset.sample(episode_seed)
        appearance = generate_procedural_appearance(sample)
        background_path = args.output / "backgrounds" / f"{index:03d}.png"
        table_path = args.output / "tables" / f"{index:03d}.png"
        _write_rgb(background_path, appearance.background_rgb)
        _write_rgb(table_path, appearance.table_rgb)
        backgrounds.append(appearance.background_rgb)
        tables.append(appearance.table_rgb)
        manifest.append(
            {
                "index": index,
                "domain_seed": episode_seed,
                "background": str(background_path.relative_to(args.output)),
                "table": str(table_path.relative_to(args.output)),
                "background_rgb": sample.background_rgb,
                "table_rgb": sample.table_rgb,
                "blur_sigma": sample.appearance_blur_sigma,
                "blob_count": sample.appearance_blob_count,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_rgb(args.output / "backgrounds_grid.png", _contact_sheet(backgrounds, args.columns, 320))
    _write_rgb(args.output / "tables_grid.png", _contact_sheet(tables, args.columns, 240))
    print(f"Wrote {args.count} procedural appearances and preview grids to {args.output}")


if __name__ == "__main__":
    main()
