# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Confirm two exported Diffusion Policy datasets differ only in appearance.

    python scripts/check_variant_pair.py <dp_export_a> <dp_export_b>

Two variants re-rendered from one replay of the same episodes are supposed to be
trajectory-identical: same states, same actions, same episode lengths, differing
only at the pixels the appearance changes. That property is what lets a training
comparison attribute a difference to the cube's appearance rather than to a
reseeded recording. It is also easy to lose silently -- a re-render against the
wrong staged root, or a finalize that admitted a different set of episodes,
produces two plausible datasets that are no longer a matched pair.

Exits non-zero if the states, actions or episode lengths differ at all. The
image difference is reported rather than judged: how much of the frame the
appearance touches is the thing being measured, not a pass criterion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The exporter stacks RGB from both cameras into one tensor, overhead first.
CAMERA_CHANNELS = {"overhead": slice(0, 3), "wrist": slice(3, 6)}
# A grey-level gap this size survives H.264 and downsampling to 96x96; below it
# the pixel is codec noise rather than a rendered difference.
CHANGED_PIXEL_THRESHOLD = 8


def load(path: Path) -> dict[str, np.ndarray]:
    npz = path / "train.npz" if path.is_dir() else path
    with np.load(npz) as data:
        return {key: data[key] for key in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="a Diffusion Policy export dir or train.npz")
    parser.add_argument("second", type=Path, help="the other export to compare against")
    args = parser.parse_args()

    a, b = load(args.first), load(args.second)

    failures = []
    for key in ("states", "actions", "traj_lengths"):
        if a[key].shape != b[key].shape:
            failures.append(f"{key} shape {a[key].shape} != {b[key].shape}")
        elif not np.array_equal(a[key], b[key]):
            worst = np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64)))
            failures.append(f"{key} differs (max |delta| {worst:g})")

    print(f"episodes {len(a['traj_lengths'])}, frames {len(a['states'])}")
    for key in ("states", "actions", "traj_lengths"):
        verdict = "differs" if any(f.startswith(key) for f in failures) else "identical"
        print(f"  {key:12s} {verdict}")

    if a["images"].shape != b["images"].shape:
        failures.append(f"images shape {a['images'].shape} != {b['images'].shape}")
    else:
        delta = np.abs(a["images"].astype(np.int16) - b["images"].astype(np.int16))
        print("  images      differ only where the appearance changes:")
        for camera, channels in CAMERA_CHANNELS.items():
            per_camera = delta[:, channels]
            changed = (per_camera.max(axis=1) > CHANGED_PIXEL_THRESHOLD).mean()
            print(
                f"    {camera:9s} mean |delta| {per_camera.mean():6.2f}  "
                f"max {per_camera.max():3d}  pixels changed {changed * 100:6.2f}%"
            )
        if not delta.any():
            failures.append("images are identical; the two variants rendered the same appearance")

    if failures:
        print("\nNOT a matched pair:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nMatched pair: trajectories identical, appearance differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
