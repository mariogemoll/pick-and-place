# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reading the artifacts a policy dataset export leaves beside a checkpoint.

An export directory holds ``export.json`` (what the dataset looked like) and
``normalization.npz`` (the per-dimension min-max bounds the states and actions
were squashed into). Every controller trained on such an export needs both to
undo at inference what the exporter did at training time, so the readers live
here rather than beside any one policy implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

BOUND_NAMES = ("obs_min", "obs_max", "action_min", "action_max")


def load_manifest(export: str | Path) -> dict[str, Any]:
    """Read ``export.json`` from an export directory."""
    with (Path(export) / "export.json").open() as file:
        manifest: dict[str, Any] = json.load(file)
    return manifest


def load_bounds(export: str | Path) -> dict[str, np.ndarray]:
    """Read the state and action min-max bounds from ``normalization.npz``."""
    with np.load(Path(export) / "normalization.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name], dtype=np.float32) for name in BOUND_NAMES}


def resolve_recording_hw(
    normalization: str | Path,
    override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve the intermediate video size a dataset export was built through.

    ``normalization`` is the ``normalization.npz`` path; ``export.json`` beside
    it records the resolution of the source dataset's videos.
    """
    if override is not None:
        height, width = override
        if height < 1 or width < 1:
            raise ValueError("recording height and width must be positive")
        return (height, width)

    export_path = Path(normalization).parent / "export.json"
    if not export_path.exists():
        raise FileNotFoundError(
            f"no export.json beside {normalization}; pass the recording height and width"
        )
    with export_path.open() as file:
        export = json.load(file)
    if "source_video_hw" not in export:
        raise ValueError(
            f"{export_path} predates source_video_hw; pass the resolution its source "
            "dataset's videos were recorded at"
        )
    height, width = export["source_video_hw"]
    return (int(height), int(width))


def normalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Squash raw values into ``[-1, 1]``, leaving degenerate dimensions at zero."""
    span = maximum - minimum
    return np.where(
        span > 1e-6, 2 * (values - minimum) / np.where(span > 1e-6, span, 1) - 1, 0
    ).astype(np.float32)


def unnormalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize`, collapsing degenerate dimensions onto the minimum."""
    span = maximum - minimum
    return np.where(span > 1e-6, (values + 1) / 2 * span + minimum, minimum).astype(np.float32)
