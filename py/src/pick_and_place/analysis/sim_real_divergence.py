# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Ask a policy the same question twice, in real pixels and in rendered ones.

`export_sim_real_pairs.py` writes, for every frame of a real episode, the real
camera image beside a MuJoCo render of the same instant -- same joints, same
cube pose, same rectified pinhole. That makes an experiment available that a
rollout cannot run: hold everything except the pixels fixed, and measure what
the appearance gap alone does to the policy's output.

The answer separates two failure modes that look identical on the rig. If the
action chunks agree, the policy sees through the appearance gap and whatever is
wrong on hardware is dynamics, calibration or compounding error -- and no amount
of rendering or randomization will help. If they disagree, the gap is the
pixels, and it is worth paying to close.

Sampling noise would otherwise swamp the comparison, since two queries of a flow
policy differ by their draw alone. Every comparison here integrates both
observations from the *same* noise, so the difference that comes back is
attributable to the conditioning.

Feeding one camera's real frame beside the other's render attributes the
divergence: the mixed conditions cost one extra query each and say which camera
is carrying the gap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Which frame each camera is taken from, per named condition. ``True`` selects
# the real image, ``False`` the render.
CONDITIONS: dict[str, tuple[bool, bool]] = {
    "sim": (False, False),
    "real": (True, True),
    "real_overhead": (True, False),
    "real_wrist": (False, True),
}

BASELINE_CONDITION = "sim"


@dataclass(frozen=True)
class DivergenceSummary:
    """How far a condition's predicted joint commands land from the baseline's.

    Degrees, over whatever frames were compared. ``per_chunk_step`` is indexed
    by position within the predicted horizon and ``per_joint`` by joint order,
    which is what separates "the whole chunk is shifted" from "the chunk drifts
    apart as it goes".
    """

    condition: str
    frames: int
    mean_deg: float
    median_deg: float
    p90_deg: float
    max_deg: float
    per_chunk_step: tuple[float, ...]
    per_joint: tuple[float, ...]


def summarize_divergence(
    condition: str, baseline: np.ndarray, other: np.ndarray
) -> DivergenceSummary:
    """Reduce two stacks of ``(frames, chunk_steps, joints)`` commands to a summary."""
    if baseline.shape != other.shape:
        raise ValueError(f"shape mismatch: {baseline.shape} against {other.shape}")
    if baseline.ndim != 3:
        raise ValueError(f"expected (frames, chunk_steps, joints), got {baseline.shape}")
    if not len(baseline):
        raise ValueError("no frames to compare")

    difference = np.abs(other - baseline)
    return DivergenceSummary(
        condition=condition,
        frames=int(len(baseline)),
        mean_deg=float(difference.mean()),
        median_deg=float(np.median(difference)),
        p90_deg=float(np.percentile(difference, 90)),
        max_deg=float(difference.max()),
        # Averaging over frames and joints leaves position within the horizon.
        per_chunk_step=tuple(float(value) for value in difference.mean(axis=(0, 2))),
        per_joint=tuple(float(value) for value in difference.mean(axis=(0, 1))),
    )


def compare_conditions(
    predict: Callable[[bool, bool, int], np.ndarray],
    frames: int,
    *,
    conditions: dict[str, tuple[bool, bool]] | None = None,
) -> dict[str, DivergenceSummary]:
    """Run every condition over ``frames`` frames and summarize against the baseline.

    ``predict`` is asked for one frame's chunk under a condition, as
    ``(real_overhead, real_wrist, frame_index)``. It selects the images itself,
    which is what lets it hold the observation *history* in the same condition
    as the frame under test rather than mixing domains within one query.
    """
    conditions = conditions or CONDITIONS
    if BASELINE_CONDITION not in conditions:
        raise ValueError(f"conditions must include the {BASELINE_CONDITION!r} baseline")
    if frames < 1:
        raise ValueError("no frames to compare")

    predictions: dict[str, np.ndarray] = {
        name: np.stack([predict(real_overhead, real_wrist, index) for index in range(frames)])
        for name, (real_overhead, real_wrist) in conditions.items()
    }

    baseline = predictions[BASELINE_CONDITION]
    return {
        name: summarize_divergence(name, baseline, chunks)
        for name, chunks in predictions.items()
        if name != BASELINE_CONDITION
    }
