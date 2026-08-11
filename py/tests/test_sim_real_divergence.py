# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Attributing a policy's output difference to the pixels it was conditioned on."""

from __future__ import annotations

import numpy as np
import pytest

from pick_and_place.analysis.sim_real_divergence import (
    CONDITIONS,
    compare_conditions,
    summarize_divergence,
)

FRAMES = 5
CHUNK_STEPS = 4
JOINTS = 6


def stacks(overhead_gap: float, wrist_gap: float):
    """Frames whose real images differ from their renders by a known constant."""
    rng = np.random.default_rng(0)
    overhead, wrist = [], []
    for _ in range(FRAMES):
        base = rng.random((2, 2))
        overhead.append((base, base + overhead_gap))
        wrist.append((base, base + wrist_gap))
    return overhead, wrist


def make_predict(overhead, wrist):
    """A stand-in policy: the chunk is a fixed multiple of what it was shown.

    Deterministic in its inputs, so any difference the comparison reports is the
    conditioning rather than a sampler draw. The wrist is weighted twice so the
    attribution tests can tell the two cameras apart.
    """

    def predict(real_overhead: bool, real_wrist: bool, index: int) -> np.ndarray:
        chosen_overhead = overhead[index][int(real_overhead)]
        chosen_wrist = wrist[index][int(real_wrist)]
        value = float(chosen_overhead.mean()) + 2.0 * float(chosen_wrist.mean())
        return np.full((CHUNK_STEPS, JOINTS), value)

    return predict


def test_identical_pixels_diverge_by_nothing() -> None:
    overhead, wrist = stacks(0.0, 0.0)
    summaries = compare_conditions(make_predict(overhead, wrist), FRAMES)
    assert set(summaries) == set(CONDITIONS) - {"sim"}
    for summary in summaries.values():
        assert summary.mean_deg == pytest.approx(0.0)
        assert summary.max_deg == pytest.approx(0.0)


def test_the_divergence_is_attributed_to_the_camera_carrying_the_gap() -> None:
    """Only the overhead frames differ, so only the conditions using them move."""
    overhead, wrist = stacks(0.5, 0.0)
    summaries = compare_conditions(make_predict(overhead, wrist), FRAMES)
    assert summaries["real_wrist"].mean_deg == pytest.approx(0.0)
    assert summaries["real_overhead"].mean_deg == pytest.approx(0.5)
    assert summaries["real"].mean_deg == pytest.approx(0.5)


def test_the_stand_in_policy_weights_the_wrist_twice() -> None:
    """Guards the attribution test: a wrist gap must show up doubled."""
    overhead, wrist = stacks(0.0, 0.5)
    summaries = compare_conditions(make_predict(overhead, wrist), FRAMES)
    assert summaries["real_overhead"].mean_deg == pytest.approx(0.0)
    assert summaries["real_wrist"].mean_deg == pytest.approx(1.0)


def test_frame_count_and_axis_summaries() -> None:
    baseline = np.zeros((FRAMES, CHUNK_STEPS, JOINTS))
    other = np.zeros_like(baseline)
    # A single joint, at a single position in the horizon, off by three degrees.
    other[:, 2, 4] = 3.0
    summary = summarize_divergence("real", baseline, other)
    assert summary.frames == FRAMES
    assert summary.max_deg == pytest.approx(3.0)
    assert summary.per_chunk_step[2] == pytest.approx(3.0 / JOINTS)
    assert summary.per_chunk_step[0] == pytest.approx(0.0)
    assert summary.per_joint[4] == pytest.approx(3.0 / CHUNK_STEPS)
    assert summary.per_joint[0] == pytest.approx(0.0)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        summarize_divergence("real", np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))


def test_an_empty_comparison_is_rejected() -> None:
    with pytest.raises(ValueError, match="no frames"):
        summarize_divergence("real", np.zeros((0, 3, 4)), np.zeros((0, 3, 4)))
    with pytest.raises(ValueError, match="no frames"):
        compare_conditions(lambda *_: np.zeros((1, 1)), 0)


def test_conditions_must_include_the_baseline() -> None:
    overhead, wrist = stacks(0.0, 0.0)
    with pytest.raises(ValueError, match="baseline"):
        compare_conditions(make_predict(overhead, wrist), FRAMES, conditions={"real": (True, True)})


