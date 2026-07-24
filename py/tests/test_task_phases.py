# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.task_phases import (
    PHASES,
    PhaseSpan,
    coarse_phase_labels,
    phase_spans_from_json,
    phase_spans_json,
    segment_phases,
)


def _scripted_trace() -> np.ndarray:
    """Open plateau, close ramp, closed carry, reopen, partial re-close."""
    return np.concatenate([
        np.full(30, 39.3),
        np.linspace(39.3, 13.9, 8),
        np.full(60, 13.9),
        np.linspace(13.9, 39.3, 8),
        np.full(10, 39.3),
        np.linspace(39.3, 24.9, 6),
    ])


def test_segments_scripted_episode() -> None:
    trace = _scripted_trace()
    boundaries = segment_phases(trace)
    midpoint = (39.3 + 13.9) / 2.0
    assert trace[boundaries.close_tick] < midpoint
    assert trace[boundaries.close_tick - 1] >= midpoint
    assert boundaries.reopen_tick is not None
    assert boundaries.reopen_tick > boundaries.close_tick
    labels = boundaries.labels()
    assert len(labels) == len(trace)
    assert set(labels) == set(PHASES)
    # Phases appear in canonical order.
    order = [PHASES.index(label) for label in labels]
    assert order == sorted(order)


def test_final_partial_reclose_stays_in_placement() -> None:
    boundaries = segment_phases(_scripted_trace())
    assert boundaries.labels()[-1] == "placement"


def test_grasp_window_is_centered_on_close() -> None:
    boundaries = segment_phases(_scripted_trace(), grasp_halfwidth=3)
    labels = boundaries.labels()
    assert labels[boundaries.close_tick] == "grasp"
    assert labels[boundaries.close_tick - 3] == "grasp"
    assert labels[boundaries.close_tick - 4] == "acquisition"
    assert labels[boundaries.close_tick + 3] == "transport"


def test_cube_at_start_until_precedes_close() -> None:
    boundaries = segment_phases(_scripted_trace())
    assert 0 < boundaries.cube_at_start_until() < boundaries.close_tick


def test_never_reopened_gripper_has_no_placement() -> None:
    trace = np.concatenate([np.full(20, 39.3), np.full(20, 13.9)])
    boundaries = segment_phases(trace)
    assert boundaries.reopen_tick is None
    assert "placement" not in set(boundaries.labels())


def test_rejects_flat_trace() -> None:
    with pytest.raises(ValueError, match="never closed"):
        segment_phases(np.full(50, 39.3))


def test_rejects_gripper_that_never_recloses() -> None:
    trace = np.concatenate([np.full(20, 13.9), np.full(20, 39.3)])
    with pytest.raises(ValueError, match="never closed after opening"):
        segment_phases(trace)


def test_partially_closed_start_opens_before_grasp() -> None:
    """Initial pose randomization may start the gripper below the midpoint."""
    trace = np.concatenate([
        np.full(10, 20.0),
        np.linspace(20.0, 39.3, 5),
        np.full(20, 39.3),
        np.linspace(39.3, 13.9, 6),
        np.full(40, 13.9),
        np.linspace(13.9, 39.3, 6),
        np.full(10, 39.3),
    ])
    boundaries = segment_phases(trace)
    assert boundaries.close_tick > 35
    labels = boundaries.labels()
    assert labels[0] == "acquisition"
    assert labels[-1] == "placement"


def _recorded_spans() -> tuple[PhaseSpan, ...]:
    return (
        PhaseSpan("approach", 0),
        PhaseSpan("descent", 40),
        PhaseSpan("grasp", 90),
        PhaseSpan("lift", 105),
        PhaseSpan("carry", 130),
        PhaseSpan("drop_descent", 200),
        PhaseSpan("release", 230),
        PhaseSpan("retreat", 250),
    )


def test_phase_spans_json_round_trip() -> None:
    spans = _recorded_spans()
    assert phase_spans_from_json(phase_spans_json(spans)) == spans


def test_phase_spans_json_rejects_nonzero_first_start() -> None:
    with pytest.raises(ValueError, match="frame 0"):
        phase_spans_json((PhaseSpan("approach", 3),))


def test_phase_spans_json_rejects_unsorted_starts() -> None:
    spans = (PhaseSpan("approach", 0), PhaseSpan("grasp", 50), PhaseSpan("descent", 40))
    with pytest.raises(ValueError, match="strictly increasing"):
        phase_spans_json(spans)


def test_coarse_phase_labels_cover_episode() -> None:
    labels = coarse_phase_labels(_recorded_spans(), length=280)
    assert len(labels) == 280
    assert labels[0] == "acquisition"
    assert labels[89] == "acquisition"
    assert labels[90] == "grasp"
    assert labels[105] == "transport"
    assert labels[199] == "transport"
    assert labels[200] == "placement"
    assert labels[279] == "placement"
    assert set(labels) == set(PHASES)


def test_coarse_phase_labels_reject_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unknown trajectory phase"):
        coarse_phase_labels((PhaseSpan("teleport", 0),), length=10)


def test_coarse_phase_labels_reject_short_length() -> None:
    with pytest.raises(ValueError, match="shorter"):
        coarse_phase_labels(_recorded_spans(), length=250)
