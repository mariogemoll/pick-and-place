# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Task-phase structure of pick-and-place episodes.

Newly recorded simulation episodes store exact per-frame phase spans taken from
the scripted controller (``PhaseSpan``). Episodes recorded before that carry no
phase information, so ``segment_phases`` reconstructs approximate boundaries
from the gripper command trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

PHASES = ("acquisition", "grasp", "transport", "placement")

# Scripted-trajectory phase names, mapped onto the four coarse task phases.
COARSE_PHASE_BY_TRAJECTORY_PHASE = {
    "approach": "acquisition",
    "descent": "acquisition",
    "grasp": "grasp",
    "lift": "transport",
    "recovery_lift": "transport",
    "carry": "transport",
    "drop_descent": "placement",
    "release": "placement",
    "retreat": "placement",
}


@dataclass(frozen=True)
class PhaseSpan:
    """One contiguous run of recorded frames executing a named trajectory phase."""

    name: str
    start_frame: int


def phase_spans_json(spans: tuple[PhaseSpan, ...]) -> str:
    """Serialize spans for one episode-metadata column."""
    _validate_spans(spans)
    return json.dumps([[span.name, span.start_frame] for span in spans])


def phase_spans_from_json(text: str) -> tuple[PhaseSpan, ...]:
    """Parse and validate the ``phase_spans`` episode-metadata column."""
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("phase_spans must be a JSON list")
    spans = tuple(PhaseSpan(name=str(name), start_frame=int(start)) for name, start in payload)
    _validate_spans(spans)
    return spans


def _validate_spans(spans: tuple[PhaseSpan, ...]) -> None:
    if not spans:
        raise ValueError("phase spans must be nonempty")
    if spans[0].start_frame != 0:
        raise ValueError("the first phase span must start at frame 0")
    starts = [span.start_frame for span in spans]
    if starts != sorted(set(starts)):
        raise ValueError("phase span starts must be strictly increasing")


def coarse_phase_labels(spans: tuple[PhaseSpan, ...], length: int) -> np.ndarray:
    """One coarse phase name per frame, from exact recorded spans."""
    _validate_spans(spans)
    if length < spans[-1].start_frame + 1:
        raise ValueError(f"episode length {length} is shorter than the last span start")
    labels = np.empty(length, dtype=object)
    for index, span in enumerate(spans):
        coarse = COARSE_PHASE_BY_TRAJECTORY_PHASE.get(span.name)
        if coarse is None:
            raise ValueError(f"unknown trajectory phase name: {span.name!r}")
        end = spans[index + 1].start_frame if index + 1 < len(spans) else length
        labels[span.start_frame : end] = coarse
    return labels

# The gripper command must span at least this fraction of typical hardware
# travel; a flatter trace means the episode never closed the gripper.
_MINIMUM_COMMAND_SPAN = 5.0


@dataclass(frozen=True)
class PhaseBoundaries:
    """Key gripper-command transitions on the trace's own tick timeline.

    ``close_tick`` is the first tick whose command is below the open/closed
    midpoint. ``reopen_tick`` is the first later tick back above the midpoint,
    or ``None`` when the gripper never reopens.
    """

    length: int
    close_tick: int
    reopen_tick: int | None
    grasp_halfwidth: int

    def labels(self) -> np.ndarray:
        """Return one phase name per tick."""
        labels = np.empty(self.length, dtype=object)
        grasp_start = max(0, self.close_tick - self.grasp_halfwidth)
        grasp_end = min(self.length, self.close_tick + self.grasp_halfwidth)
        reopen = self.length if self.reopen_tick is None else self.reopen_tick
        grasp_end = min(grasp_end, reopen)
        labels[:grasp_start] = "acquisition"
        labels[grasp_start:grasp_end] = "grasp"
        labels[grasp_end:reopen] = "transport"
        labels[reopen:] = "placement"
        return labels

    def phase_of(self, tick: int) -> str:
        if not 0 <= tick < self.length:
            raise IndexError(f"tick {tick} outside episode of length {self.length}")
        return str(self.labels()[tick])

    def cube_at_start_until(self, safety_ticks: int = 2) -> int:
        """Last tick (exclusive) where the cube provably rests at its start pose."""
        return max(0, self.close_tick - safety_ticks)


def segment_phases(gripper_command: np.ndarray, *, grasp_halfwidth: int = 5) -> PhaseBoundaries:
    """Segment one episode's gripper command trace (larger values = more open).

    The scripted demonstrations open the gripper, close it once on the cube,
    carry, and reopen to place. The retreat at the end of some episodes closes
    the gripper partway again; only the first closing counts.
    """
    trace = np.asarray(gripper_command, dtype=np.float64).reshape(-1)
    if len(trace) < 4:
        raise ValueError("gripper trace is too short to segment")
    if not np.all(np.isfinite(trace)):
        raise ValueError("gripper trace contains non-finite values")
    if grasp_halfwidth < 1:
        raise ValueError("grasp_halfwidth must be at least 1")
    span = float(trace.max() - trace.min())
    if span < _MINIMUM_COMMAND_SPAN:
        raise ValueError(f"gripper command span {span:.2f} is too small; gripper never closed")
    fraction = (trace - float(trace.min())) / span
    # Episodes may start with the gripper partly closed and open it during the
    # approach, so anchor closing detection to the first near-fully-open tick.
    open_ticks = np.flatnonzero(fraction >= 0.75)
    if len(open_ticks) == 0:
        raise ValueError("gripper command never reaches the open level")
    below = np.flatnonzero(fraction[open_ticks[0] :] < 0.5)
    if len(below) == 0:
        raise ValueError("gripper never closed after opening")
    close_tick = int(open_ticks[0]) + int(below[0])
    above_after_close = np.flatnonzero(fraction[close_tick:] > 0.5)
    reopen_tick = None if len(above_after_close) == 0 else close_tick + int(above_after_close[0])
    return PhaseBoundaries(
        length=len(trace),
        close_tick=close_tick,
        reopen_tick=reopen_tick,
        grasp_halfwidth=grasp_halfwidth,
    )
