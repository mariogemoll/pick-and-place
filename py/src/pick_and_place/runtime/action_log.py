# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Per-attempt logs of what a policy predicted against what was executed."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ActionLog:
    """Accumulate one attempt's per-tick actions and write them to
    ``<root>/attempt_NNN.npz`` when the attempt ends.

    Logged per tick: the measured joint state, the action the policy returned
    (the ensembled one when temporal ensembling is on), and the velocity-capped
    command actually sent. Whenever the model predicted a fresh chunk that tick
    (every tick under ensembling, every ``n_action_steps`` ticks otherwise), the
    whole chunk is logged too, keyed by the tick it arrived on. Everything is in
    the real frame (degrees, gripper 0-100). Row 0 of a chunk is the model's
    freshest prediction for its arrival tick, so ``chunks[i, 0] - action`` at
    ``chunk_tick[i]`` measures how far the ensemble lags the newest prediction,
    and comparing chunks across arrival ticks exposes mode flips.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempt = 0
        self._clear()

    def _clear(self) -> None:
        self._tick: list[int] = []
        self._t: list[float] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._commanded: list[np.ndarray] = []
        self._chunk_tick: list[int] = []
        self._chunks: list[np.ndarray] = []

    def start_attempt(self) -> None:
        self.attempt += 1
        self._clear()

    def log_tick(
        self,
        tick: int,
        t: float,
        state: np.ndarray,
        action: np.ndarray,
        commanded: np.ndarray,
        chunk: np.ndarray | None = None,
    ) -> None:
        self._tick.append(tick)
        self._t.append(t)
        self._state.append(np.asarray(state, dtype=np.float32))
        self._action.append(np.asarray(action, dtype=np.float32))
        self._commanded.append(np.asarray(commanded, dtype=np.float32))
        if chunk is not None:
            self._chunk_tick.append(tick)
            self._chunks.append(np.asarray(chunk, dtype=np.float32))

    def end_attempt(self, outcome: str) -> None:
        if not self._tick:
            return
        path = self.root / f"attempt_{self.attempt:03d}.npz"
        np.savez_compressed(
            path,
            tick=np.array(self._tick, dtype=np.int64),
            t=np.array(self._t, dtype=np.float64),
            state=np.stack(self._state),
            action=np.stack(self._action),
            commanded=np.stack(self._commanded),
            chunk_tick=np.array(self._chunk_tick, dtype=np.int64),
            chunks=(
                np.stack(self._chunks)
                if self._chunks
                else np.zeros((0, 0, 0), dtype=np.float32)
            ),
            outcome=np.array(outcome),
        )
        print(f"Wrote action log: {path}")
