# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generic per-episode recording: accumulate named per-tick arrays, then save.

Used by both the sim batch script (``record_episodes.py``) and the real-arm
executor (``executor.py``) to log their per-tick joint data and save it to
``.npz`` alongside episode-level metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EpisodeRecorder:
    """Accumulates named per-tick fields over an episode, then stacks and saves
    them as one ``.npz``.

    Call ``log(commanded=..., measured=..., **fields)`` once per tick with the
    same field names every time; each value is appended to that field's list.
    ``save(path, **metadata)`` stacks every field into an array (preserving the
    order ticks were logged in) via ``np.asarray``, merges in any episode-level
    metadata (already an array, or anything ``np.savez`` accepts), and writes
    the result with ``np.savez_compressed``.

    ``commanded`` and ``measured`` are required on every call: the joint
    targets sent this tick, and what was actually observed for those same
    joints. This is the one contract both backends must satisfy regardless of
    what else they log — sim's ``commanded`` is the ``ctrl`` set point and
    ``measured`` is ``qpos`` restricted to the robot joints; the real-arm
    executor's ``commanded`` is the servo target and ``measured`` is the
    encoder read-back. Anything beyond that (sim's full ``qpos``/``qvel``,
    the real executor's ``t``) is caller-defined extra fields.
    """

    def __init__(self) -> None:
        self._fields: dict[str, list] = {}

    def log(self, *, commanded: object, measured: object, **fields: object) -> None:
        self._fields.setdefault("commanded", []).append(commanded)
        self._fields.setdefault("measured", []).append(measured)
        for key, value in fields.items():
            self._fields.setdefault(key, []).append(value)

    def __len__(self) -> int:
        return len(next(iter(self._fields.values()), []))

    def stacked(self) -> dict[str, np.ndarray]:
        """The logged fields stacked into arrays, without saving."""
        return {key: np.asarray(values) for key, values in self._fields.items()}

    def save(self, path: str | Path, **metadata: object) -> dict[str, np.ndarray]:
        """Stack the logged fields, merge in ``metadata``, and write to ``path``.

        Returns the merged dict that was written.
        """
        record = self.stacked()
        record.update(metadata)
        np.savez_compressed(path, **record)
        return record
