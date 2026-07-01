# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reset-snapshot source for the RL snapshot curriculum.

A pool of recorded pick-and-place episodes (the ``.npz`` files written by
``record_episodes.py``) is the distribution of valid states the curriculum env
resets into. It can be used for strict reverse curricula, but the current
training setup composes skills more flexibly: learn a useful carry skill, add
the drop, and later move the reset window back into grasping and approach.

Every recorded frame is a full ``qpos``/``qvel`` snapshot the sim can be restored
to exactly, and each episode carries the frame index at which each scripted phase
begins, so "reset at the start of the carry phase" resolves to the right frame in
every episode regardless of how long it ran.

Only successful episodes back the distribution: an episode that missed the
target or clipped something is not a valid state to finish from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResetSnapshot:
    """A full-state snapshot to restore the sim into at the start of an episode."""

    qpos: np.ndarray  # (nq,) full sim configuration
    qvel: np.ndarray  # (nv,) full sim velocity
    ctrl: np.ndarray  # (6,) recorded joint set point at this frame, JOINT_NAMES order
    target_xy: np.ndarray  # (2,) drop target in the floor plane
    frame: int  # sampled frame index within the source episode
    total_frames: int  # length of the source episode
    source: Path  # the episode the snapshot was drawn from


@dataclass(frozen=True)
class _EpisodeIndex:
    """The cheap per-episode metadata read up front to drive sampling."""

    path: Path
    phase_boundaries: np.ndarray  # (n_phases,) first frame of each scripted phase
    total_frames: int
    target_xy: np.ndarray  # (2,)


class EpisodePool:
    """Successful recorded episodes, queryable for per-phase reset snapshots.

    The light per-episode metadata (phase boundaries, frame count, target) is read
    once at construction; the heavy ``qpos``/``qvel``/``commanded`` arrays are
    loaded and cached lazily the first time an episode is actually sampled, so a
    large pool costs memory only for the episodes the curriculum has reached.
    """

    def __init__(self, directory: Path, *, require_success: bool = True) -> None:
        paths = sorted(Path(directory).glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no episode_*.npz found in {directory}")

        self._phase_names: tuple[str, ...] | None = None
        self.nq: int | None = None
        self.nv: int | None = None
        self.control_hz: float | None = None
        self._episodes: list[_EpisodeIndex] = []
        skipped_failures = 0
        for path in paths:
            with np.load(path, allow_pickle=True) as record:
                if "phase_boundaries" not in record:
                    raise ValueError(
                        f"{path.name} has no phase_boundaries; re-record with the "
                        "current record_episodes.py"
                    )
                if require_success and not bool(record["success"]):
                    skipped_failures += 1
                    continue
                names = tuple(str(n) for n in record["phase_names"])
                qpos = record["qpos"]
                self._ensure_consistent(path, names, qpos.shape[1], record)
                target = record["cube_target"]
                self._episodes.append(
                    _EpisodeIndex(
                        path=path,
                        phase_boundaries=np.asarray(record["phase_boundaries"], dtype=np.int64),
                        total_frames=int(qpos.shape[0]),
                        target_xy=np.asarray(target[:2], dtype=np.float64),
                    )
                )

        if not self._episodes:
            raise ValueError(
                f"no usable episodes in {directory} "
                f"({skipped_failures} present but unsuccessful)"
            )
        self.skipped_failures = skipped_failures
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _ensure_consistent(
        self, path: Path, names: tuple[str, ...], nq: int, record
    ) -> None:
        """Pin the pool-wide invariants (phase order, state width) to the first
        episode and reject any later one that disagrees — a mixed pool would make
        a phase resolve to incomparable frames or restore into the wrong model."""
        if self._phase_names is None:
            self._phase_names = names
            self.nq = nq
            self.nv = int(record["qvel"].shape[1])
            self.control_hz = float(record["control_hz"])
            return
        if names != self._phase_names:
            raise ValueError(
                f"{path.name} phase order {names} != pool's {self._phase_names}"
            )
        if nq != self.nq:
            raise ValueError(f"{path.name} qpos width {nq} != pool's {self.nq}")

    @property
    def phase_names(self) -> tuple[str, ...]:
        assert self._phase_names is not None
        return self._phase_names

    def __len__(self) -> int:
        return len(self._episodes)

    def _left_edge(self, episode: _EpisodeIndex, phase: str, phase_fraction: float) -> int:
        """First frame the reset is allowed to start at for ``phase``.

        ``phase_fraction`` is the escape hatch for a phase jump that is too hard:
        instead of starting at the phase boundary it starts that fraction of the
        way into the phase, measured in this episode's own frames so unequal
        trajectory lengths are handled automatically.
        """
        names = self.phase_names
        if phase not in names:
            raise ValueError(f"phase {phase!r} not in {names}")
        index = names.index(phase)
        left = int(episode.phase_boundaries[index])
        if phase_fraction > 0.0:
            if index + 1 < len(episode.phase_boundaries):
                nxt = int(episode.phase_boundaries[index + 1])
            else:
                nxt = episode.total_frames
            left += int(round(phase_fraction * (nxt - left)))
        return min(left, episode.total_frames - 1)

    def _phase_frame(
        self, episode: _EpisodeIndex, phase: str, phase_fraction: float
    ) -> int:
        """Frame at ``phase_fraction`` through ``phase`` for this episode."""
        names = self.phase_names
        if phase not in names:
            raise ValueError(f"phase {phase!r} not in {names}")
        index = names.index(phase)
        left = int(episode.phase_boundaries[index])
        if index + 1 < len(episode.phase_boundaries):
            right = int(episode.phase_boundaries[index + 1])
        else:
            right = episode.total_frames
        frame = left + int(round(phase_fraction * (right - left)))
        return min(max(left, frame), episode.total_frames - 1)

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(path)
        if cached is None:
            with np.load(path, allow_pickle=True) as record:
                cached = (
                    np.asarray(record["qpos"], dtype=np.float64),
                    np.asarray(record["qvel"], dtype=np.float64),
                    np.asarray(record["commanded"], dtype=np.float64),
                )
            self._cache[path] = cached
        return cached

    def sample_reset(
        self,
        rng: np.random.Generator,
        phase: str,
        *,
        phase_fraction: float = 0.0,
        phase_end_fraction: float | None = None,
    ) -> ResetSnapshot:
        """Draw a reset snapshot for the curriculum stage that begins at ``phase``.

        An episode is picked uniformly at random, then a start frame is sampled
        uniformly over that episode's allowed region. By default this is from the
        phase's left edge through the final frame, so the policy keeps rehearsing
        the later phases it has already solved. ``phase_end_fraction`` narrows the
        right edge to a fraction through the selected phase, which is useful for
        learning a phase by sweeping backward within it.
        """
        episode = self._episodes[int(rng.integers(len(self._episodes)))]
        left = self._left_edge(episode, phase, phase_fraction)
        right = (
            self._phase_frame(episode, phase, phase_end_fraction)
            if phase_end_fraction is not None
            else episode.total_frames - 1
        )
        if right < left:
            raise ValueError(
                f"empty reset window for phase {phase!r}: left frame {left}, right frame {right}"
            )
        frame = int(rng.integers(left, right + 1))
        qpos, qvel, commanded = self._load(episode.path)
        return ResetSnapshot(
            qpos=qpos[frame].copy(),
            qvel=qvel[frame].copy(),
            ctrl=commanded[frame].copy(),
            target_xy=episode.target_xy.copy(),
            frame=frame,
            total_frames=episode.total_frames,
            source=episode.path,
        )
