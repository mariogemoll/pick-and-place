# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Does the input noise move this policy's actions at all?

DSRL can only re-weight modes the behavior-cloned policy already has. It cannot
invent an action the diffusion policy would never produce for any ``w``, because
every action it can reach is ``pi_dp(s, w)`` for some ``w``. So the method has a
precondition, and this task is a plausible place for it to fail: every
demonstration here comes from a deterministic analytic planner, and a diffusion
policy fitted to a near-unimodal demonstrator can collapse to a map that ignores
its noise. If it has, there is nothing to steer and no configuration of the
learner will help.

Two measurements, from cheapest to most decisive:

- **Action spread.** At a state, denoise ``K`` independent noise draws and look
  at how far apart the resulting chunks are. Reported against two reference
  scales, because a raw standard deviation in normalized action units means
  nothing on its own: the spread of the policy's action *across consecutive
  control ticks* (how much it moves anyway), and the full normalized range.
- **Outcome spread.** Whether different draws at the same scene produce
  different episode outcomes. This is the one that matters -- action diversity
  that never changes an outcome is not a lever -- and it needs no new rollout
  code, because the noise draw is already the only randomness in an evaluation
  and ``check_dppo_rl_env.py --seed`` already varies it. Run it ``R`` times over
  the same scene stream and :func:`summarize_outcome_spread` reads the scores.

The gate is a floor, not a prediction of success: clearing it says only that the
latent space has something in it worth searching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionSpread:
    """How far ``K`` noise draws move the chunk, at one batch of states."""

    # Per-action-dimension standard deviation across the noise draws, averaged
    # over dimensions and states, in normalized action units.
    noise_std: float
    # The same quantity for consecutive control ticks within one chunk: how much
    # the commanded action moves anyway as the trajectory advances. The ratio of
    # the two is the readable number.
    step_std: float
    # Largest pairwise distance between two denoised chunks, per state, averaged.
    max_pairwise_l2: float
    # Per-dimension spread of the first executed action only, which is what the
    # environment acts on before the next observation arrives.
    first_action_std: float
    samples: int

    @property
    def ratio(self) -> float:
        """Noise-driven spread as a multiple of ordinary tick-to-tick motion."""
        return self.noise_std / self.step_std if self.step_std > 0 else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "noise_std": self.noise_std,
            "step_std": self.step_std,
            "ratio": self.ratio,
            "max_pairwise_l2": self.max_pairwise_l2,
            "first_action_std": self.first_action_std,
            "samples": self.samples,
        }


def measure_action_spread(chunks: np.ndarray) -> ActionSpread:
    """Summarize ``K`` denoised chunks per state.

    Args:
        chunks: ``(K, B, horizon_steps, action_dim)`` normalized actions, the
            same ``B`` states denoised from ``K`` independent noise draws.

    Returns:
        The spread statistics for this batch.
    """
    if chunks.ndim != 4:
        raise ValueError(f"expected (K, B, T, D) chunks, got {chunks.shape}")
    draws, batch = chunks.shape[:2]
    if draws < 2:
        raise ValueError("action spread needs at least two noise draws")

    noise_std = float(chunks.std(axis=0).mean())
    # Ordinary motion within a chunk: the tick-to-tick difference of a single
    # draw, which is the scale the noise-driven spread has to be read against.
    step_std = float(np.abs(np.diff(chunks, axis=2)).mean())
    flattened = chunks.reshape(draws, batch, -1)
    distances = np.linalg.norm(
        flattened[:, None, :, :] - flattened[None, :, :, :], axis=-1
    )
    max_pairwise = float(distances.max(axis=(0, 1)).mean())
    first_action_std = float(chunks[:, :, 0, :].std(axis=0).mean())
    return ActionSpread(
        noise_std=noise_std,
        step_std=step_std,
        max_pairwise_l2=max_pairwise,
        first_action_std=first_action_std,
        samples=batch,
    )


def combine_action_spreads(spreads: list[ActionSpread]) -> ActionSpread:
    """Average per-batch spreads, weighted by how many states each covered."""
    if not spreads:
        raise ValueError("nothing to combine")
    weights = np.array([spread.samples for spread in spreads], dtype=np.float64)
    weights = weights / weights.sum()

    def mean(name: str) -> float:
        return float(sum(w * getattr(s, name) for w, s in zip(weights, spreads, strict=True)))

    return ActionSpread(
        noise_std=mean("noise_std"),
        step_std=mean("step_std"),
        max_pairwise_l2=mean("max_pairwise_l2"),
        first_action_std=mean("first_action_std"),
        samples=int(sum(spread.samples for spread in spreads)),
    )


@dataclass(frozen=True)
class OutcomeSpread:
    """How often repeated noise draws disagree about the same scene."""

    scenarios: int
    repeats: int
    # Scenes where at least one draw succeeded and at least one failed. These
    # are the only scenes an outcome-driven learner can get traction on: they
    # are where the return depends on something the latent policy controls.
    contested: int
    always_success: int
    always_failure: int
    mean_success_rate: float

    @property
    def contested_fraction(self) -> float:
        return self.contested / self.scenarios if self.scenarios else float("nan")

    @property
    def headroom(self) -> float:
        """Success rate an oracle picking the best draw per scene would reach."""
        return (
            (self.always_success + self.contested) / self.scenarios
            if self.scenarios
            else float("nan")
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "scenarios": self.scenarios,
            "repeats": self.repeats,
            "contested": self.contested,
            "contested_fraction": self.contested_fraction,
            "always_success": self.always_success,
            "always_failure": self.always_failure,
            "mean_success_rate": self.mean_success_rate,
            "oracle_headroom": self.headroom,
        }


def summarize_outcome_spread(runs: list[list[dict[str, Any]]]) -> OutcomeSpread:
    """Pair repeated scorings of the same scene stream by scenario id.

    Args:
        runs: One list of episode records per repeat, as
            ``check_dppo_rl_env.py --output`` writes them. Each record needs
            ``scenario_id`` and ``success``.

    Returns:
        The outcome spread over scenarios every repeat covered. Scenarios
        missing from any repeat are dropped rather than counted with fewer
        draws, so every scene contributes the same number of trials.
    """
    if len(runs) < 2:
        raise ValueError("outcome spread needs at least two repeats")
    per_run = [
        {record["scenario_id"]: bool(record["success"]) for record in run} for run in runs
    ]
    shared = set(per_run[0])
    for outcomes in per_run[1:]:
        shared &= set(outcomes)
    if not shared:
        raise ValueError(
            "the repeats share no scenario ids -- they have to score the same "
            "scene stream, so scene_seed_base and episode count must match"
        )

    contested = always_success = always_failure = 0
    total = 0.0
    for scenario in shared:
        results = [outcomes[scenario] for outcomes in per_run]
        total += sum(results)
        if all(results):
            always_success += 1
        elif not any(results):
            always_failure += 1
        else:
            contested += 1
    return OutcomeSpread(
        scenarios=len(shared),
        repeats=len(runs),
        contested=contested,
        always_success=always_success,
        always_failure=always_failure,
        mean_success_rate=total / (len(shared) * len(runs)),
    )


def load_episode_records(path: Path) -> list[dict[str, Any]]:
    """Read the episode list out of a ``check_dppo_rl_env.py`` score file."""
    payload = json.loads(Path(path).read_text())
    return payload["episodes"] if isinstance(payload, dict) else payload
