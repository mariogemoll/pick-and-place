# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The target sequence a chained, unattended run places onto.

A normal rig run localizes a paper plate and an operator moves it between
episodes. A chained run replaces both: the targets are decided up front, and
episode *n+1* picks the cube up from wherever episode *n* put it, so nothing
has to be reset by hand and a hundred episodes can run with nobody in the room.

That only works while every target is somewhere the cube can be **picked up
from**. The pickup sector is narrower than the drop sector — azimuth-limited
where the drop sector is not — so a target that is a legal placement is not
necessarily a legal start. Everything here exists to make that check happen
before the arm moves rather than at episode 40 with nobody watching.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.workspace_bounds import is_cube_recovery_target_allowed
from pick_and_place.scripted.scenario_sampling import (
    MIN_CHAIN_STEP_M,
    TARGET_INTERIOR_MARGIN_M,
    comfortably_interior,
    sample_target_chain,
)
from pick_and_place.spec.workspace import CUBE_REST_Z


def is_chainable_target(x: float, y: float) -> bool:
    """Whether a placement here is also a legal start for the next episode."""
    return comfortably_interior(
        x, y, TARGET_INTERIOR_MARGIN_M, is_cube_recovery_target_allowed
    )


def load_target_chain(path: Path) -> tuple[CubePose, ...]:
    """Read a pre-drawn sequence of ``[x, y]`` points in workspace metres.

    Every point is checked, and the first bad one names its own index. A
    hand-written or hand-edited sequence is exactly where an unreachable target
    comes from, and the cost of finding out on the rig is a stranded run.
    """
    with Path(path).open() as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must hold a non-empty JSON list of [x, y] points")
    chain: list[CubePose] = []
    for index, point in enumerate(payload):
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError(f"{path} entry {index} is not an [x, y] pair: {point!r}")
        x, y = (float(value) for value in point)
        if not is_chainable_target(x, y):
            raise ValueError(
                f"{path} entry {index} ({x:.3f}, {y:.3f}) is not a chainable target: "
                "the cube could be placed there but not reliably picked up again, "
                "which strands an unattended run"
            )
        chain.append(CubePose(x=x, y=y, z=CUBE_REST_Z))
    return tuple(chain)


class TargetChain:
    """The targets a chained run places onto: a fixed list, or drawn forever.

    Both shapes are the same idea and differ only in when the draw happens. A
    scored eval wants its targets **decided up front**, so the run is
    reproducible from a seed and an undrawable chain fails before the arm
    moves. An open-ended recording run cannot pre-draw anything, because it has
    no length — so it draws the next target when it needs it, from the same
    distribution and against the same chainability screen.
    """

    def __init__(
        self,
        targets: tuple[CubePose, ...] = (),
        *,
        rng: np.random.Generator | None = None,
        minimum_step_m: float = MIN_CHAIN_STEP_M,
    ) -> None:
        self._targets = list(targets)
        self._rng = rng
        self._minimum_step_m = minimum_step_m

    @property
    def endless(self) -> bool:
        """Whether this chain keeps drawing rather than running out."""
        return self._rng is not None

    @property
    def drawn(self) -> tuple[CubePose, ...]:
        """Every target decided so far, which for an endless chain grows."""
        return tuple(self._targets)

    def __len__(self) -> int:
        return len(self._targets)

    def target_for(self, index: int) -> CubePose:
        """The 1-based ``index``-th target, drawing it now if need be."""
        if index < 1:
            raise IndexError(f"episode index must be 1-based, got {index}")
        while index > len(self._targets):
            if self._rng is None:
                raise IndexError(
                    f"episode {index} has no target in a chain of {len(self._targets)}"
                )
            # Continue the chain from its own end, so the step constraint holds
            # across the join exactly as it does inside a pre-drawn chain.
            previous = self._targets[-1:]
            self._targets.extend(
                sample_target_chain(
                    self._rng,
                    1,
                    minimum_step_m=self._minimum_step_m,
                    start_after=previous[0] if previous else None,
                )
            )
        return self._targets[index - 1]


def resolve_target_chain(

    *, chain_seed: int | None, sequence_path: Path | None, episodes: int
) -> TargetChain | None:
    """The targets for this run, or ``None`` when it localizes a plate instead.

    ``episodes`` is the run's budget, and ``0`` means run until stopped. A
    seeded chain is pre-drawn to a bounded budget and drawn as it goes when
    there is none; a supplied sequence must cover the budget, so the run cannot
    walk off the end of its own targets partway through.
    """
    if chain_seed is None and sequence_path is None:
        return None
    if chain_seed is not None and sequence_path is not None:
        raise ValueError("pass a chain seed or a sequence file, not both")
    if episodes < 0:
        raise ValueError("episodes must not be negative")
    if chain_seed is not None:
        rng = np.random.default_rng(chain_seed)
        if episodes == 0:
            return TargetChain(rng=rng)
        # Bounded: draw it all now, so a chain that cannot be completed says so
        # before the run starts rather than partway through, unattended.
        return TargetChain(sample_target_chain(rng, episodes))
    assert sequence_path is not None
    chain = load_target_chain(sequence_path)
    if episodes == 0:
        raise ValueError(
            f"{sequence_path} holds {len(chain)} targets, which cannot drive a run "
            "with no episode budget; pass --episodes, or a chain seed to draw forever"
        )
    if len(chain) < episodes:
        raise ValueError(
            f"{sequence_path} holds {len(chain)} targets but the run asks for "
            f"{episodes} episodes"
        )
    return TargetChain(chain)


def write_target_chain(path: Path, chain: TargetChain | tuple[CubePose, ...]) -> None:
    """Save a drawn chain so a run can be repeated, or inspected before it runs."""
    targets = chain.drawn if isinstance(chain, TargetChain) else chain
    Path(path).write_text(
        json.dumps([[float(target.x), float(target.y)] for target in targets], indent=2)
    )


def pin_episode_target(controller: object, chain: TargetChain, index: int) -> str:
    """Point the controller at the chain's ``index``-th target (1-based).

    Pinning is what makes the controller skip plate localization altogether, so
    a chained run needs no plate in the scene. Returns the line to log, because
    an unattended run's transcript is the only record of what it aimed at.
    """
    target = chain.target_for(index)
    total = "endless" if chain.endless else str(len(chain))
    controller.drop_target_xy = (float(target.x), float(target.y))
    return f"Target {index}/{total} pinned at ({target.x:.3f}, {target.y:.3f})."
