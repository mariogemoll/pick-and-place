# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Per-episode lifecycle for a continuous run: indexing against an episode
budget and triggering periodic cooldowns, without touching what an episode
actually does.

``episode_loop`` is a generator the task's own loop iterates: it yields one
handle per attempt and owns only the bookkeeping around it (how many episodes
are left, when a cooldown is due). The task marks a yielded attempt complete by
calling ``ep.complete()``; attempts that are abandoned (cube not found, plan
infeasible, episode restarted) never call it, so they don't count toward the
budget or the cooldown cadence — matching how the task's own try/except already
decides abort-vs-continue.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field


@dataclass
class EpisodeHandle:
    """One attempt yielded by ``episode_loop``.

    ``attempt`` counts every attempt yielded so far (1-based). ``index`` is the
    1-based episode number this attempt would become if it completes. Call
    ``complete()`` once the episode has actually run to completion (no
    abort/restart) so the loop counts it against the budget and cooldown
    cadence.
    """

    attempt: int
    index: int
    _completed: bool = field(default=False, repr=False)

    def complete(self) -> None:
        self._completed = True


def episode_loop(
    *,
    target: int,
    rest_every: int,
    cooldown: Callable[[], None],
    should_continue: Callable[[], bool] = lambda: True,
) -> Iterator[EpisodeHandle]:
    """Yield one ``EpisodeHandle`` per attempt until ``target`` episodes
    complete (``target == 0`` means unbounded) or ``should_continue()`` turns
    false. After a completed episode, runs ``cooldown()`` if it lands on a
    ``rest_every`` boundary and isn't the last episode of the budget.
    """
    attempt = 0
    completed = 0
    while (target == 0 or completed < target) and should_continue():
        attempt += 1
        ep = EpisodeHandle(attempt=attempt, index=completed + 1)
        yield ep
        if ep._completed:
            completed += 1
            is_last = target != 0 and completed >= target
            if not is_last and rest_every > 0 and completed % rest_every == 0:
                cooldown()
