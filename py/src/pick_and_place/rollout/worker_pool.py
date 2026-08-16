# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run episode indices across a pool of worker processes that outlive their failures.

Recording a large collection is embarrassingly parallel, so the interesting part
is not the split but what happens when a worker stops doing its job. Two ways
have been seen on real runs, and neither announces itself:

*wedging* -- a worker spins at 100% CPU and never finishes its episode, before
recording anything or partway through. It cannot time itself out, because it is
not running Python that would notice, so the deadline has to be enforced from
outside.

*crashing* -- a worker dies of an exception. Nothing is stuck, so a watchdog
that only judges the living sees a healthy pool that happens to be smaller, and
the run finishes quietly at a fraction of its width.

Both are handled the same way: kill if needed, start a replacement, keep going.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Callable


def find_wedged_workers(
    status: dict,
    worker_ids,
    *,
    now: float,
    episode_timeout: float,
) -> list[tuple[int, int, float]]:
    """Return ``(worker_id, episode, age)`` for each worker past its deadline.

    A worker is only judged while an episode is in flight. Between episodes it
    reports ``None``, and an idle worker with an empty queue would otherwise
    look indistinguishable from a wedged one and be killed forever.
    """
    wedged = []
    for worker_id in worker_ids:
        episode, since = status.get(worker_id, (None, now))
        if episode is None:
            continue
        age = now - since
        if age > episode_timeout:
            wedged.append((worker_id, episode, age))
    return wedged


def claim_retry(attempts: dict[int, int], episode: int, episode_retries: int) -> bool:
    """Record a wedge against ``episode``; return whether to requeue it.

    Bounding this matters: requeuing unconditionally would spin forever on an
    index that wedges every time it is attempted.
    """
    attempts[episode] = attempts.get(episode, 0) + 1
    return attempts[episode] <= episode_retries


def run_pool(
    job: dict,
    *,
    worker: Callable,
    indices: list[int],
    workers: int,
    episode_timeout: float,
    episode_retries: int = 1,
    poll_interval: float = 5.0,
) -> None:
    """Run ``indices`` across ``workers`` processes, replacing any that stop working.

    Workers pull from a shared queue, so one that finishes early takes more work
    rather than idling on a pre-assigned block. ``worker`` is the process entry
    point, called with ``(job, index_queue, status, worker_id)``.

    A killed episode is requeued at most ``episode_retries`` times and then
    abandoned. Unbounded requeuing would spin forever if an index wedges
    deterministically; abandoning costs one episode, and the caller already
    treats episodes as attempts rather than guaranteed successes.
    """
    # Spawn rather than fork: each worker needs its own MuJoCo GL context, which
    # does not survive a fork. Spawn is the default on macOS and safe on Linux.
    ctx = multiprocessing.get_context("spawn")
    index_queue = ctx.Queue()
    for index in indices:
        index_queue.put(index)
    status = ctx.Manager().dict()

    def start(worker_id: int):
        status[worker_id] = (None, time.time())
        proc = ctx.Process(
            target=worker,
            args=(
                {**job, "label": f"[w{worker_id}] ", "show_progress": worker_id == 0},
                index_queue,
                status,
                worker_id,
            ),
        )
        proc.start()
        return proc

    procs = {worker_id: start(worker_id) for worker_id in range(workers)}
    killed = 0
    crashed = 0
    # A worker that dies instantly must not spin here forever.
    crash_budget = max(workers * 3, 12)
    abandoned: list[int] = []
    attempts: dict[int, int] = {}
    try:
        while True:
            alive = {wid: p for wid, p in procs.items() if p.is_alive()}
            # Exit code zero means the queue ran dry, which is how a worker is
            # supposed to end. Anything else is a death worth replacing.
            for worker_id, proc in list(procs.items()):
                if proc.is_alive() or proc.exitcode in (0, None):
                    continue
                if crashed >= crash_budget:
                    continue
                crashed += 1
                print(
                    f"\n[watchdog] worker {worker_id} died with exit code "
                    f"{proc.exitcode}; replacing it ({crashed}/{crash_budget})"
                )
                procs[worker_id] = start(worker_id)
                alive[worker_id] = procs[worker_id]
            if not alive:
                break
            now = time.time()
            wedged = find_wedged_workers(
                status, list(alive), now=now, episode_timeout=episode_timeout
            )
            for worker_id, episode, age in wedged:
                retry = claim_retry(attempts, episode, episode_retries)
                print(
                    f"\n[watchdog] worker {worker_id} stuck on episode {episode} "
                    f"for {age:.0f}s (limit {episode_timeout:.0f}s); killing and "
                    + ("requeuing" if retry else "abandoning it (retry limit reached)")
                )
                alive[worker_id].kill()
                alive[worker_id].join(timeout=30)
                if retry:
                    index_queue.put(episode)
                else:
                    abandoned.append(episode)
                killed += 1
                procs[worker_id] = start(worker_id)
            time.sleep(poll_interval)
    finally:
        for proc in procs.values():
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=30)

    failed = [wid for wid, p in procs.items() if p.exitcode not in (0, -9)]
    if killed:
        print(f"[watchdog] replaced {killed} wedged worker(s) during the run")
    if crashed:
        print(f"[watchdog] replaced {crashed} crashed worker(s) during the run")
    if abandoned:
        print(f"[watchdog] abandoned episode(s) after repeated wedges: {sorted(abandoned)}")
    if failed:
        # Loud, not silent: a bare join() could not distinguish this from success.
        print(f"WARNING: worker(s) exited with an error: {failed}")
