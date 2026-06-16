#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Roll out a policy through the shared sim eval harness and report success.

Evaluates either a trained BC checkpoint (``--checkpoint``) or the analytic
planner baseline (``--analytic``) on freshly sampled, analytically-feasible
episodes — the same sampler that generated the training demos, so the comparison
is apples-to-apples. ``--analytic`` is the harness's own sanity check: it should
land near 100%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pick_and_place.il.bc import BCPolicy, resolve_device
from pick_and_place.il.rollout import AnalyticPolicy, RolloutResult, evaluate

_DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "out" / "bc_policy.pt"


def _summarize(results: list[RolloutResult]) -> None:
    n = len(results)
    success = sum(r.success for r in results)
    placed = sum(r.placed for r in results)
    xy = np.array([r.xy_error for r in results])
    yaw = np.degrees([r.yaw_error for r in results])
    collided = sum(r.n_collisions > 0 for r in results)
    print(f"\n{'='*48}")
    print(f"episodes:           {n}")
    print(f"success (clean):    {success}/{n}  ({100*success/n:.1f}%)")
    print(f"placed (any):       {placed}/{n}  ({100*placed/n:.1f}%)")
    print(f"with collisions:    {collided}/{n}")
    print(f"xy error  median:   {np.median(xy)*1000:.1f} mm   max: {xy.max()*1000:.1f} mm")
    print(f"yaw error median:   {np.median(yaw):.1f}°   max: {yaw.max():.1f}°")
    print(f"{'='*48}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=Path, default=None, help="BC checkpoint (.pt)")
    group.add_argument(
        "--analytic", action="store_true", help="evaluate the analytic planner baseline"
    )
    parser.add_argument("-n", "--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000, help="held-out eval seed (≠ record seed)")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress per-episode lines")
    args = parser.parse_args()

    if args.analytic:
        make_policy = lambda ep: AnalyticPolicy(ep.trajectory, args.control_hz)  # noqa: E731
        label = "analytic baseline"
    else:
        ckpt = args.checkpoint or _DEFAULT_CKPT
        policy = BCPolicy.load(ckpt, device=resolve_device(args.device))
        make_policy = lambda ep: policy  # noqa: E731, ARG005
        label = f"BC checkpoint {ckpt.name}"

    print(f"Evaluating {label} on {args.num_episodes} episodes (seed {args.seed})")
    results = evaluate(
        make_policy,
        n_episodes=args.num_episodes,
        seed=args.seed,
        control_hz=args.control_hz,
        verbose=not args.quiet,
    )
    _summarize(results)


if __name__ == "__main__":
    main()
