#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a learned or scripted controller on a frozen simulator manifest."""

from __future__ import annotations

import argparse

from pick_and_place.cli.eval_policy_sim import build_parser, validate


def run(args: argparse.Namespace) -> None:
    """Score the controller the parser resolved and report what it wrote."""
    # Imported here so that --help and a rejected argument cost neither torch
    # nor a compiled MuJoCo scene.
    from pick_and_place.rollout.evaluation import EvaluationRun, run_evaluation

    config = EvaluationRun.from_args(args)
    summary = run_evaluation(config)
    print(
        f"Wrote {config.output}: {summary['success_count']}/{summary['episode_count']} "
        f"successes ({summary['success_rate']:.1%})."
    )


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
