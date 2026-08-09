#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export privileged simulation state as generic conditional-flow examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pick_and_place.data.flow_policy_dataset import export_flow_policy_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="finalized LeRobot dataset")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--policy-hz", type=int, default=10)
    parser.add_argument("--observation-steps", type=int, default=2)
    parser.add_argument("--prediction-steps", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    manifest = export_flow_policy_dataset(
        args.src,
        args.output,
        policy_hz=args.policy_hz,
        observation_steps=args.observation_steps,
        prediction_steps=args.prediction_steps,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_episodes=args.max_episodes,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
