# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export a LeRobot dataset for visual Diffusion Policy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pick_and_place.cli.common import add_output_argument
from pick_and_place.cli.dataset import (
    add_image_size_argument,
    add_max_episodes_argument,
    add_source_dataset_argument,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.data.diffusion_policy_dataset import (
    DEFAULT_POLICY_HZ,
    GOAL_TARGET_XY,
    export_diffusion_policy_dataset,
)
from pick_and_place.spec.action_encoding import ActionEncoding, parse_action_encoding


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the dataset exporter."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_source_dataset_argument(parser)
    add_output_argument(parser, required=True, help="new Diffusion Policy dataset directory")
    add_image_size_argument(parser)
    parser.add_argument(
        "--policy-hz",
        type=int,
        default=DEFAULT_POLICY_HZ,
        help=f"output sampling rate; must divide the source FPS (default: {DEFAULT_POLICY_HZ})",
    )
    add_max_episodes_argument(parser, help="export only the first N episodes for a smoke run")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel video decoding processes (default: 2)",
    )
    parser.add_argument(
        "--action-encoding",
        choices=[encoding.value for encoding in ActionEncoding],
        default=ActionEncoding.ABSOLUTE.value,
        help=(
            "what the policy predicts: the joint command itself, or its offset "
            "from the joints measured on the same control tick (default: absolute)"
        ),
    )
    parser.add_argument(
        "--goal",
        choices=["none", "target-xy"],
        default="none",
        help=(
            "append a goal slot to the state vector. 'none' leaves the state as the "
            "source recorded it, which is every export written before this flag "
            "existed. 'target-xy' appends the episode's target point, so the policy "
            "is told where to place instead of having to read it off the drop plate. "
            "The slot is normalized with the same min-max bounds as every other state "
            "dimension, and its width is recorded in export.json as goal_dim "
            "(default: none)"
        ),
    )
    parser.add_argument(
        "--bounds-from",
        type=Path,
        default=None,
        help=(
            "reuse the state and action min-max bounds of an earlier export "
            "(a directory, or a normalization.npz) instead of fitting them to this "
            "data. Required when the export will continue training an existing "
            "checkpoint: the weights learned what a normalized unit means under "
            "the original bounds, so refitting rescales the inputs and actions out "
            "from under them and a fine-tune partly measures recovery from that "
            "rescaling rather than what it set out to measure"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Write the export the image flow policy trains on."""
    manifest = export_diffusion_policy_dataset(
        args.src,
        args.output,
        image_size=args.image_size,
        policy_hz=args.policy_hz,
        max_episodes=args.max_episodes,
        workers=args.workers,
        action_encoding=parse_action_encoding(args.action_encoding),
        goal=None if args.goal == "none" else GOAL_TARGET_XY,
        bounds_from=args.bounds_from,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
