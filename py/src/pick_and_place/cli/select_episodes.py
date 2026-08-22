# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Print the ``episode_index`` of episodes matching a filter, one per line.

A generic selector over a LeRobotDataset's episode metadata. By default it
keeps the *successful* episodes: the cube was seen by the overhead camera after
placement (``placement_detected``) and landed within ``--xy-tolerance`` metres
of the target centre. Pass ``--query`` for an arbitrary pandas expression over
the episode-metadata columns instead. The kept indices go to stdout (one per
line, in ``episode_index`` order) while the human-readable summary goes to
stderr, so stdout stays clean for redirecting to a file or piping straight into
a consumer such as ``convert_dataset_resolution.py --episodes-file -``.

Success is treated as a derived notion, not a stored column: the placement
error is recomputed as ``hypot(cube_end - target)`` in the XY plane from the raw
points every dataset records, so the selector depends on no stored derived
column. A synthesized ``placement_xy`` column is available to ``--query`` for
the same reason.

This script only *selects*; it never writes a dataset. That keeps the selection
policy in one place and lets any consumer decide what to do with the list --
convert its resolution, split it, copy it, count it, etc.

Examples:

    # Good episodes to a file, then convert only those to 512x512
    pap select-episodes --src datasets/20260702 > good.txt
    pap convert-dataset-resolution \
        --src datasets/20260702 --width 512 --height 512 --episodes-file good.txt

    # Same thing in one pipe
    pap select-episodes --src datasets/20260702 \
        | pap convert-dataset-resolution \
            --src datasets/20260702 --width 512 --height 512 --episodes-file -

    # Arbitrary filter (placement_xy is synthesized and available)
    pap select-episodes --src datasets/20260702 \
        --query "placement_detected and placement_xy <= 0.02"
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pick_and_place.cli.dataset import (
    add_source_dataset_argument,
    add_success_tolerance_argument,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.data.dataset_subset import (
    SUCCESS_XY_TOLERANCE_M,
    load_all_episodes,
    successful_episode_mask,
)


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the episode selector."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_source_dataset_argument(parser)
    add_success_tolerance_argument(
        parser,
        help=f"success placement-XY tolerance in metres (default: {SUCCESS_XY_TOLERANCE_M})",
    )
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "pandas query over episode-metadata columns, overriding the default success "
            'filter (e.g. "placement_detected and placement_xy <= 0.02")'
        ),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """List the episodes of a dataset that pass the filter."""
    episodes = load_all_episodes(args.src).copy()

    if args.query is None:
        kept = episodes[successful_episode_mask(episodes, args.xy_tolerance)]
        what = f"placement_detected and placement_xy <= {args.xy_tolerance}"
    else:
        # Synthesize placement_xy so a --query can reference it uniformly.
        episodes["placement_xy"] = np.hypot(
            episodes["cube_end_x"] - episodes["target_x"],
            episodes["cube_end_y"] - episodes["target_y"],
        )
        kept = episodes.query(args.query)
        what = f"query {args.query!r}"

    kept_indices = [int(i) for i in kept["episode_index"].tolist()]

    print(
        f"{args.src}: {len(episodes)} episode(s), {len(kept_indices)} match {what}",
        file=sys.stderr,
    )
    for idx in kept_indices:
        print(idx)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
