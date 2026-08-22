# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The image-flow sim runner's parser, separate from the run itself.

Kept apart for the reason the evaluator's parser is: the command imports
torch at module scope, so asking it what its flags are would otherwise cost a
deep-learning stack.

The flag spellings here are unprefixed -- ``--checkpoint``, ``--export``,
``--act-steps``, ``--integration-steps``. The ``--flow-*`` names elsewhere exist
because the commands with leaves must disambiguate against the lerobot flags,
and this command has nothing to disambiguate against.
"""

from __future__ import annotations

from pathlib import Path

from pick_and_place.cli.common import add_output_argument
from pick_and_place.cli.policy import (
    add_device_argument,
    add_flow_export_arguments,
    add_integration_steps_argument,
    add_save_video_argument,
)
from pick_and_place.cli.scene import (
    add_scene_appearance_arguments,
    add_seed_base_argument,
    add_viewer_argument,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser


def build_parser(description: str | None = None) -> SuggestingArgumentParser:
    """Return the parser for the image-flow sim runner."""
    parser = SuggestingArgumentParser(description=description)
    add_flow_export_arguments(parser)
    parser.add_argument(
        "--scenarios",
        type=int,
        default=50,
        help="how many scenes of the seed stream to run (0 = keep going until the viewer "
        "is closed or Ctrl-C)",
    )
    add_seed_base_argument(parser, default=6_000_000)
    parser.add_argument(
        "--act-steps", type=int, default=8, help="executed actions per policy query (default: 8)"
    )
    add_integration_steps_argument(parser)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--noise-correlation",
        type=float,
        default=0.0,
        help="how much of the previous query's noise to carry into the next, from 0 "
        "(independent draws) to 1 (reused wherever the horizons overlap). Correlating "
        "the draws keeps consecutive chunks in the same mode (default: 0)",
    )
    add_device_argument(parser, default="cuda")
    add_scene_appearance_arguments(parser)
    add_output_argument(parser, help="directory for the per-scenario rollout JSON")
    parser.add_argument(
        "--record-trace",
        type=Path,
        default=None,
        help="directory to write one .bin per scenario holding the replay state and the "
        "flow integration path behind every generated horizon, for the web viewer",
    )
    add_viewer_argument(
        parser,
        help="watch the rollouts in the MuJoCo viewer, throttled to the control rate "
        "(run under mjpython)",
    )
    add_save_video_argument(
        parser, help="directory to write one mp4 per scenario of the frames the policy sees"
    )
    return parser
