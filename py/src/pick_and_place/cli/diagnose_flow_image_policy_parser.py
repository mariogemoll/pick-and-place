# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The flow-policy diagnostic's parser, separate from the diagnosis itself.

Kept apart for the reason the evaluator's parser is: the command imports
torch at module scope, so asking it what its flags are would otherwise cost a
second and a half and a deep-learning stack. Here it costs an argparse import.
"""

from __future__ import annotations

from pick_and_place.cli.common import add_seed_argument
from pick_and_place.cli.policy import (
    add_device_argument,
    add_flow_export_arguments,
    add_integration_steps_argument,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser


def build_parser(description: str | None = None) -> SuggestingArgumentParser:
    """Return the parser for the open-loop flow diagnostic."""
    parser = SuggestingArgumentParser(description=description)
    add_flow_export_arguments(parser)
    parser.add_argument(
        "--batches", type=int, default=20, help="held-out batches to sample (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="windows per batch (default: 32)"
    )
    add_integration_steps_argument(parser)
    add_device_argument(parser, default="cuda")
    add_seed_argument(parser, default=0, help="Torch seed for the flow's noise draw")
    return parser
