# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The frozen-manifest evaluator's parser, separate from the evaluation itself.

Kept apart so that asking this command what its flags are costs an ``argparse``
import rather than torch, lerobot and a MuJoCo scene. That is what lets a
reference generator ask the parser instead of reading the source with ``ast``,
and what a dispatcher needs to render ``--help`` for a controller it may never
load.

Nothing here runs anything: :func:`build_parser` declares, :func:`validate`
rejects. Both are importable and testable without a simulator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pick_and_place.cli.policy import (
    add_checkpoint_argument,
    add_device_argument,
    add_flow_image_arguments,
    add_lerobot_arguments,
    add_policy_image_arguments,
)
from pick_and_place.cli.scene import add_render_size_arguments, add_scene_appearance_arguments

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "config" / "evaluation" / "smoke_v1.json"


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Return the evaluator's parser: a shared world, three controller leaves."""
    # The manifest, the output and the world the scenarios run in: shared, so a
    # flow number and a SmolVLA number are produced under one declaration.
    common = argparse.ArgumentParser(add_help=False)
    parser = common
    add_policy_image_arguments(parser)
    add_render_size_arguments(parser)
    add_scene_appearance_arguments(parser)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"frozen scenario manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--output", type=Path, required=True, help="new evaluation run directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N scenarios for a non-headline wiring check",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "skip the first N scenarios, applied before --limit. Together the two shard one "
            "suite across concurrent workers; the shards stay comparable because each scenario "
            "is independent and carries its own seed"
        ),
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help=("cap each scenario's simulated duration; useful for fast approach-only diagnostics"),
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="save the exact overhead and wrist policy frames for every scenario",
    )
    parser.add_argument(
        "--save-rollouts",
        action="store_true",
        help=(
            "save every scenario's per-frame qpos as a PPRL file the browser episode "
            "viewer replays; a few kilobytes an episode against megabytes of video"
        ),
    )
    parser = argparse.ArgumentParser(description=description)
    leaves = parser.add_subparsers(dest="controller", required=True, metavar="CONTROLLER")

    lerobot = leaves.add_parser(
        "lerobot",
        parents=[common],
        help="a LeRobot checkpoint (ACT, SmolVLA, pi0.5, ...)",
        description="Score a LeRobot checkpoint against a frozen scenario manifest.",
    )
    add_lerobot_arguments(
        lerobot, checkpoint_default=None, checkpoint_required=True, n_action_steps_default=None
    )

    flow_image = leaves.add_parser(
        "flow-image",
        parents=[common],
        help="the image-conditioned flow-matching policy",
        description="Score an image-flow export against a frozen scenario manifest.",
    )
    add_checkpoint_argument(
        flow_image, default=None, required=True, help="flow-policy checkpoint-*.pt file"
    )
    add_device_argument(flow_image)
    # recording_hw=False: the evaluator renders at the target resolution directly,
    # so the flag the live runners need would parse here and do nothing.
    add_flow_image_arguments(flow_image, recording_hw=False, flow_export_required=True)

    scripted = leaves.add_parser(
        "scripted",
        parents=[common],
        help="the expert: localize, plan, servo the descent, replan at each phase",
        description="Score the expert against a frozen scenario manifest.",
    )
    scripted.add_argument(
        "--scripted-perception",
        choices=("geometric", "detector"),
        default="geometric",
        help=(
            "simulated overhead perception: geometric uses the 80%% segmentation "
            "visibility gate and controlled pose beliefs; detector runs the real "
            "optical pipeline (default: geometric)"
        ),
    )
    return parser


def validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject what the parser cannot express, calling ``parser.error`` on failure."""
    # Imported here rather than at module scope: parsing appearance names pulls
    # in a MuJoCo scene, and building the parser should not.
    from pick_and_place.variants.appearance import parse_appearance

    if (args.image_height is None) != (args.image_width is None):
        parser.error("pass both --image-height and --image-width, or neither")
    if args.image_height is not None and min(args.image_height, args.image_width) < 1:
        parser.error("image dimensions must be positive")
    if min(args.render_height, args.render_width) < 1:
        parser.error("render dimensions must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.offset < 0:
        parser.error("--offset must not be negative")
    if args.scene_appearance is not None:
        try:
            parse_appearance(args.scene_appearance)
        except ValueError as exc:
            parser.error(str(exc))
    if args.max_episode_seconds is not None and (
        not math.isfinite(args.max_episode_seconds) or args.max_episode_seconds <= 0.0
    ):
        parser.error("--max-episode-seconds must be a positive finite number")
    if args.controller == "flow-image":
        for name, path in (("checkpoint", args.checkpoint), ("flow-export", args.flow_export)):
            if not Path(path).exists():
                parser.error(f"--{name} does not exist: {path}")
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
