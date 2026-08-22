# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for choosing a policy and for configuring how it is queried.

Shared by the sim runner, the hardware runner and the evaluation harness, which
must agree on what a checkpoint is and how it is queried or their numbers cannot
be compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.policies.policy import DEFAULT_CHECKPOINT, DEFAULT_INSTRUCTION


def add_policy_image_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the resolution the frames handed to a controller are reduced to.

    Shared rather than leaf-specific: a learned policy takes the size it was
    trained at, and the expert solves its camera intrinsics for the same size,
    so both need to agree on what the controller is looking at.
    """
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="height of the frames fed to the policy "
        "(default: the checkpoint's training height, else 480)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="width of the frames fed to the policy "
        "(default: the checkpoint's training width, else 640)",
    )


def add_checkpoint_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = DEFAULT_CHECKPOINT,
    required: bool = False,
    help: str = "HF policy checkpoint or local LeRobot directory",
) -> None:
    """Add ``--checkpoint``, which every learned policy needs and the expert does not.

    Separate from :func:`add_lerobot_arguments` because more than one leaf takes
    a checkpoint without taking the rest of how a LeRobot policy is queried.
    """
    parser.add_argument("--checkpoint", default=default, required=required, help=help)


def add_device_argument(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    """Add ``--device``, which any leaf that runs a network needs.

    ``default`` is not shared: the commands that run wherever they are started
    resolve "auto", while the two flow commands were written for a rented GPU box
    and default to ``cuda`` so that forgetting the flag fails loudly instead of
    quietly training on the CPU.
    """
    parser.add_argument("--device", default=default, help="auto | cpu | mps | cuda")


def add_lerobot_arguments(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_default: str | None = DEFAULT_CHECKPOINT,
    checkpoint_required: bool = False,
    n_action_steps_default: int | None = 100,
) -> None:
    """Add what a LeRobot checkpoint is and how it is queried.

    Every flag here is meaningless to a controller that is not a LeRobot policy,
    which is why it is a group of its own: a command with leaves declares it on
    the leaf, so nothing else has to police whether it applies.
    """
    add_checkpoint_argument(
        parser,
        default=checkpoint_default,
        required=checkpoint_required,
        help="HF policy checkpoint or local LeRobot directory",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="base model a LoRA checkpoint adapts, when the path recorded in its "
        "adapter_config.json does not exist here (also PAP_PI05_BASE)",
    )
    add_device_argument(parser)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=n_action_steps_default,
        help="queued actions to execute before re-querying a chunked policy"
        + (
            f" (default: {n_action_steps_default}; matches common ACT checkpoints; "
            "temporal ensembling uses 1)"
            if n_action_steps_default is not None
            else " (default: the checkpoint's own setting)"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="enable ACT temporal ensembling with this coefficient, e.g. 0.01; "
        "requires --n-action-steps 1",
    )


def add_flow_image_arguments(
    parser: argparse.ArgumentParser,
    *,
    recording_hw: bool = True,
    flow_export_required: bool = False,
) -> None:
    """Add the flags that configure the image-conditioned flow policy.

    The checkpoint holds only weights, so ``--flow-export`` names the dataset
    export it was trained against: the normalization bounds, the control rate and
    the input resolution all come from there.

    ``recording_hw`` adds ``--recording-hw``, which only the live runners need:
    they reduce camera frames through the training videos' resolution on the way
    to the model, while the evaluation harness renders at that resolution
    directly.

    ``flow_export_required`` lets a leaf that cannot run without the export say
    so in the parser, rather than checking it after the fact.
    """
    if recording_hw:
        parser.add_argument(
            "--recording-hw",
            type=int,
            nargs=2,
            default=None,
            metavar=("HEIGHT", "WIDTH"),
            help="resolution the training videos were recorded at, which observations "
            "are downsampled through on the way to the policy's input size. Defaults "
            "to the source_video_hw recorded by the dataset export, read from "
            "export.json beside the checkpoint's --flow-export",
        )
    parser.add_argument(
        "--flow-export",
        type=Path,
        default=None,
        required=flow_export_required,
        help="dataset export directory the checkpoint was trained on "
        "(holds export.json and normalization.npz)",
    )
    parser.add_argument(
        "--flow-act-steps",
        type=int,
        default=8,
        help="executed actions per policy query (default: 8)",
    )
    parser.add_argument(
        "--flow-integration-steps",
        type=int,
        default=10,
        help="Euler steps used to integrate the flow (default: 10)",
    )
    parser.add_argument(
        "--flow-seed",
        type=int,
        default=0,
        help="Torch seed for the flow's noise draw (default: 0)",
    )
    parser.add_argument(
        "--flow-noise-correlation",
        type=float,
        default=0.0,
        help="how much of the previous query's noise to carry into the next, from 0 "
        "(independent draws) to 1 (reused wherever the horizons overlap). Correlating "
        "the draws keeps consecutive chunks in the same mode, which smooths the motion "
        "at replan boundaries (default: 0)",
    )


def add_flow_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the checkpoint and the export that together are one image-flow policy.

    The checkpoint holds only weights: the normalization bounds, the control rate
    and the input resolution live in the export beside it, so neither is a policy
    on its own and no command takes one without the other.

    These spell themselves ``--checkpoint`` and ``--export``, unprefixed. The
    ``--flow-*`` names in :func:`add_flow_image_arguments` exist because the
    commands with leaves have to disambiguate against the lerobot flags; the two
    standalone flow commands have nothing to disambiguate against.
    """
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="flow-policy checkpoint-*.pt file"
    )
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="dataset export directory the checkpoint was trained on "
        "(holds export.json and normalization.npz)",
    )


def add_integration_steps_argument(parser: argparse.ArgumentParser, *, default: int = 10) -> None:
    """Add ``--integration-steps``, the Euler steps the flow is integrated over."""
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=default,
        help=f"Euler steps used to integrate the flow (default: {default})",
    )


def add_save_video_argument(parser: argparse.ArgumentParser, *, help: str) -> None:
    """Add ``--save-video``: write out the exact frames the policy was fed.

    Not a render of the scene -- the frames as the controller saw them, which is
    what makes it a diagnostic rather than a recording. ``help`` names the files
    each command writes, because that is all that differs.
    """
    parser.add_argument("--save-video", type=Path, default=None, help=help)


def add_step_limit_argument(parser: argparse.ArgumentParser, *, forever: str) -> None:
    """Add ``--steps``, the control-tick budget a closed-loop run stops after.

    Zero means "do not stop", and ``forever`` names what ends the run instead --
    Ctrl-C at the rig, a closed viewer in the sim. That the budget is counted in
    control ticks rather than seconds is the part both runners must agree on: a
    sim run and a rig run of the same length are then the same number of policy
    queries.
    """
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=f"stop after this many control ticks (0 = run until {forever})",
    )
