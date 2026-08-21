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


def add_lerobot_arguments(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_default: str | None = DEFAULT_CHECKPOINT,
    n_action_steps_default: int | None = 100,
) -> None:
    """Add what a LeRobot checkpoint is and how it is queried.

    Every flag here is meaningless to a controller that is not a learned policy,
    which is why it is a group of its own: a command with leaves declares it on
    the leaf, so nothing else has to police whether it applies.
    """
    parser.add_argument(
        "--checkpoint",
        default=checkpoint_default,
        help="HF policy checkpoint or local LeRobot directory, or a flow-policy "
        "checkpoint-*.pt file",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="base model a LoRA checkpoint adapts, when the path recorded in its "
        "adapter_config.json does not exist here (also PAP_PI05_BASE)",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
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


def add_policy_arguments(
    parser: argparse.ArgumentParser,
    *,
    controllers: tuple[str, ...] = ("lerobot",),
    checkpoint_default: str | None = DEFAULT_CHECKPOINT,
    n_action_steps_default: int | None = 100,
) -> None:
    """Add the controller choice and every controller's flags at once.

    This is the flat shape: one namespace, so a command using it accepts each
    family's flags whichever controller was chosen, and has to reject the
    inapplicable ones by hand. Commands with leaves should take
    :func:`add_policy_image_arguments` and :func:`add_lerobot_arguments`
    separately instead, and let the parser do that work.
    """
    parser.add_argument(
        "--controller",
        choices=controllers,
        default="lerobot",
        help="policy implementation (default: lerobot)",
    )
    add_lerobot_arguments(
        parser,
        checkpoint_default=checkpoint_default,
        n_action_steps_default=n_action_steps_default,
    )
    add_policy_image_arguments(parser)


def add_flow_image_arguments(
    parser: argparse.ArgumentParser, *, recording_hw: bool = True
) -> None:
    """Add the flags that configure the image-conditioned flow policy.

    The checkpoint holds only weights, so ``--flow-export`` names the dataset
    export it was trained against: the normalization bounds, the control rate and
    the input resolution all come from there.

    ``recording_hw`` adds ``--recording-hw``, which only the live runners need:
    they reduce camera frames through the training videos' resolution on the way
    to the model, while the evaluation harness renders at that resolution
    directly.
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
