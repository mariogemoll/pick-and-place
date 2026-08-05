# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for choosing a policy and for the Diffusion Policy server.

Shared by the sim runner, the hardware runner and the evaluation harness, which
must agree on what a checkpoint is and how it is queried or their numbers cannot
be compared.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pick_and_place.core.paths import REPO_ROOT
from pick_and_place.policies.policy import DEFAULT_CHECKPOINT, DEFAULT_INSTRUCTION

DEFAULT_DIFFUSION_POLICY_CONFIG = (
    REPO_ROOT / "config" / "diffusion_policy" / "pretrain_so101_unet_img.yaml"
)


def add_policy_arguments(
    parser: argparse.ArgumentParser,
    *,
    controllers: tuple[str, ...] = ("lerobot", "diffusion-policy"),
    checkpoint_default: str | None = DEFAULT_CHECKPOINT,
    n_action_steps_default: int | None = 100,
) -> None:
    """Add the controller choice, the checkpoint, and how it is queried.

    ``controllers`` lists the implementations this command can run — the
    evaluation harness also scores the analytic ``scripted`` policy against the
    learned ones. ``checkpoint_default`` is ``None`` where a command has no
    sensible default and the caller must name one.
    """
    parser.add_argument(
        "--controller",
        choices=controllers,
        default="lerobot",
        help="policy implementation (default: lerobot)",
    )
    parser.add_argument(
        "--checkpoint",
        default=checkpoint_default,
        help="HF policy checkpoint or local LeRobot directory, or a Diffusion "
        "Policy state_*.pt file",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="language task string")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
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


def add_diffusion_policy_arguments(
    parser: argparse.ArgumentParser, *, recording_hw: bool = True
) -> None:
    """Add the flags that configure the out-of-process Diffusion Policy server.

    ``recording_hw`` adds ``--recording-hw``, which only the live runners need:
    they reduce camera frames through the training videos' resolution on the way
    to the model, while the evaluation harness renders at that resolution
    directly.
    """
    parser.add_argument(
        "--diffusion-policy-python",
        type=Path,
        default=os.environ.get("DIFFUSION_POLICY_PYTHON"),
        help="interpreter of the DPPO virtual environment (default: $DIFFUSION_POLICY_PYTHON)",
    )
    parser.add_argument(
        "--diffusion-policy-config",
        type=Path,
        default=DEFAULT_DIFFUSION_POLICY_CONFIG,
        help=f"training configuration YAML (default: {DEFAULT_DIFFUSION_POLICY_CONFIG})",
    )
    parser.add_argument(
        "--diffusion-policy-normalization",
        type=Path,
        help="normalization.npz written by the Diffusion Policy dataset export",
    )
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
            "export.json beside --diffusion-policy-normalization",
        )
    parser.add_argument(
        "--diffusion-policy-act-steps",
        type=int,
        default=None,
        help="executed actions per policy query (default: the training configuration)",
    )
    parser.add_argument(
        "--diffusion-policy-seed",
        type=int,
        default=0,
        help="Torch seed for DDPM action sampling (default: 0)",
    )
    parser.add_argument(
        "--diffusion-policy-ddim-steps",
        type=int,
        default=None,
        help="sample with DDIM using this many steps instead of the trained DDPM "
        "schedule; much faster, but not the training sampler, so not for headline runs",
    )
