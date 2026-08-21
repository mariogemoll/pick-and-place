# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for what the simulated scene contains and how it is rendered.

The recorder, the sim runner and the evaluation harness must set the scene up
identically or a policy is trained on one distribution and scored on another.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.data.recording_config import RENDER_HEIGHT, RENDER_WIDTH
from pick_and_place.runtime.preflight import PreflightDebug
from pick_and_place.variants.appearance import APPEARANCE_PRESETS



def add_cube_pose_arguments(
    parser: argparse.ArgumentParser,
    *,
    source_default: tuple[float, float] | None = None,
    source_yaw: bool = True,
) -> None:
    """Add the flags that pin the cube and the drop zone instead of sampling them."""
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=source_default,
        help="pin the source cube (x, y); omit to sample one in the clearance annulus",
    )
    if source_yaw:
        parser.add_argument(
            "--source-yaw",
            type=float,
            default=0.0,
            help="source cube yaw in degrees, only used with --source (default: 0.0)",
        )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the drop-zone center (x, y); omit to sample one in the clearance annulus",
    )


def add_render_size_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the offscreen MuJoCo render size the camera frames start from."""
    parser.add_argument(
        "--render-width",
        type=int,
        default=RENDER_WIDTH,
        help="MuJoCo source render width before downsampling/cropping "
        f"(default: {RENDER_WIDTH})",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=RENDER_HEIGHT,
        help="MuJoCo source render height before downsampling/cropping "
        f"(default: {RENDER_HEIGHT})",
    )


def add_scene_appearance_arguments(
    parser: argparse.ArgumentParser, *, default: str | None = None
) -> None:
    """Add the recolouring that matches a re-rendered dataset's look.

    ``default`` names the palette a command renders in when it is not told
    otherwise; ``None`` leaves the scene as compiled.
    """
    parser.add_argument(
        "--scene-appearance",
        type=str,
        default=default,
        metavar="NAME",
        help="recolour the scene the way a re-rendered dataset was rendered, either a "
        f"preset ({', '.join(sorted(APPEARANCE_PRESETS))}) or an ad-hoc spec such as "
        "'cube=blue,floor=dark-gray' (default: "
        + (f"{default})" if default is not None else "the scene as compiled)"),
    )


def add_scene_texture_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the photographic backdrop and table surface of the finite-floor scene."""
    parser.add_argument(
        "--background-panorama",
        type=Path,
        default=None,
        help="equirectangular room panorama to render as a skybox behind the scene",
    )
    parser.add_argument(
        "--table-texture",
        type=Path,
        default=None,
        help="top-down table texture reconstructed from overhead footage, for the floor",
    )


def add_preflight_debug_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the diagnostics for trajectory candidates preflight rejected."""
    parser.add_argument(
        "--preflight-debug",
        action="store_true",
        help="print detailed collision diagnostics for rejected trajectory candidates",
    )
    parser.add_argument(
        "--preflight-debug-limit",
        type=int,
        default=12,
        help="maximum detailed contact rows to print per rejected candidate",
    )
    parser.add_argument(
        "--save-failed-trajectories",
        type=Path,
        default=None,
        help="directory for replayable .npz rollouts of rejected preflight candidates",
    )
    parser.add_argument(
        "--failed-trajectory-limit",
        type=int,
        default=8,
        help="maximum rejected candidates to save",
    )


def preflight_debug_from_args(args: argparse.Namespace) -> PreflightDebug:
    """Collect the flags :func:`add_preflight_debug_arguments` added."""
    return PreflightDebug(
        print_contacts=args.preflight_debug,
        contact_limit=args.preflight_debug_limit,
        trajectory_dir=args.save_failed_trajectories,
        trajectory_limit=args.failed_trajectory_limit,
    )
