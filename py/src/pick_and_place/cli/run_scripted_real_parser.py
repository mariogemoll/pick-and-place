# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The scripted rig runner's parser, separate from the run itself.

Kept apart for the reason ``run_policy_real_parser`` is: the runner imports
mujoco at module scope, so asking it what its flags are cost a second and a
half, and here it costs an argparse import. And the same second reason —
``run_scripted_real.py`` sits just under the repository's 40 KB ceiling, and the
flag declarations that moved here are bytes it no longer carries.

Nothing here runs anything: :func:`build_parser` declares, :func:`validate`
rejects. Both are importable and testable without an arm attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.cli.rig import (
    add_drop_zone_arguments,
    add_follower_arguments,
    add_joint_zeros_argument,
    add_operator_alert_arguments,
    add_overhead_recalibration_arguments,
    add_rig_camera_arguments,
    add_target_chain_arguments,
)
from pick_and_place.cli.scene import (
    add_preflight_debug_arguments,
    add_speed_argument,
    add_viewer_argument,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.paths import REPO_ROOT


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the rig runner."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_follower_arguments(parser)
    add_joint_zeros_argument(
        parser,
        default=REPO_ROOT / "config" / "joint_zeros.json",
        help="session joint-zero calibration mapping servo readback into the model frame "
        "(default: config/joint_zeros.json)",
    )
    parser.add_argument(
        "--allow-uncalibrated-debug",
        action="store_true",
        help="run without joint-zero correction for safe bench diagnostics only",
    )
    add_rig_camera_arguments(
        parser, wrist_intrinsics=True, workspace_camera=True, workspace_intrinsics=True
    )
    add_drop_zone_arguments(parser)
    add_target_chain_arguments(parser)
    parser.add_argument("--episodes", type=int, default=1, help="episodes to run; 0 means continuous")
    parser.add_argument(
        "--rest-every",
        type=int,
        default=10,
        help="completed episodes between cooldowns; 0 disables cooldowns",
    )
    parser.add_argument("--rest-duration", type=float, default=30.0)
    add_operator_alert_arguments(parser)
    parser.add_argument("--target-change-min-distance", type=float, default=0.03)
    parser.add_argument("--target-change-alert-min-seconds", type=float, default=10.0)
    parser.add_argument("--target-change-alert-max-seconds", type=float, default=120.0)
    add_speed_argument(parser)
    parser.add_argument(
        "--recording-format", choices=("video", "dataset", "none"), default="video"
    )
    parser.add_argument("--recording-root", type=Path, default=REPO_ROOT / "episodes")
    parser.add_argument("--dataset-repo-id", default="physical-scripted-v2")
    parser.add_argument(
        "--max-steps", type=int, default=450, help="30 Hz ticks per episode (default: 450)"
    )
    parser.add_argument("--max-localization-steps", type=int, default=60)
    parser.add_argument("--localization-steps-per-search", type=int, default=15)
    parser.add_argument("--planning-attempts", type=int, default=40)
    add_preflight_debug_arguments(parser)
    parser.add_argument(
        "--show-camera-feeds",
        action="store_true",
        help="show the rectified overhead and wrist observations",
    )
    parser.add_argument(
        "--debug-servo",
        action="store_true",
        help="open the wrist servo window and log what it saw, frame by frame",
    )
    add_viewer_argument(parser, help="show the measured arm and localized objects in MuJoCo")
    parser.add_argument("--rng-seed", type=int, default=0)
    add_overhead_recalibration_arguments(parser, drift_checks=True)
    parser.add_argument("--recalibrate-check-min-cooldown", type=float, default=15.0)
    parser.add_argument("--cube-recovery-attempts", type=int, default=3)
    parser.add_argument(
        "--park-speed",
        type=float,
        default=30.0,
        help="maximum arm-joint speed while parking, in degrees/s",
    )
    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject the ranges argparse's own types cannot check."""
    if args.episodes < 0:
        parser.error("--episodes must be non-negative")
    if args.rest_every < 0:
        parser.error("--rest-every must be non-negative")
    if args.rest_duration < 0.0:
        parser.error("--rest-duration must be non-negative")
    if args.target_change_min_distance < 0.0:
        parser.error("--target-change-min-distance must be non-negative")
    if args.target_change_alert_min_seconds <= 0.0:
        parser.error("--target-change-alert-min-seconds must be positive")
    if args.target_change_alert_max_seconds < args.target_change_alert_min_seconds:
        parser.error("--target-change-alert-max-seconds must be at least the minimum")
    if args.recalibrate_check_min_cooldown < 0.0:
        parser.error("--recalibrate-check-min-cooldown must be non-negative")
    if args.recalibrate_drift_mm < 0.0 or args.recalibrate_drift_deg < 0.0:
        parser.error("camera drift limits must be non-negative")
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be in (0, 1]")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.planning_attempts < 1:
        parser.error("--planning-attempts must be at least 1")
    if args.preflight_debug_limit < 1:
        parser.error("--preflight-debug-limit must be at least 1")
    if args.failed_trajectory_limit < 0:
        parser.error("--failed-trajectory-limit must be non-negative")
    if args.park_speed <= 0.0:
        parser.error("--park-speed must be positive")
    if args.cube_recovery_attempts < 1:
        parser.error("--cube-recovery-attempts must be at least 1")
