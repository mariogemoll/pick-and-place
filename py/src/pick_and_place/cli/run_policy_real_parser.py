# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The hardware policy runner's parser, separate from the run itself.

Kept apart for the reason the evaluator's parser is, and more so: this
command imports mujoco and lerobot at module scope, so asking it what its flags
are cost 1.3 seconds. Here it costs an argparse import.

There is a second reason for this one. ``run_policy_real.py`` is the file still
over the repository's 40 KB ceiling, with a 1,074-line ``main``; the two hundred
lines of flag declarations that moved here are lines it no longer carries.

Nothing here runs anything: :func:`build_parser` declares, :func:`validate`
rejects. Both are importable and testable without an arm attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.cli.policy import (
    add_checkpoint_argument,
    add_device_argument,
    add_flow_image_arguments,
    add_lerobot_arguments,
    add_policy_image_arguments,
    add_save_video_argument,
    add_step_limit_argument,
)
from pick_and_place.cli.rig import (
    add_drop_zone_arguments,
    add_follower_arguments,
    add_joint_zeros_argument,
    add_max_joint_speed_argument,
    add_operator_alert_arguments,
    add_overhead_recalibration_arguments,
    add_rig_camera_arguments,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser


def build_parser(description: str | None = None) -> SuggestingArgumentParser:
    """Return the runner's parser: a shared rig, two controller leaves."""
    # The rig and the run, shared by every leaf so that the two controllers
    # drive the same arm through literally the same declaration. What a
    # checkpoint is and how it is queried belongs to the leaf that has one.
    common = argparse.ArgumentParser(add_help=False)
    parser = common
    add_policy_image_arguments(parser)
    add_follower_arguments(parser)
    add_rig_camera_arguments(parser, workspace_camera=True)
    add_step_limit_argument(parser, forever="Ctrl-C")
    add_joint_zeros_argument(
        parser,
        default=None,
        help=(
            "session joint-zero calibration (config/joint_zeros.json) to map the "
            "servo readback into the model frame before the policy sees it, and "
            "the policy's action back before it is sent. Required for a "
            "sim-trained checkpoint; omit for one fine-tuned on real recordings, "
            "which learned the servo frame directly"
        ),
    )
    parser.add_argument(
        "--tracking-bias-scale",
        type=float,
        default=0.0,
        help=(
            "compensate the fitted servo steady-state bias, so a joint settles on "
            "the commanded angle rather than 2.16 deg (shoulder_lift) away from "
            "it; 1.0 is the measured arm, 0 sends the policy's action verbatim"
        ),
    )
    add_max_joint_speed_argument(
        parser,
        default=10.0,
        extra_help=(
            ". Each tick the command may move at most this far from the arm's measured "
            "pose, so a wild prediction can only ever crawl; lower it (e.g. 3) to go really slow"
        ),
    )
    add_save_video_argument(
        parser,
        help="directory to write <dir>/wrist.mp4 and <dir>/overhead.mp4 with the exact "
        "frames fed to the policy each tick",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        help=(
            "root directory for continuous native-rate MP4s of the whole run; each run "
            "writes into <dir>/<timestamp>/: undistorted full-resolution wrist_live.mp4 "
            "and overhead_live.mp4 (plus workspace_live.mp4 with --workspace-camera) on "
            "a shared clock — unlike --save-video's cropped per-tick policy-input frames"
        ),
    )
    parser.add_argument(
        "--action-log",
        type=Path,
        default=None,
        help=(
            "root directory for per-attempt action logs; each run writes "
            "<dir>/<timestamp>/attempt_NNN.npz with the per-tick state, returned "
            "(ensembled) action, sent command, and every raw predicted chunk, all "
            "in the real frame"
        ),
    )
    parser.add_argument(
        "--record-audio",
        action="store_true",
        help="capture the audio input and mux it into every --record-video MP4",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="sounddevice input name or index (default: system input device)",
    )
    parser.add_argument(
        "--send-substeps",
        type=int,
        default=3,
        help=(
            "sends per policy query. The policy emits setpoints at its own rate "
            "(10 Hz for the flow policies), which leaves the servos one step per "
            "period to chase and nothing in between; splitting it into this many "
            "equal sends spreads the same travel with a fraction of the per-send "
            "jump. 1 sends the undivided step (default: 3, i.e. 30 Hz sends)"
        ),
    )
    parser.add_argument(
        "--measure-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "locate the cube and drop zone overhead and detect success automatically. "
            "Requires the tagged cube, whose pose the overhead camera can read. Pass "
            "--no-measure-scene for a plain blue cube, which has no measurable pose: "
            "the rollout then runs once, unscored, and the operator judges it "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "on success, alert the operator to reset the scene and continue with a "
            "new attempt instead of exiting; without it the run exits on the first "
            "success"
        ),
    )
    parser.add_argument(
        "--attempt-timeout",
        type=float,
        default=20.0,
        help="seconds before an unsuccessful attempt is abandoned and retried from a "
        "fresh randomish start; <=0 disables the timeout (default: 20)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=1.0,
        help="seconds between overhead success scans during an attempt (default: 1)",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=0.04,
        help="cube-to-target xy distance counted as placed, in metres (default: 0.04)",
    )
    parser.add_argument(
        "--place-height-tolerance",
        type=float,
        default=0.02,
        help="how far the cube centre may sit above its resting height and still count "
        "as placed (not carried), in metres (default: 0.02)",
    )
    parser.add_argument(
        "--success-dwell",
        type=float,
        default=1.0,
        help="seconds the cube must stay placed at the target before the placement is "
        "confirmed (default: 1)",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=3.0,
        help="max seconds after the cube is first seen placed before success fires "
        "regardless of arm motion; also caps how long the slow-down is awaited "
        "(default: 3)",
    )
    parser.add_argument(
        "--settle-speed",
        type=float,
        default=5.0,
        help="arm peak joint speed (deg/s) below which it counts as slowed, letting "
        "success fire as soon as the placement dwell is met (default: 5)",
    )
    parser.add_argument(
        "--max-hunt-tries",
        type=int,
        default=5,
        help="pan-around search poses to try while looking for the cube or drop zone "
        "before asking the operator for help (default: 5)",
    )
    add_overhead_recalibration_arguments(parser, drift_checks=True)
    parser.add_argument(
        "--recalibrate-check-interval",
        type=float,
        default=120.0,
        help="minimum seconds between periodic overhead drift checks, run at attempt boundaries "
        "while the arm is at neutral; the run stops if the camera has drifted past the limits "
        "below. <=0 disables (default: 120)",
    )
    add_drop_zone_arguments(parser)
    add_operator_alert_arguments(parser)
    parser = SuggestingArgumentParser(description=description)
    leaves = parser.add_subparsers(dest="controller", required=True, metavar="CONTROLLER")

    lerobot = leaves.add_parser(
        "lerobot",
        parents=[common],
        help="a LeRobot checkpoint (ACT, SmolVLA, pi0.5, ...)",
        description="Run a LeRobot checkpoint on the physical arm, closed-loop.",
    )
    # Default to the checkpoint's own n_action_steps. The 100 that
    # add_lerobot_arguments otherwise supplies matches ACT's chunk and exceeds
    # the 50 that smolvla_base, every SmolVLA fine-tune and pi0.5 carry, so it
    # rejected all three before a single frame was read.
    add_lerobot_arguments(lerobot, n_action_steps_default=None)

    flow_image = leaves.add_parser(
        "flow-image",
        parents=[common],
        help="the image-conditioned flow-matching policy",
        description="Run an image-flow export on the physical arm, closed-loop.",
    )
    add_checkpoint_argument(
        flow_image, default=None, required=True, help="flow-policy checkpoint-*.pt file"
    )
    add_device_argument(flow_image)
    add_flow_image_arguments(flow_image, flow_export_required=True)

    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject what the parser cannot express, before the arm is opened.

    Everything here is decidable from the arguments alone. Whether the flow
    export beside the checkpoint is readable, and whether the loaded model was
    trained at the requested image size, are not -- those are discovered while
    loading and stay runtime failures in the runner.
    """
    if args.workspace_camera is not None and args.record_video is None:
        parser.error("--workspace-camera requires --record-video")
    if args.record_audio and args.record_video is None:
        parser.error("--record-audio requires --record-video")
    if args.send_substeps < 1:
        parser.error(f"--send-substeps must be at least 1, got {args.send_substeps}")
    override = (args.image_height, args.image_width)
    if any(override) and not all(override):
        parser.error("pass both --image-height and --image-width, or neither")
