# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for the physical rig: the arm, its cameras, and the operator.

Every command that touches hardware opens the same devices in the same way, and
they have to agree — a run recorded with one camera assignment and replayed with
another is not the same experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.runtime.overhead_detection import DEFAULT_ALERT_SOUND


def add_follower_arguments(
    parser: argparse.ArgumentParser, *, port: bool = True, port_required: bool = True
) -> None:
    """Add the serial port and calibration id of the SO-101 follower.

    ``port`` is false for a command that names an arm without opening it -- the
    calibration comparison reads the stored file and never touches the serial
    line, so offering it a port would be offering a flag it must then ignore.
    ``port_required`` keeps the flag but makes it optional, for the wrist-camera
    solve, which drives the leader and moves the follower only if it is there.
    """
    if port:
        parser.add_argument(
            "--follower-port",
            required=port_required,
            help="serial port of the SO-101 follower"
            + ("" if port_required else " (optional; omit to run without it)"),
        )
    parser.add_argument(
        "--follower-id", default="folly", help="follower calibration id (default: folly)"
    )


def add_leader_arguments(parser: argparse.ArgumentParser, *, port: bool = True) -> None:
    """Add the serial port and calibration id of the SO-101 leader.

    ``port`` follows the same rule as :func:`add_follower_arguments`.
    """
    if port:
        parser.add_argument(
            "--leader-port", required=True, help="serial port of the SO-101 leader"
        )
    parser.add_argument(
        "--leader-id", default="liddy", help="leader calibration id (default: liddy)"
    )


def add_capture_size_arguments(parser: argparse.ArgumentParser, *, width: int, height: int) -> None:
    """Add the resolution a command asks a camera device for.

    Distinct from the size of what a command *produces*
    (:func:`pick_and_place.cli.common.add_output_size_arguments`): a driver may
    refuse the resolution asked for and hand back another, which is what
    camera_fps_probe.py exists to find out.
    """
    parser.add_argument(
        "--width",
        type=int,
        default=width,
        help=f"capture width to request from the camera (default: {width})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=height,
        help=f"capture height to request from the camera (default: {height})",
    )


def add_rig_camera_arguments(
    parser: argparse.ArgumentParser,
    *,
    wrist_intrinsics: bool = False,
    workspace_camera: bool = False,
    workspace_intrinsics: bool = False,
) -> None:
    """Add the overhead and wrist cameras and the model's overhead camera name.

    The optional flags are opt-in rather than always present, so no command
    accepts a camera or a calibration file it will then ignore.
    """
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera")
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera")
    parser.add_argument(
        "--camera-name", default="overhead_camera", help="overhead camera name in the model"
    )
    parser.add_argument(
        "--overhead-intrinsics",
        type=Path,
        default=None,
        help="overhead camera intrinsics JSON (default: the local sidecar)",
    )
    if wrist_intrinsics:
        parser.add_argument(
            "--wrist-intrinsics",
            type=Path,
            default=None,
            help="wrist camera intrinsics JSON (default: the local sidecar)",
        )
    if workspace_camera:
        parser.add_argument(
            "--workspace-camera",
            default=None,
            help="optional OpenCV index/path of a synchronized workspace camera",
        )
    if workspace_intrinsics:
        parser.add_argument(
            "--workspace-intrinsics",
            type=Path,
            default=None,
            help="workspace camera intrinsics JSON (default: the local sidecar)",
        )


def add_overhead_recalibration_arguments(
    parser: argparse.ArgumentParser, *, drift_checks: bool = False
) -> None:
    """Add the startup solve of the overhead camera extrinsics.

    ``drift_checks`` adds the limits that stop a long run once the camera has
    moved away from where it was solved.
    """
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve the overhead camera extrinsics live from the workspace-frame "
        "AprilTags at startup and refuse to start if the solve is implausible "
        "(--no-recalibrate uses the saved sidecar extrinsics instead)",
    )
    parser.add_argument(
        "--recalibrate-samples",
        type=int,
        default=10,
        help="overhead frames to average per extrinsics solve (default: 10)",
    )
    parser.add_argument(
        "--recalibrate-max-seconds",
        type=float,
        default=15.0,
        help="time budget to gather the solve frames before giving up (default: 15)",
    )
    if drift_checks:
        parser.add_argument(
            "--recalibrate-drift-mm",
            type=float,
            default=10.0,
            help="translation drift from the startup solve that stops the run "
            "(default: 10 mm)",
        )
        parser.add_argument(
            "--recalibrate-drift-deg",
            type=float,
            default=2.0,
            help="rotation drift from the startup solve that stops the run (default: 2 deg)",
        )


def add_operator_alert_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the spoken/audible prompts that ask the operator to intervene."""
    parser.add_argument(
        "--operator-alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play a sound and speak operator alerts on macOS (default: on)",
    )
    parser.add_argument(
        "--alert-sound",
        default=DEFAULT_ALERT_SOUND,
        help=f"sound file played before spoken operator alerts (default: {DEFAULT_ALERT_SOUND})",
    )


def add_drop_zone_arguments(parser: argparse.ArgumentParser) -> None:
    """Add which printed drop-zone square counts as the target."""
    parser.add_argument(
        "--drop-zone-color",
        choices=("black", "white"),
        default="black",
        help="color of the drop-zone square to detect as the target (default: black)",
    )
