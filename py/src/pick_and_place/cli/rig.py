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
    pap camera-fps-probe exists to find out.
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


def add_camera_device_argument(
    parser: argparse.ArgumentParser, *, required: bool = False, default: str | None = "0"
) -> None:
    """Add ``--camera`` for a command that opens exactly one camera.

    The rig commands take :func:`add_rig_camera_arguments` instead, which names
    the overhead and wrist cameras together. This is for the two calibration
    commands that point at a single device: what the string may be -- an OpenCV
    index or a device path -- is the same either way.
    """
    parser.add_argument(
        "--camera",
        required=required,
        default=None if required else default,
        help="OpenCV camera index or device path",
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


def add_target_chain_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the two ways to run chained onto scripted targets instead of a plate.

    A chained run places onto a supplied sequence and never resets the scene:
    each placement is where the next episode picks the cube up from, which is
    what lets a hundred episodes run with nobody in the room. Both runners want
    this, so it is declared once here.
    """
    chain = parser.add_mutually_exclusive_group()
    chain.add_argument(
        "--target-chain-seed",
        type=int,
        default=None,
        help="run chained and unattended: draw the run's targets from the chainable "
        "distribution with this seed and place onto them in order, instead of "
        "localizing a physical drop plate between episodes",
    )
    chain.add_argument(
        "--target-sequence",
        type=Path,
        default=None,
        help="run chained onto a pre-drawn sequence: a JSON list of [x, y] points in "
        "workspace metres, checked against the chainable region before the arm "
        "moves because a target that cannot be picked up from strands the run",
    )


def add_joint_zeros_argument(
    parser: argparse.ArgumentParser, *, default: Path | None, help: str
) -> None:
    """Add ``--joint-zeros``, the session calibration the servo readback is mapped through.

    ``default`` differs by command on purpose: the scripted runner reads the
    committed sidecar, while the policy runner requires the file to be named,
    because a learned policy fed uncorrected readback is being shown a different
    arm than the one it was trained on.
    """
    parser.add_argument("--joint-zeros", type=Path, default=default, help=help)


def add_max_joint_speed_argument(
    parser: argparse.ArgumentParser, *, default: float, extra_help: str = ""
) -> None:
    """Add ``--max-joint-speed``, the per-joint velocity cap in deg/s.

    Both commands that take it feed ``runtime.ramp``, so the units and the
    "``<=0`` means no cap" convention are shared -- that convention is the part
    worth declaring once, since a command reading a non-positive value as an
    error rather than as "uncapped" would be wrong in a way nothing catches.

    ``default`` is not shared: parking an arm that may be anywhere ramps at its
    own pace unless told otherwise, while a policy run caps every tick by
    default. ``extra_help`` is where a command adds what else it does with it.
    """
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=default,
        help=f"hard per-joint velocity cap in deg/s; <=0 disables the cap (default: {default:g})"
        + extra_help,
    )
