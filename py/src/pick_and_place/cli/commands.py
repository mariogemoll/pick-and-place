# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The command table ``pap`` dispatches from: names, one-line summaries, and where to look.

**Nothing here imports a command.** The table is data, so ``pap --help`` can
list fifty commands and their summaries without importing torch, lerobot or a
MuJoCo scene -- which the dispatcher would otherwise have to do just to ask each
command what it is called. That is the whole reason this is a table rather than
a registry built by decorators: a decorator has to run, and running it means
importing the module it decorates.

The cost is that a summary lives here rather than beside its command, and can go
stale. ``tests/test_commands.py`` checks the table against the tree instead:
every command resolves, every script is either registered or deliberately not.

Each command names two things the dispatcher imports **after** parsing:

``parser``
    the module exposing ``build_parser()``, and optionally ``validate(parser,
    args)``. It defaults to the script itself, which is right whenever importing
    the script is cheap. Where it is not -- the commands that pull torch or
    lerobot at module scope -- the parser lives in ``pick_and_place.cli.<name>``
    and this field names it, so ``pap <command> --help`` costs an argparse
    import rather than a deep-learning stack.

``script``
    the file exposing ``run(args)``, relative to ``py/scripts``. Imported by
    path rather than by name, because ``scripts/`` is not a package and
    ``scripts/pick_and_place/`` would shadow the real one if it were.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One entry in the ``pap`` tree."""

    name: str
    summary: str
    script: str
    parser: str | None = None

    @property
    def parser_module(self) -> str | None:
        """The importable module holding ``build_parser``, or ``None`` for the script."""
        return self.parser


#: Every command ``pap`` offers, in the order ``pap --help`` lists them.
COMMANDS: tuple[Command, ...] = (
    Command(
        name="eval-policy-sim",
        summary="Score a controller against a frozen scenario manifest.",
        script="eval_policy_sim.py",
        parser="pick_and_place.cli.eval_policy_sim",
    ),
    Command(
        name="select-episodes",
        summary="List a dataset's episodes that pass a success filter.",
        script="select_episodes.py",
    ),
    Command(
        name="combine-datasets",
        summary="Merge several LeRobot datasets into one.",
        script="combine_datasets.py",
    ),
    Command(
        name="consolidate-datasets",
        summary="Merge run directories into one dataset per day.",
        script="consolidate_datasets.py",
    ),
    Command(
        name="split-train-val-episodes",
        summary="Split a dataset's episodes into train and validation sets.",
        script="split_train_val_episodes.py",
    ),
    Command(
        name="keep-successful-episodes",
        summary="Copy a dataset keeping only its successful episodes.",
        script="keep_successful_episodes.py",
    ),
    Command(
        name="export-policy-dataset",
        summary="Export a LeRobot dataset as the image policy's training export.",
        script="export_diffusion_policy_dataset.py",
    ),
    Command(
        name="export-generic-robot",
        summary="Export any robot_descriptions model as a web manifest.",
        script="export_generic_robot.py",
    ),
    Command(
        name="generate-charuco-board",
        summary="Render the printable ChArUco calibration board.",
        script="generate_charuco_board.py",
    ),
    Command(
        name="measure-hand-eye-offset",
        summary="Measure the wrist camera's offset from the gripper.",
        script="measure_hand_eye_offset.py",
    ),
    Command(
        name="convert-dataset-resolution",
        summary="Re-render a dataset's video at another resolution.",
        script="convert_dataset_resolution.py",
    ),
    Command(
        name="merge-evaluation-shards",
        summary="Merge sharded evaluation runs into one result.",
        script="merge_evaluation_shards.py",
    ),
    Command(
        name="compare-policy-evaluations",
        summary="Compare evaluation runs against a baseline.",
        script="compare_policy_evaluations.py",
    ),
    Command(
        name="park-follower",
        summary="Ramp the follower arm to its rest pose and release it.",
        script="park_follower.py",
    ),
    Command(
        name="render-scene-thumbnails",
        summary="Render the initial overhead frame of chosen scenes.",
        script="render_scene_thumbnails.py",
    ),
    Command(
        name="check-overhead-localization",
        summary="Measure whether simulated overhead perception misses by as much as the rig does.",
        script="check_overhead_localization.py",
    ),
    Command(
        name="check-calibration",
        summary="Compare the leader and follower calibrations in the lerobot cache.",
        script="check_calibration.py",
    ),
    Command(
        name="capture-rest-pose",
        summary="Read the arm's current pose and print it as the rest position.",
        script="capture_rest_pose.py",
    ),
    Command(
        name="camera-fps-probe",
        summary="Measure a camera's real frame rate at several resolutions.",
        script="camera_fps_probe.py",
    ),
    Command(
        name="fit-pan-zero",
        summary="Fit the shoulder-pan zero from a hand-eye measurement.",
        script="fit_pan_zero.py",
    ),
    Command(
        name="fit-joint-zeros",
        summary="Fit the arm's joint zeros from hand-eye measurements.",
        script="fit_joint_zeros.py",
    ),
    Command(
        name="fit-sag",
        summary="Fit how far each joint droops from its commanded angle.",
        script="fit_sag.py",
    ),
    Command(
        name="generate-parity-fixtures",
        summary="Regenerate the cross-language parity fixtures.",
        script="generate_parity_fixtures.py",
    ),
    Command(
        name="render-apriltag-textures",
        summary="Render the AprilTag PNG textures the simulated scene needs.",
        script="render_apriltag_textures.py",
    ),
    Command(
        name="generate-apriltags",
        summary="Generate printable AprilTag 41h12 PDFs.",
        script="generate_apriltags.py",
    ),
    Command(
        name="replay-episode",
        summary="Replay a recorded episode in the viewer or to an mp4.",
        script="replay_episode.py",
    ),
    Command(
        name="view-scene",
        summary="Open the composed MuJoCo scene in the viewer.",
        script="view_scene.py",
    ),
    Command(
        name="preview-cameras",
        summary="Serve a browser preview of every attached camera.",
        script="preview_cameras.py",
    ),
    Command(
        name="calibrate-robot-dynamics",
        summary="Fit the follower's actuator dynamics from recorded datasets.",
        script="calibrate_robot_dynamics.py",
    ),
    Command(
        name="generate-scenario-manifest",
        summary="Generate a frozen scenario manifest.",
        script="generate_scenario_manifest.py",
    ),
    Command(
        name="export-episode-rolls",
        summary="Export episode rolls for the browser replay viewer.",
        script="export_episode_rolls.py",
    ),
    Command(
        name="freeze-scenario-rig",
        summary="Rewrite a suite so every scene faces one frozen rig.",
        script="freeze_scenario_rig.py",
    ),
    Command(
        name="measure-cube-visibility",
        summary="Measure how visible the cube is in the policy's own frames.",
        script="measure_cube_visibility.py",
    ),
)

COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}
