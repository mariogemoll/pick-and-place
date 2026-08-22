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

``module``
    the module exposing ``run(args)``, imported by name.

``script``
    what a command not yet moved into the package names instead: the file
    exposing ``run(args)``, relative to ``py/scripts``, imported by path
    because ``scripts/`` is not a package and ``scripts/pick_and_place/``
    would shadow the real one if it were. Every command holds exactly one of
    these two, and ``script`` goes away with the last one that needs it.

``typed_config``
    true for the one command that has no ``ArgumentParser`` to expose. The
    trainer takes a dataclass config through ``tyro`` so a run can be written
    out and read back with ``--config``; its parser module offers
    ``parse_arguments(argv)`` instead of ``build_parser()``. One command in
    fifty carrying its own idiom is worth a field here -- hiding it behind a
    shim that pretended to be argparse would not be.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """One entry in the ``pap`` tree."""

    name: str
    summary: str
    script: str = ""
    module: str = ""
    parser: str | None = None
    typed_config: bool = False
    group: str = ""

    def __post_init__(self) -> None:
        if bool(self.script) == bool(self.module):
            raise ValueError(f"{self.name} must name either a script or a module, not both")

    @property
    def parser_module(self) -> str | None:
        """The importable module holding ``build_parser``, or ``None`` for the script."""
        return self.parser


#: Every command ``pap`` offers, grouped the way ``pap --help`` lists them.
#: The groups follow the script categories in ``pick-and-place``'s AGENTS.md,
#: because a flat list of forty-eight names is not something anyone reads.
COMMANDS: tuple[Command, ...] = (
    # Run a policy
    Command(
        name="run-policy-sim",
        summary="Run a policy in the sim, closed-loop.",
        module="pick_and_place.cli.run_policy_sim",
        group="Run a policy",
    ),
    Command(
        name="run-policy-real",
        summary="Run a policy on the physical arm, closed-loop.",
        module="pick_and_place.cli.run_policy_real",
        parser="pick_and_place.cli.run_policy_real_parser",
        group="Run a policy",
    ),
    Command(
        name="run-flow-image-sim",
        summary="Run the image-conditioned flow policy over a stream of scenes.",
        module="pick_and_place.cli.run_flow_image_policy_sim",
        parser="pick_and_place.cli.run_flow_image_policy_sim_parser",
        group="Run a policy",
    ),
    Command(
        name="run-scripted-real",
        summary="Run the scripted expert on the rig.",
        module="pick_and_place.cli.run_scripted_real",
        group="Run a policy",
    ),
    # Score a policy
    Command(
        name="eval-policy-sim",
        summary="Score a controller against a frozen scenario manifest.",
        module="pick_and_place.cli.eval_policy_sim",
        parser="pick_and_place.cli.eval_policy_sim_parser",
        group="Score a policy",
    ),
    Command(
        name="eval-scripted-parallel",
        summary="Score the expert over a manifest across worker processes.",
        module="pick_and_place.cli.eval_scripted_parallel",
        group="Score a policy",
    ),
    Command(
        name="merge-evaluation-shards",
        summary="Merge sharded evaluation runs into one result.",
        module="pick_and_place.cli.merge_evaluation_shards",
        group="Score a policy",
    ),
    Command(
        name="compare-policy-evaluations",
        summary="Compare evaluation runs against a baseline.",
        module="pick_and_place.cli.compare_policy_evaluations",
        group="Score a policy",
    ),
    Command(
        name="generate-scenario-manifest",
        summary="Generate a frozen scenario manifest.",
        module="pick_and_place.cli.generate_scenario_manifest",
        group="Score a policy",
    ),
    Command(
        name="freeze-scenario-rig",
        summary="Rewrite a suite so every scene faces one frozen rig.",
        module="pick_and_place.cli.freeze_scenario_rig",
        group="Score a policy",
    ),
    # Train
    Command(
        name="train-flow-image",
        summary="Train the image-conditioned flow-matching policy.",
        module="pick_and_place.cli.train_flow_image_policy",
        parser="pick_and_place.cli.train_flow_image_policy_parser",
        typed_config=True,
        group="Train",
    ),
    Command(
        name="export-policy-dataset",
        summary="Export a LeRobot dataset as the image policy's training export.",
        module="pick_and_place.cli.export_diffusion_policy_dataset",
        group="Train",
    ),
    Command(
        name="diagnose-flow-image-policy",
        summary="Report an image flow policy's open-loop action error.",
        module="pick_and_place.cli.diagnose_flow_image_policy",
        parser="pick_and_place.cli.diagnose_flow_image_policy_parser",
        group="Train",
    ),
    # Record and shape datasets
    Command(
        name="record-sim",
        summary="Record scripted demonstration episodes in the simulator.",
        module="pick_and_place.cli.record_sim",
        group="Record and shape datasets",
    ),
    Command(
        name="finalize-sim-dataset",
        summary="Merge staged sim episodes into a finished dataset.",
        module="pick_and_place.cli.finalize_sim_dataset",
        group="Record and shape datasets",
    ),
    Command(
        name="combine-datasets",
        summary="Merge several LeRobot datasets into one.",
        module="pick_and_place.cli.combine_datasets",
        group="Record and shape datasets",
    ),
    Command(
        name="consolidate-datasets",
        summary="Merge run directories into one dataset per day.",
        module="pick_and_place.cli.consolidate_datasets",
        group="Record and shape datasets",
    ),
    Command(
        name="convert-dataset-resolution",
        summary="Re-render a dataset's video at another resolution.",
        module="pick_and_place.cli.convert_dataset_resolution",
        group="Record and shape datasets",
    ),
    Command(
        name="keep-successful-episodes",
        summary="Copy a dataset keeping only its successful episodes.",
        module="pick_and_place.cli.keep_successful_episodes",
        group="Record and shape datasets",
    ),
    Command(
        name="select-episodes",
        summary="List a dataset's episodes that pass a success filter.",
        module="pick_and_place.cli.select_episodes",
        group="Record and shape datasets",
    ),
    Command(
        name="split-train-val-episodes",
        summary="Split a dataset's episodes into train and validation sets.",
        module="pick_and_place.cli.split_train_val_episodes",
        group="Record and shape datasets",
    ),
    # Calibrate the rig
    Command(
        name="calibrate-camera-intrinsics",
        summary="Solve a camera's intrinsics against the ChArUco board.",
        module="pick_and_place.cli.calibrate_camera_intrinsics",
        group="Calibrate the rig",
    ),
    Command(
        name="calibrate-joint-zeros",
        summary="Measure the arm's joint zeros at the start of a session.",
        module="pick_and_place.cli.calibrate_joint_zeros",
        group="Calibrate the rig",
    ),
    Command(
        name="calibrate-robot-dynamics",
        summary="Fit the follower's actuator dynamics from recorded datasets.",
        module="pick_and_place.cli.calibrate_robot_dynamics",
        group="Calibrate the rig",
    ),
    Command(
        name="wrist-cam-align-solve",
        summary="Solve the wrist camera's alignment against the workspace tags.",
        module="pick_and_place.cli.wrist_cam_align_solve",
        group="Calibrate the rig",
    ),
    Command(
        name="generate-charuco-board",
        summary="Render the printable ChArUco calibration board.",
        module="pick_and_place.cli.generate_charuco_board",
        group="Calibrate the rig",
    ),
    Command(
        name="export-camera-calibrations",
        summary="Export generic camera calibration JSON for recorded datasets.",
        module="pick_and_place.cli.export_camera_calibrations",
        group="Calibrate the rig",
    ),
    Command(
        name="measure-hand-eye-offset",
        summary="Measure the wrist camera's offset from the gripper.",
        module="pick_and_place.cli.measure_hand_eye_offset",
        group="Calibrate the rig",
    ),
    Command(
        name="fit-pan-zero",
        summary="Fit the shoulder-pan zero from a hand-eye measurement.",
        module="pick_and_place.cli.fit_pan_zero",
        group="Calibrate the rig",
    ),
    Command(
        name="fit-joint-zeros",
        summary="Fit the arm's joint zeros from hand-eye measurements.",
        module="pick_and_place.cli.fit_joint_zeros",
        group="Calibrate the rig",
    ),
    Command(
        name="fit-sag",
        summary="Fit how far each joint droops from its commanded angle.",
        module="pick_and_place.cli.fit_sag",
        group="Calibrate the rig",
    ),
    Command(
        name="check-calibration",
        summary="Compare the leader and follower calibrations in the lerobot cache.",
        module="pick_and_place.cli.check_calibration",
        group="Calibrate the rig",
    ),
    Command(
        name="capture-rest-pose",
        summary="Read the arm's current pose and print it as the rest position.",
        module="pick_and_place.cli.capture_rest_pose",
        group="Calibrate the rig",
    ),
    Command(
        name="park-follower",
        summary="Ramp the follower arm to its rest pose and release it.",
        module="pick_and_place.cli.park_follower",
        group="Calibrate the rig",
    ),
    # What the policy can see
    Command(
        name="measure-cube-visibility",
        summary="Measure how visible the cube is in the policy's own frames.",
        module="pick_and_place.cli.measure_cube_visibility",
        group="What the policy can see",
    ),
    Command(
        name="measure-episode-visibility",
        summary="Measure object visibility across staged episodes.",
        module="pick_and_place.cli.measure_episode_visibility",
        group="What the policy can see",
    ),
    Command(
        name="check-overhead-localization",
        summary="Measure whether simulated overhead perception misses by as much as the rig does.",
        module="pick_and_place.cli.check_overhead_localization",
        group="What the policy can see",
    ),
    # Web and print assets
    Command(
        name="export-generic-robot",
        summary="Export any robot_descriptions model as a web manifest.",
        module="pick_and_place.cli.export_generic_robot",
        group="Web and print assets",
    ),
    Command(
        name="export-episode-rolls",
        summary="Export episode rolls for the browser replay viewer.",
        module="pick_and_place.cli.export_episode_rolls",
        group="Web and print assets",
    ),
    Command(
        name="render-apriltag-textures",
        summary="Render the AprilTag PNG textures the simulated scene needs.",
        module="pick_and_place.cli.render_apriltag_textures",
        group="Web and print assets",
    ),
    Command(
        name="generate-apriltags",
        summary="Generate printable AprilTag 41h12 PDFs.",
        module="pick_and_place.cli.generate_apriltags",
        group="Web and print assets",
    ),
    Command(
        name="render-scene-thumbnails",
        summary="Render the initial overhead frame of chosen scenes.",
        module="pick_and_place.cli.render_scene_thumbnails",
        group="Web and print assets",
    ),
    Command(
        name="generate-parity-fixtures",
        summary="Regenerate the cross-language parity fixtures.",
        module="pick_and_place.cli.generate_parity_fixtures",
        group="Web and print assets",
    ),
    # Viewers and probes
    Command(
        name="view-scene",
        summary="Open the composed MuJoCo scene in the viewer.",
        module="pick_and_place.cli.view_scene",
        group="Viewers and probes",
    ),
    Command(
        name="replay-episode",
        summary="Replay a recorded episode in the viewer or to an mp4.",
        module="pick_and_place.cli.replay_episode",
        group="Viewers and probes",
    ),
    Command(
        name="preview-cameras",
        summary="Serve a browser preview of every attached camera.",
        module="pick_and_place.cli.preview_cameras",
        group="Viewers and probes",
    ),
    Command(
        name="camera-fps-probe",
        summary="Measure a camera's real frame rate at several resolutions.",
        module="pick_and_place.cli.camera_fps_probe",
        group="Viewers and probes",
    ),
    Command(
        name="showcamfeed",
        summary="Show one camera's live feed in a window.",
        module="pick_and_place.cli.showcamfeed",
        group="Viewers and probes",
    ),
    Command(
        name="showcamfeeds",
        summary="Show every attached camera's live feed at once.",
        script="showcamfeeds.py",
        group="Viewers and probes",
    ),
)

COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}
