<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Project Overview

**pick-and-place** builds and studies a cube pick-and-place task on the
[Standard Open SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100), in
simulation and on real hardware. Its visible output is the pick-and-place page
on <https://mariogemoll.com/pick-and-place>; most of the code is the research
that produced that page's content.

The task: an overhead camera locates a cube on a table, the arm picks it up and
places it on a target position given as an xy coordinate. An analytic planner
solves this directly and generates all training demonstrations; learned
policies are trained to imitate it and scored against it.

## Mandates

These are non-negotiable and override convenience.

- **`SO-ARM100/` is vendored upstream truth.** It is a git submodule. Never
  modify anything inside it. Compose on top of it instead.
- **`third_party/dppo` is vendored upstream too.** Same rule.
- **The collision box values are a hand-tuned asset, not generated output.**
  `py/src/pick_and_place/collision_boxes.py` and
  `wrist_camera_mount_collision_boxes.py` hold numbers that were tuned by hand
  against the real arm. Do not regenerate, round, or "clean up" the numbers.
- **Every source file carries SPDX headers**, enforced in CI:
  - `SPDX-FileCopyrightText: 2026 Mario Gemoll`
  - `SPDX-License-Identifier: 0BSD`
- **No large files.** `scripts/check_files_in_repo.sh` fails the build above
  40 KB per file, with four documented exceptions (`executor.py`,
  `trajectory.py`, `pick_and_place/real.py`, `run_policy_real.py`). Those
  ceilings are a ratchet — do not raise them to land new code.
- **Never commit datasets, checkpoints, renders, or recordings.** See
  [Local and generated data](#local-and-generated-data).

## Environment

Use Python 3.13, as CI does. The dependency graph is tightly constrained —
`lerobot==0.5.1` pins `transformers==5.3.0` — so newer interpreters are
untested and best avoided.

```sh
uv python install 3.13
uv venv --python 3.13 ~/venvs/pick-and-place
VIRTUAL_ENV=~/venvs/pick-and-place uv pip install -e py --group dev
```

Prefer a plain venv over pyenv inside a container. Use `uv pip install` to
populate it, then call `python` directly rather than `uv run`.

System packages needed for a headless Linux box:

```sh
sudo apt-get install -y linux-libc-dev build-essential libegl1 libgl1 libopengl0
```

`linux-libc-dev` is not optional on arm64: `lerobot` pulls in `pynput` which
pulls in `evdev`, which ships no arm64 wheel and compiles against
`linux/input.h`. (`evdev-binary` is x86-64 only, so it is not a way out.)

Set `MUJOCO_GL=egl` for headless rendering. On macOS, scripts that open a
MuJoCo viewer need `mjpython` rather than `python`; most such scripts offer a
`--no-viewer` flag to avoid it.

### `PAP_DATA_ROOT`

Datasets, checkpoints, renders, and reports live under one directory outside
the repository, named by `PAP_DATA_ROOT`:

```sh
export PAP_DATA_ROOT=~/pick-and-place-data   # holds datasets/ and outputs/
```

Scripts resolve it lazily, only when a default is actually needed, so an
explicit path on the command line works whether or not the variable is set. A
script that needs it and cannot find it fails immediately with a message naming
the variable, rather than silently writing into the source tree.

`pick_and_place.paths` is the only place that reads it. Never reintroduce a
path default that points inside the repository.

## Standard commands

```sh
# Python — from py/
python -m ruff check .
MUJOCO_GL=egl python -m pytest

# TypeScript — from ts/
pnpm i
pnpm test          # vitest
pnpm typecheck     # tsc --noEmit
pnpm lint          # eslint + stylelint
pnpm build         # tsc && vite build
pnpm dev           # vite dev server

# Repository
scripts/check-license-headers.sh
scripts/check_files_in_repo.sh
```

Use pnpm 10, as CI does. pnpm 11 turns `sharp`'s ignored build script into a
hard error and scaffolds a stray `ts/pnpm-workspace.yaml` on every run;
`npx -y pnpm@10 <cmd>` is a working escape hatch. There is no `packageManager`
pin and `pnpm-lock.yaml` is gitignored, so dependency resolution is not
currently reproducible.

### The TypeScript tests need a generated fixture

Five test files read `ts/public/so101.json`, which is **not** in the
repository. Generate it before running them:

```sh
cd py && MUJOCO_GL=egl python -m pick_and_place.export -o ../ts/public/so101.xml
```

That command writes both `so101.xml` and `so101.json`; only the `.json` is
needed for tests. CI gets this by accident — its step reads as a smoke test of
the exporter and deletes the `.xml` afterwards, leaving the `.json` behind.

## Repository map

| Directory | Contents |
| --- | --- |
| `SO-ARM100/` | Vendored hardware submodule: CAD, STL, URDF, MJCF, BOM. |
| `py/` | The `pick_and_place` package (85 modules), 84 CLI scripts, 42 test files. Simulation, real-robot control, calibration, datasets, policies. |
| `ts/` | Vite + Three.js browser app: the visualizations embedded in the web page. |
| `mesh_optimization/` | Standalone Python subproject that decimates high-poly STL into web-ready GLB. |
| `scripts/` | Repository-level shell/TS tooling: license headers, file-size check, mesh pipeline, remote-GPU job scripts. |
| `config/` | Committed configuration: evaluation manifests, training configs, fitted robot dynamics. Camera calibration JSON lives here but is gitignored. |
| `stl/` | Committed printable geometry for the physical workspace frame. |
| `assets/` | Generated AprilTag textures. Gitignored. |
| `third_party/dppo` | Vendored DPPO submodule, used for its diffusion pre-training agent. |

### What the Python package does

`spec/` sits under all of it: the physical facts and contracts every strand
has to agree on — the cube's size and face tag ids, the drop-zone and corner
plate sizes, the workspace frame pose, the joint names and their order, and
the `PolicyController` boundary. It imports nothing else in the package and
needs no heavy dependency, which is what lets the simulator and the detector
agree by construction instead of by importing into each other's internals.
Then ten strands, roughly in dependency order:

1. **Model composition** — `builder`, `scene`, `materials`, `collision_boxes`,
   `wrist_camera`, `camera_module`, `workspace_overlays`, `paper_target_marker`,
   `derive_kinematics`, `export`. Loads the
   stock MJCF from `SO-ARM100/` with `MjSpec` and replaces full-mesh collision
   geoms with the hand-tuned box model. `python -m pick_and_place.export`
   writes standalone MJCF plus a web manifest for the browser app and external
   consumers.
2. **Scripted policy** — `trajectory`, `task_phases`, `scripted_policy`,
   `geometry`, `ik`, `kinematics`, `transforms`, `workspace_bounds`,
   `episode_sampling`, `episodes`, `scenario_sampling`. The analytic planner: grasp-pose search, an 8-phase
   trajectory, preflight validation. Generates every demonstration and is the
   expert baseline.
3. **Policy evaluation** — `policy_evaluation`, `policy_controllers`,
   `policy_sim`, `policy`. Frozen scenario manifests in `config/evaluation/`, a
   success oracle, and a controller boundary that makes scripted, ACT, and
   Diffusion Policy interchangeable.
4. **Sim dataset generation** — `sim_recorder`, `sim_dataset_staging`,
   `domain_randomization`, `render_randomization`, `episode_rerender`,
   `miscalibration`, `dataset_metadata`, `dataset_subset`. Two pipelines: direct
   recording, and a two-pass scheme that records ground truth once and
   re-renders it under any scene appearance.
5. **Diffusion Policy training** — `diffusion_policy_pretrain`,
   `diffusion_policy_dataset`, `diffusion_policy_client`. Trains and serves the
   current best policy. ACT and SmolVLA are *evaluated* here but trained
   externally via the `lerobot` CLI.
6. **Real-robot control** — `executor`, `follower`, `joint_frames`, `physical_rig`,
   `physical_collection`, `safety`, `session_calibration`, `episode_loop`,
   `recording`, `recorder`, `episode_video`, `policy_real`, `policy_recording`.
   Hardware lifecycle, control loop, recording, and recovery.
7. **Perception** — `cube_detection`, `detector_process`, `overhead_detection`,
   `overhead_localization`, `paper_detection`, `camera_projection`,
   `visual_servo`, `image_rectify`.
   AprilTag-based cube and target localization, plus the wrist-camera descent
   servo that absorbs residual sim/real mismatch.
8. **Camera calibration** — `camera_intrinsics`, `camera_extrinsics`,
   `cam_align_solve`, `camera_calibration_export`, `camera_compare`,
   `camera_pose_envelope`, `environment`, `workspace_alignment`.
9. **Sim-to-real measurement** — `joint_zero_fit`, `robot_dynamics`,
   `miscalibration`, `scene_visibility`, `scene_appearance`,
   `background_panorama`. Measures each sim/real gap from recorded episodes,
   corrects the predictable part, randomizes over the residual.
10. **DPPO RL fine-tuning** — `dppo_rl/`. Fine-tuning the pretrained Diffusion
    Policy with PPO. **This did not work**: no configuration beat the
    pretrained policy in a paired evaluation. Kept for a future attempt; do not
    treat it as a working path.

### Script categories

`py/scripts/` holds more code than the package does. Broadly:

- **Run the task** — `pick_and_place/{sim,real,record_sim,record_teleop,finalize_sim_dataset}.py`
- **Run a policy** — `run_policy_{sim,real}.py`, `eval_policy_sim.py`,
  `eval_scripted_parallel.py`, `compare_policy_evaluations.py`,
  `generate_scenario_manifest.py`
- **Datasets** — `combine_datasets.py`, `consolidate_datasets.py`,
  `split_train_val_episodes.py`, `convert_dataset_resolution.py`,
  `keep_successful_episodes.py`, `select_episodes.py`,
  `export_diffusion_policy_dataset.py`, `rerender_episodes.py`
- **Calibration** — `calibrate_camera_intrinsics.py`, `calibrate_joint_zeros.py`,
  `calibrate_robot_dynamics.py`, `wrist_cam_align_solve.py`,
  `generate_charuco_board.py`, `export_camera_calibrations.py`
- **Sim-to-real measurement** — `export_sim_real_pairs.py`,
  `measure_hand_eye_offset.py`, `fit_{pan_zero,joint_zeros,sag}.py`,
  `probe_camera_pose_envelope.py`
- **Web assets** — `export_generic_robot.py`, `export_episode_rolls.py`,
  `distill_grasp_policy.py`, `render_apriltag_textures.py`
- **Viewers and diagnostics** — `view_*.py`, `replay_*.py`, `showcamfeed*.py`,
  `diagnose_cube_tracking.py`
- **Figures** — `generate_architecture_figure.py`, `plot_*.py`,
  `make_*_grid.py`, `render_scene_thumbnails.py`

Scripts should parse arguments and delegate. Stable algorithms, file formats,
and calibration logic belong in the package even when a CLI is the only caller.

## Mesh pipeline

Two mandatory steps turn raw hardware STL into web-ready assets:

```sh
python mesh_optimization/scripts/simplify_meshes.py   # STL -> intermediary-glb/
cd ts && pnpm run optimize-meshes                     # -> ts/public/so101_assets/
```

`scripts/convert_meshes.sh [dst_dir]` runs both and copies the result.

## Local and generated data

`ts/public/` does not exist in a clean checkout — every file in it is generated
and individually gitignored by `ts/.gitignore`: the robot web manifests, the
optimized mesh bundles, and the `episodes/` rolls the replay viewer plays back.

Gitignored, machine-local, and sometimes very large:

| Path | Contents |
| --- | --- |
| `docs/` | Working notes. **Not version controlled** — treat as a lab notebook, not documentation. `docs/README.md` indexes it. |
| `ts/public/` | All generated web assets. |
| `datasets/` | Recorded and simulated LeRobot datasets. Tens of GB. |
| `outputs/`, `output/` | Training runs, checkpoints, diagnostics. |
| `intermediary-glb/`, `dist_assets/` | Mesh pipeline intermediates. |
| `config/camera_{intrinsics,extrinsics}/*.json` | Measured per-rig calibration. |

Because `docs/` is not committed, nothing in the repository may depend on it.
Do not add code comments that reference a `docs/` file; write the rationale
into the comment itself so it stands alone.

Scripts must not default to writing inside the repository. Resolve a default
through `pick_and_place.paths` instead — see [`PAP_DATA_ROOT`](#pap_data_root).
The in-tree locations above are ignored so that pre-existing local data does not
surface in `git status`, not because they are still a valid place to write.

## Conventions

- **Style** — Python: PEP 8 via Ruff, 100-column lines. TypeScript: strict,
  type-aware ESLint plus Prettier conventions and Stylelint for CSS.
- **Prefer functions over classes**, small independent units, and unit tests
  alongside them.
- **Be strict about types.** This is greenfield, single-owner code.
- **Do not add backward-compatibility shims.** There are no external consumers.
  Change the call sites and pick the cleanest design.
- **American English** in prose and identifiers.
- **`diffusion_policy_*` is the working policy; `dppo_*` is the failed RL
  experiment.** Spell the prefix out — never `dp_`. `DPPO` in prose means
  upstream `third_party/dppo` or the RL fine-tuning strand, nothing else.

## Known rough edges

Recorded here so they are not mistaken for intentional design:

- `executor.py`, `trajectory.py`, `pick_and_place/real.py` and
  `run_policy_real.py` are oversized and combine too many responsibilities.
- Scripts hold more code than the package (~27k lines against ~22k).
- No dependency lockfiles are committed for either language.
- Python and TypeScript reimplement the same kinematics, grasp selection, and
  trajectory logic with no cross-language parity fixtures.
- The browser entry point eagerly imports every visualization, producing an
  ~870 KB main chunk.
