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
  `py/src/pick_and_place/core/collision_boxes.py` and
  `wrist_camera_mount_collision_boxes.py` hold numbers that were tuned by hand
  against the real arm. Do not regenerate, round, or "clean up" the numbers.
- **Every source file carries SPDX headers**, enforced in CI:
  - `SPDX-FileCopyrightText: 2026 Mario Gemoll`
  - `SPDX-License-Identifier: 0BSD`
- **No large files.** `scripts/check_files_in_repo.sh` fails the build above
  40 KB per file. One file in the tree is still over it —
  `scripts/run_policy_real.py`. Those ceilings are a ratchet: lower them as the
  files shrink, never raise them to land new code. The check walks the whole
  history, so paths that have since shrunk or moved stay listed at the ceiling
  their largest historical blob needs — `runtime/executor.py` is 21 KB now but
  keeps a 61 KB entry, because a 60.7 KB blob of it is still in the history.
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

`pick_and_place.core.paths` is the only place that reads it. Never reintroduce a
path default that points inside the repository.

## Standard commands

```sh
# Python — from py/
python -m ruff check .
MUJOCO_GL=egl python -m pytest
python scripts/check_package_layering.py

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
cd py && MUJOCO_GL=egl python -m pick_and_place.sim.export -o ../ts/public/so101.xml
```

That command writes both `so101.xml` and `so101.json`; only the `.json` is
needed for tests. CI gets this by accident — its step reads as a smoke test of
the exporter and deletes the `.xml` afterwards, leaving the `.json` behind.

### The simulator needs generated AprilTag textures

`assets/apriltags/textures/` is **not** in the repository — the textures are
renders, and renders are not committed. A fresh clone therefore fails to compile
any MuJoCo scene:

```
ValueError: Error: Error opening file
  '.../assets/apriltags/textures/tagStandard41h12_00014_60x60mm_tag40mm.png'
```

Generate all fourteen before running anything that builds a scene:

```sh
MUJOCO_GL=egl python py/scripts/render_apriltag_textures.py --all-defaults
```

The output is deterministic, so a regenerated set matches any other machine's.
`vast_pap_provision.sh` runs this, so rented pods are covered.

**This hides until the worst moment.** Training never compiles a scene — it
reads the dataset — so a missing texture set costs nothing for hours and then
fails the first evaluation, after the GPU time is already spent. It also decides
the cube's appearance, so a run evaluated without regenerating them would not be
scoring the scene it trained on.

### Cross-language parity

`fixtures/parity/` is the shared oracle for the logic both languages implement:
the arm's kinematics, the grasp transforms, the closed-form IK, the forward
kinematics and the canonical grasp search. Python writes the fixtures,
`py/tests/test_parity.py` fails when Python stops reproducing them, and the
tests in `ts/src/parity/` fail when TypeScript does.

**Python is the source of truth.** When a planner change makes
`test_parity.py` fail, regenerate:

```sh
cd py && MUJOCO_GL=egl python scripts/generate_parity_fixtures.py
```

then expect the TypeScript tests to fail until that side follows. Review the
fixture diff; never regenerate to silence a failure you have not explained.
`fixtures/parity/README.md` covers what the files pin and what is deliberately
left out.

## Repository map

| Directory | Contents |
| --- | --- |
| `SO-ARM100/` | Vendored hardware submodule: CAD, STL, URDF, MJCF, BOM. |
| `py/` | The `pick_and_place` package (100 modules in 13 subpackages), 85 CLI scripts, 48 test files. Simulation, real-robot control, calibration, datasets, policies. |
| `ts/` | Vite + Three.js browser app: the visualizations embedded in the web page. |
| `mesh_optimization/` | Standalone Python subproject that decimates high-poly STL into web-ready GLB. |
| `scripts/` | Repository-level shell/TS tooling: license headers, file-size check, mesh pipeline, remote-GPU job scripts. |
| `config/` | Committed configuration: evaluation manifests, training configs, fitted robot dynamics. Camera calibration JSON lives here but is gitignored. |
| `stl/` | Committed printable geometry for the physical workspace frame. |
| `fixtures/` | Committed cross-language test fixtures. `parity/` holds the shared Python/TypeScript oracle; see its `README.md`. |
| `assets/` | Generated AprilTag textures. Gitignored. |
| `third_party/dppo` | Vendored DPPO submodule, used for its diffusion pre-training agent. |

### How the Python package is laid out

**The package is a fan, not a stack.** `perception` needs OpenCV, `sim` needs
MuJoCo, `hardware` needs lerobot, `policies` needs Torch — and none of them
needs another. They are siblings on a shared foundation, and they meet only
above, where work genuinely combines capabilities.

| Tier | Packages | Rule |
| --- | --- | --- |
| Foundation | `spec`, `core` | `spec` imports nothing else in the package; `core` imports only `spec`. |
| Capability branches | `planning`, `perception`, `sim`, `hardware`, `data`, `policies` | Each owns one heavy dependency. **No branch may import another.** |
| Convergence | `runtime`, `calibration`, `analysis`, `cli` | May import anything, including each other. Nothing below them may import them. |

`scripts/check_package_layering.py` enforces this in CI. When a module needs
two capabilities it belongs in the convergence tier by construction; when it
reaches sideways for a *fact* or a *contract*, that fact belongs in `spec`.
`dppo_rl/` and `dsrl/`, the two RL strands, sit above everything and are exempt.

- **`spec/`** — the physical facts and the contracts every branch agrees on:
  the cube's size and face tag ids, the drop-zone and corner plate sizes, the
  workspace frame pose, the joint names and their order, the rig's control rate
  (`CONTROL_HZ`), the camera modules' nominal optics, and the `PolicyController`
  boundary. This is what lets the simulator and the detector agree by
  construction rather than by importing into each other's internals.
- **`core/`** — pure computation over the spec: `geometry`, `transforms`,
  `rotations`, `ik`, `kinematics`, `workspace_bounds`, `joint_frames` (sim↔real
  conversions and the joint-limit clamp), `image_ops`, `miscalibration`,
  `robot_dynamics`, `camera_calibration` (the rig's measured calibration files),
  `paths`.
- **`planning/`** — the analytic planner, which generates every demonstration
  and is the expert baseline: `motion` (interpolation, easing, how long a move
  takes), `grasp` (where to take hold), `carry` (getting the cube across),
  `trajectory` (the eight phases assembled), `replan` (resuming from a
  checkpoint), `visual_servo`, and the declared reset distribution
  (`episode_sampling`, `scenario_sampling`).
- **`perception/`** — AprilTag cube and drop-zone localization:
  `cube_detection`, `paper_detection`, `overhead_localization`,
  `detector_process`, `image_rectify`.
- **`sim/`** — composing and randomizing the MuJoCo scene: `builder`, `scene`,
  `model` (compile a runnable model and move things in it), `collisions`,
  `environment`, `materials`, `wrist_camera`, `camera_module`,
  `workspace_overlays`, `paper_target_marker`, `frame_tags`,
  `derive_kinematics`, `domain_randomization`, `render_randomization`,
  `camera_pose_envelope`, `camera_extrinsics`, `export`. Loads the stock MJCF
  from `SO-ARM100/` with `MjSpec` and replaces full-mesh collision geoms with
  the hand-tuned box model; `python -m pick_and_place.sim.export` writes
  standalone MJCF plus a web manifest.
- **`hardware/`** — the physical arm: `follower`, `physical_rig`,
  `physical_collection`, `joint_zero_fit`.
- **`data/`** — recording and datasets: `recording`, `recorder`,
  `recording_config` (what one recording run is: the scene it draws, its frame
  sizes, where it lands), `dataset_metadata`, `dataset_subset`,
  `sim_dataset_staging`, `diffusion_policy_dataset`.
- **`policies/`** — controller implementations and the contract they are scored
  against: `policy_controllers`, `policy`, `policy_evaluation` (frozen scenario
  manifests in `config/evaluation/` and a success oracle),
  `diffusion_policy_pretrain`, `diffusion_policy_client`, and the state-only
  flow-matching policy (`flow_matching`, `flow_policy`,
  `diffusion_policy_unet`). ACT and SmolVLA are *evaluated* here but trained
  externally via the `lerobot` CLI.
- **`runtime/`** — running an episode. `executor` orchestrates one: it opens the
  cameras, ramps the arm onto the start pose, and then alternates between
  `phase_playback` (the tick loop — evaluate the phase, step physics, command
  the servos, read back) and `checkpoint` (after each phase, replan from
  measured state, or fly straight on where a checkpoint would do more harm than
  good). `wrist_servo` runs the descent's cube detector on its own thread and
  `descent` folds its estimates back into the running phase; `tick_recorder`
  turns the run into dataset rows, one per control tick.

  `sim_recorder` is the same episode with no arm in it, and is built from the
  matching set: `sim_phase_playback` (the tick loop, capturing each row *before*
  it commands it), `sim_wrist_servo` (render the wrist camera, detect the cube
  in it — inline, not on a thread, which is what keeps a recorded episode a pure
  function of its seed), `sim_tick_recorder` (one dataset row per tick, plus its
  phase spans), and `wrist_mixed_view` (the true and believed wrist views
  blended, for watching the servo converge). `believed_frame` is what the two
  worlds meet through: with a miscalibration draw, commands and recorded rows
  live in the believed frame while physics runs the true one. `checkpoint` and
  `descent` are shared with the hardware path, so both agree by construction on
  which phase boundaries replan.

  Around those: `episodes` (sample one that runs clean), `preflight` (vet a
  trajectory under live physics), `frame_reader` (one background thread per
  camera, holding only the newest frame), `ramp` (ease the arm onto a pose),
  `scripted_policy`, `episode_rerender`, `policy_sim`, `policy_real`,
  `overhead_detection`, `episode_loop`, `training_scenes`.
- **`calibration/`** — solving the rig by rendering the scene and comparing it
  to a real image: `cam_align_solve`, `camera_compare`,
  `camera_calibration_export`, `session_calibration`.
- **`analysis/`** — reports about recorded runs and about the scene:
  `episode_video`, `policy_recording`, `scene_visibility`.
- **`cli/`** — the argument groups the scripts compose their parsers from, one
  per subsystem: `policy` (controller choice and the Diffusion Policy server),
  `rig` (follower, cameras, recalibration, operator alerts), `scene` (cube
  pinning, render size, appearance, preflight diagnostics), `dataset`. A flag
  two commands share is declared once, here, not agreed by hand in each.
- **`dppo_rl/`** — fine-tuning the pretrained Diffusion Policy with PPO. **Works
  as a train-and-select procedure on the recovery base, not as a reliable
  optimizer**: across six seeds (2026-08-08) there is no average effect at any
  fixed iteration, but four of six produced a significantly-better checkpoint at
  seed-specific times, and the oracle-selected winner (seed 42, itr 60) validated
  at **0.746 vs 0.684/0.674 for the recovery/absolute bases** on 512 scenes
  untouched by training, selection, or prior scoring (McNemar p = 0.0032/0.0039)
  — the strongest policy in the project. Two preconditions, both load-bearing:
  the braked launcher defaults (zero collapses in six runs; the pre-brake
  configuration collapsed twelve times), and a base policy whose failures are
  recoverable — on the no-retry absolute base the same configuration is provably
  flat. **The gain is a fixed increment, not a fraction of the remaining gap**:
  the same procedure from an undertrained base (epoch 150, 0.492) gained the same
  ~7 points and validated at 0.561, so a weaker start ends weaker roughly one for
  one and headroom is not the constraint (2026-08-09, 24 cells, none
  significant). The procedure **replicates**: two independent six-seed matrices,
  each selecting and validating on its own draws, landed at 0.746 and 0.740, so
  ~0.74 is a property of the setup rather than a lucky seed. `reward_horizon`
  was found defaulting to 4 against `act_steps` 8 — half of every executed chunk
  was excluded from the gradient — and is now bound to `act_steps`; the fix is
  correct and made no measurable difference. Read the August 8–9 sections of
  `docs/DPPO_RL_FINETUNING.md` before opening a new configuration.
- **`dsrl/`** — the second RL strand: freeze the Diffusion Policy entirely and
  learn which input noise it denoises from
  ([arXiv:2506.15799](https://arxiv.org/abs/2506.15799)). `noise_policy`
  presents the checkpoint as the deterministic `a = pi_dp(s, w)` and exposes the
  frozen visual features its own U-Net conditions on; `sac` is soft actor-critic
  over that latent-noise space; `replay` caches those features rather than
  pixels, which is what makes an off-policy buffer affordable; `trainer` joins
  them to the unchanged `dppo_rl` environment; `steerability` measures the
  precondition the method rests on — that the noise moves the action at all.
  The base weights are loaded read-only, so
  unlike DPPO a bad run cannot degrade the policy. Read `docs/RL_DSRL.md` first.

### Script categories

`py/scripts/` holds more code than the package does. Broadly:

- **Run the task** — `pick_and_place/{sim,real,record_sim,record_teleop,finalize_sim_dataset}.py`
- **Run a policy** — `run_policy_{sim,real}.py`, `eval_policy_sim.py`,
  `run_flow_policy_sim.py`, `eval_scripted_parallel.py`,
  `compare_policy_evaluations.py`, `generate_scenario_manifest.py`
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

### Current state flow policy

The selected state-only flow checkpoint as of 2026-08-10 is the 30,000-update
temporal 1D U-Net with cube-symmetry augmentation:

```text
s3://allyouneed/pick-and-place/outputs/flow-policy-unet1d-rot6-cubeaug-30k-seed0/checkpoint.pt
```

Its matching export is:

```text
s3://allyouneed/pick-and-place/flow-policy-data/flow-policy-state-recovery-far-clean-993ep-rot6-cubeaug-val10/
```

The checkpoint SHA-256 is
`9ce2818a6c23676fe4c352ddff49ad991e22847548a48d30010dd323c5601247`.
Do not run it with a different export: `export.json` and `normalization.npz` are
part of the model contract.

The deployment operating point is predict 16, execute 8, and integrate the flow
with 10 Euler steps. A paired 20-scene check found the same 19/20 success at 10
and 100 steps, while 10 steps was 7.65 times faster. On 200 development-selection
scenes from seed stream 6,000,000, it scored **188/200 settled placements
(94.0%)**, with 11.35 mm median and 18.74 mm p90 final planar error. This is a
selection result, not the untouched seed-7,000,000 validation result; do not
promote the 94% number to a final benchmark until that one-time evaluation runs.

Use `py/scripts/run_flow_policy_sim.py` for an interactive or headless rollout.
Pass `--integration-steps 10 --act-steps 8`; the script's older integration-step
default is 100. Full provenance and episode records are at:

```text
s3://allyouneed/pick-and-place/outputs/flow-policy-unet1d-rot6-cubeaug-30k-seed0/evaluation-selection-seed6m-20260810/
```

Read `docs/FLOW_POLICY.md` for the matched augmentation experiment and 100,000-
update overfitting result, and `docs/POLICY_EVALUATION.md` before running the
untouched validation stream.

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
through `pick_and_place.core.paths` instead — see [`PAP_DATA_ROOT`](#pap_data_root).
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
- **Imports obey the layering above**, checked by
  `scripts/check_package_layering.py`. A module that needs two capability
  branches moves up to the convergence tier; a fact two branches share moves
  down to `spec`.
- **Spell policy prefixes out — never `dp_`.** `diffusion_policy_*` is the
  behavior-cloned policy every RL strand starts from. `dppo_*` is the PPO
  fine-tuning strand, and `DPPO` in prose means that or upstream
  `third_party/dppo`, nothing else. `dsrl_*` is the latent-noise steering
  strand, and `DSRL` in prose means that or the paper it implements. The two RL
  strands share the environment and the scoring harness and change different
  things, so a name that does not say which one is a name that will be misread.

## Known rough edges

Recorded here so they are not mistaken for intentional design:

- `scripts/run_policy_real.py` is oversized and combines too many
  responsibilities: one 938-line `main` holding ten nested closures that share
  state through `nonlocal`.
- Scripts hold more code than the package (~27k lines against ~23k).
- **`execute_episode`'s recording branch has no caller.** Its one production
  caller is `calibrate_joint_zeros.py`, which passes only the episode, follower,
  viewer and wrist camera. `recording`, `overhead_camera_cap`,
  `workspace_camera_cap`, `record_rest_to_rest`, `success_metadata`,
  `failed_trajectory_dir` and `joint_offsets_deg` are reachable from nothing —
  real-robot recording goes through
  `runtime/policy_real.run_physical_policy_episode` instead. The branch is
  covered by tests but exercised by no command, so treat it as unproven against
  hardware and decide whether it earns its keep before building on it.
- No dependency lockfiles are committed for either language.
- Python and TypeScript reimplement the same kinematics, grasp selection and
  trajectory logic. The first three are held together by the parity fixtures
  (see below); the two trajectory builders have genuinely diverged and are not.
- The browser entry point eagerly imports every visualization, producing an
  ~870 KB main chunk.
