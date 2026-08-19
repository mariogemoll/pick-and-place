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
  their largest historical blob needs — `runtime/executor.py` has moved to
  `rollout/real.py` and is 23 KB there, but keeps a 61 KB entry under its old
  path because a 60.7 KB blob of it is still in the history.
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

### The TypeScript tests need generated fixtures

Five test files read `ts/public/so101.json`, which is **not** in the
repository. Generate it before running them:

```sh
cd py && MUJOCO_GL=egl python -m pick_and_place.sim.export -o ../ts/public/so101.xml
```

That command writes both `so101.xml` and `so101.json`; only the `.json` is
needed for tests. CI gets this by accident — its step reads as a smoke test of
the exporter and deletes the `.xml` afterwards, leaving the `.json` behind.

The live policy page's parity test
(`ts/src/visualizations/live-policy/parity.test.ts`) needs more than that: a
compiled scene, an exported policy, and a recorded rollout. It **skips itself**
when they are absent rather than failing, so a clone still runs green — which
also means a green run does not prove that check ran. See
[The live policy page](#the-live-policy-page).

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
| `py/` | The `pick_and_place` package (155 modules in 17 subpackages), 110 CLI scripts, 82 test files. Simulation, real-robot control, calibration, datasets, policies. |
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
| Capability branches | `scripted`, `perception`, `sim`, `hardware`, `data`, `policies` | Each owns one heavy dependency. **No branch may import another.** |
| Convergence | `runtime`, `plant`, `rollout`, `variants`, `calibration`, `analysis`, `cli` | May import anything, including each other. Nothing below them may import them. |

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
  `appearance` (its opposite: one draw of everything that is only pixels),
  `physics` (one draw of the arm itself — gain, time constant, mass, friction,
  damping, stiction, droop — behind a single amount dial),
  `robot_dynamics`, `camera_calibration` (the rig's measured calibration files),
  `paths`.
- **`scripted/`** — the analytic expert, which generates every demonstration and
  is the baseline every learned policy is scored against: `motion`
  (interpolation, easing, how long a move takes), `grasp` (where to take hold),
  `carry` (getting the cube across), `trajectory` (the eight phases assembled),
  `replan` (resuming from a checkpoint), `checkpoint` (which phase boundaries
  earn one), `visual_servo` and `descent` (steering onto what the wrist camera
  saw), `policy` (the whole controller, driven tick by tick from observations),
  and the declared reset distribution (`episode_sampling`, `scenario_sampling`).

  It is a branch, which is a real constraint and not a filing decision: it
  **consumes** sightings rather than producing them, and its episode preparation
  and preflight are injected, because each needs a capability — a tag detector,
  a compiled scene, live physics — that an expert has no other use for. What is
  left imports nothing but the physical facts and pure geometry, and is drivable
  from exactly the observations a learned policy is drivable from.
- **`perception/`** — AprilTag cube and drop-zone localization:
  `cube_detection`, `paper_detection`, `overhead_localization`,
  `detector_process`, `image_rectify`.
- **`sim/`** — composing and randomizing the MuJoCo scene: `builder`, `scene`,
  `model` (compile a runnable model and move things in it), `collisions`,
  `environment`, `materials`, `wrist_camera`, `camera_module`,
  `workspace_overlays`, `paper_target_marker`, `frame_tags`,
  `derive_kinematics`, `camera_pose_envelope`, `camera_extrinsics`, `export`,
  `physics` (apply an arm draw to a compiled scene, and take it back),
  `physics_export` (the policy scene with everything that is only pixels taken
  out, for the browser to step),
  and `domain_randomization` — the randomization envelope plus the half of a
  draw that shapes behavior (the wrist camera's mount error, the cube's resting
  orientation, the miscalibration). Loads the stock MJCF
  from `SO-ARM100/` with `MjSpec` and replaces full-mesh collision geoms with
  the hand-tuned box model; `python -m pick_and_place.sim.export` writes
  standalone MJCF plus a web manifest.
- **`hardware/`** — the physical arm: `follower`, `physical_rig`,
  `physical_collection`, `joint_zero_fit`.
- **`data/`** — recording and datasets: `recording`, `recorder`,
  `recording_config` (what one recording run is: the scene it draws, its frame
  sizes, where it lands), `dataset_metadata`, `dataset_subset`,
  `sim_dataset_staging`, `diffusion_policy_dataset`, and `trajectory_artifact`
  (one episode's behavior with no pixels in it — the true world and the believed
  one side by side, which is what a scene can be re-rendered from).
- **`policies/`** — controller implementations and the contract they are scored
  against: `policy_controllers`, `policy`, `policy_evaluation` (frozen scenario
  manifests in `config/evaluation/` and a success oracle),
  `diffusion_policy_pretrain`, `diffusion_policy_client`, and the state-only
  flow-matching policy (`flow_matching`, `flow_policy`,
  `diffusion_policy_unet`, `flow_onnx` — the sampler as one traceable graph, for
  the browser). ACT and SmolVLA are *evaluated* here but trained
  externally via the `lerobot` CLI.
- **`runtime/`** — what an episode needs around the loop that runs it, and the
  policy runners that do not use that loop. `episodes` samples an episode that
  runs clean and `preflight` vets a trajectory under live physics — the two the
  scripted expert has injected into it, because it may not reach for a scene or
  for physics itself. `wrist_servo` runs the rig's cube detector on its own
  thread, `frame_reader` gives each camera a thread holding only its newest
  frame, `wrist_preview` puts that on screen, and `ramp` eases the arm onto a
  pose. `sim_wrist_servo` is the simulated counterpart — render the wrist
  camera, detect the cube in it, inline rather than on a thread, which is what
  keeps a recorded episode a pure function of its seed — and `wrist_mixed_view`
  blends the true and believed wrist views for watching the servo converge.

  `believed_frame` is what the two worlds meet through: with a miscalibration
  draw, commands and recorded rows live in the believed frame while physics runs
  the true one.

  Around those: `policy_sim` and `policy_real` (running a *learned* policy,
  which needs no trajectory and so no phase loop), `overhead_detection`,
  `episode_loop`, `training_scenes`, `recorded_scenes`, `action_log`,
  `move_to_random_pose`.
- **`plant/`** — the two things you command and observe: hardware, and sim. Both
  are the same shape — **a true world plus a believed shadow**. On the rig the
  true world is the physical arm and the shadow is a MuJoCo model stepped at the
  commanded joints; in sim the true world is a MuJoCo model and the shadow is a
  second one over it. Both step MuJoCo, both take the wrist camera pose from
  forward kinematics of the believed shadow, and both solve tag detection
  against it.

  What differs is narrow — where the image comes from, what receives the
  commands, whether the detector runs on a thread or inline, and what drives the
  clock — and all four fit behind `interface`'s three operations: command
  joints, read back joints, give me the latest cube sighting. `wrist_localizer`
  is the other half of the same rule: turning an image into a cube pose is
  detection, so it lives here rather than inside the controller that consumes
  the answer.

  `overhead` locates the cube and the drop plate the same way — render the
  overhead camera at detection resolution, run the detector, solve through where
  the camera is *believed* to be. Doing that honestly makes sim **better** than
  the rig (0.4 mm against 6-9 mm), so a residual calibration error is drawn and
  the belief error becomes an outcome rather than a value applied to the truth.
  `overhead_check` is the measurement that says whether it lands on the rig's
  distribution; `scripts/check_overhead_localization.py` runs it. Two known
  properties: the arm can stand in the way, which is why the rig hunts and why
  sim now does too, and simulated yaw is still cleaner than the rig's (~0.4
  against ~2 degrees) because rendered tags carry no sensor noise. `Sighting.fresh`
  is where the thread/inline difference surfaces: the rig returns the same solve
  on consecutive ticks and folding it in twice would pull the grasp too far.
- **`rollout/`** — one episode runner, over any controller and any plant. `phase`
  is the tick loop: evaluate the phase, observe, record, command, repeat, with
  the descent's visual servo folded in. There is one of it rather than two,
  because once the world sits behind `plant/` a rig run and a sim run *are* the
  same loop. `sim` and `real` are the two setups that build a plant and drive it
  — a scene and a camera rig on one side, cameras and a ramp onto the start pose
  on the other — plus `sim_dataset`/`real_dataset` (a dataset row per tick) and
  `records` (what a finished episode leaves behind).

  **A tick is observed before it is commanded**, which is the dataset's central
  invariant: a row pairs the observation at time t with the action issued from
  it. It is also why a phase's last tick is recorded but never commanded.

  `localized_episode` prepares an episode the way the rig does — put the plate
  down, look, hunt if something is hidden, then plan on what was seen — which is
  what `record_sim.py --overhead-perception` runs. `--physics-randomization`
  draws the arm itself; the draw is applied **before** the episode is planned,
  because planning ends in a preflight and preflight runs live physics, so
  vetting against the nominal arm when a drawn one will fly it checks a
  different world than the one that follows. `checkpoint` carries out a
  replan, and `scripted` hands the expert the scene and physics it is not
  allowed to reach for itself.
- **`variants/`** — one recorded trajectory, rendered many ways. Everything here
  answers "no" to the question that organizes the sim/real split: *if I change
  this, does the correct action change?* Lighting, materials, colours,
  backgrounds, viewpoint, exposure and noise move pixels and nothing else, so
  they are drawn against an episode that already succeeded and its action labels
  stay correct. `appearance` (the named palettes), `draw` (the envelopes a
  variant samples from), `scene` (applying a draw to a compiled model),
  `renderer` (replaying an artifact through the recording camera pipeline),
  `render` (one artifact into N variants — variant outer, frame inner, so the
  scene is restyled once instead of per frame), `video` (encoding a variant the
  way the recording was encoded).

  The input is a trajectory artifact, which is why none of this needs the
  planner, the detectors or physics — and why a domain-randomization experiment
  costs a render pass rather than a fresh collection run.
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
- **`dppo_rl/`** — fine-tuning a pretrained policy with PPO. Two families run
  through one episode loop, reward and scene stream; `observations.py` holds
  what each is shown and what its normalized action means. The **state flow
  policy** substitutes only the transition kernel (`flow_ppo.py`, over the actor
  adapter in `flow_actor.py`): integrating a flow ODE has no per-step likelihood
  for PPO to differentiate, while the SDE with the same marginals does, and its
  noise schedule vanishes as the chain ends so exploration never lands
  undiminished on the emitted action. It reads privileged task state, so nothing
  renders and a rollout costs about a second. Its objective is **speed, not
  success**: the base places at a median 81 ticks with 0.94 success, so the
  dense return has range to move in where the success rate has almost none.
  Gate and score it with `check_flow_rl_env.py`. **It does not work yet**: PPO
  degrades the policy at every step size tried (0.92 to 0.66 over 121 iterations
  at lr 3e-6; 0.92 to 0.14 over 301 at 3e-7, so a smaller step only postpones
  it), and it does so with the trust region engaging, the critic explaining 70%
  of return variance, and no log-probability clamping — none of which the visual
  strand ever achieved. The leading untested suspect is that the likelihood
  floor is three to six times wider than the SDE's own per-step standard
  deviation. No fine-tuned checkpoint is worth scoring. The **visual Diffusion
  Policy** uses DPPO's own diffusion model, and
  **works as a train-and-select procedure on the recovery base, not as a reliable
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
  correct and made no measurable difference.
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
  unlike DPPO a bad run cannot degrade the policy.

### Script categories

`py/scripts/` holds about as much code as the package does. Broadly:

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
  `probe_camera_pose_envelope.py`, `check_overhead_localization.py` (does
  simulated overhead perception miss by as much as the rig does?)
- **Web assets** — `export_generic_robot.py`, `export_episode_rolls.py`,
  `distill_grasp_policy.py`, `render_apriltag_textures.py`,
  `export_web_policy_scene.py`, `export_flow_policy_onnx.py`,
  `export_policy_parity_fixture.py`
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

## The live policy page

`ts/policy.html` runs the task in the reader's browser: MuJoCo's WebAssembly
build steps the scene, the state flow policy runs through `onnxruntime-web`, and
the cube and target plate are dragged wherever the reader wants them. It is a
separate page from the index on purpose — the simulator and the inference
runtime together are an order of magnitude heavier than everything the index
loads.

**It is not a benchmark, and the page says so.** A different engine build rounds
differently and one episode is one episode; the scored numbers come from the
Python harness over the frozen manifests in `config/evaluation/`.

### Building its assets

The page needs five files, none of them committed. Two are the ordinary web
manifests the other visualizations already use; three are new.

Prerequisites, if this is a fresh clone: the submodules
(`git submodule update --init`), the AprilTag textures
(`MUJOCO_GL=egl python py/scripts/render_apriltag_textures.py --all-defaults`),
and a Python environment with `mujoco`, `torch`, `onnx` and `onnxruntime`.

The checkpoint and its export are the ones named in
[Current state flow policy](#current-state-flow-policy), and they live in S3
rather than on disk:

```sh
export PAP_FLOW=$PAP_DATA_ROOT/flow-policy
aws s3 cp s3://allyouneed/pick-and-place/outputs/\
flow-policy-unet1d-rot6-cubeaug-30k-seed0/checkpoint.pt "$PAP_FLOW"/
aws s3 sync s3://allyouneed/pick-and-place/flow-policy-data/\
flow-policy-state-recovery-far-clean-993ep-rot6-cubeaug-val10/ "$PAP_FLOW"/export/
```

Then, from `py/`:

```sh
# The robot and the environment, as the other visualizations use them.
MUJOCO_GL=egl python -m pick_and_place.sim.export -o ../ts/public/so101.xml
MUJOCO_GL=egl python -m pick_and_place.sim.export --environment-only \
    -o ../ts/public/environment.xml

# The scene the browser steps, and the policy it runs.
MUJOCO_GL=egl python scripts/export_web_policy_scene.py -o ../ts/public/policy-scene
MUJOCO_GL=egl python scripts/export_flow_policy_onnx.py \
    --checkpoint "$PAP_FLOW"/checkpoint.pt --export "$PAP_FLOW"/export \
    -o ../ts/public/flow-policy

# Only needed to run the parity test.
MUJOCO_GL=egl python scripts/export_policy_parity_fixture.py \
    --checkpoint "$PAP_FLOW"/checkpoint.pt --export "$PAP_FLOW"/export \
    -o ../ts/test-fixtures/policy-parity.json
```

Both `sim.export` invocations write an `.xml` beside the `.json`. Only the
`.json` is read, by the page and by the tests; the `.xml` is a by-product.

The parity fixture goes to `ts/test-fixtures/` rather than `ts/public/` because
everything under `public/` is copied into the build and served to every visitor,
and that file is an input to one test.

The ONNX exporter prints the checkpoint's SHA-256, which should read
`9ce2818a…5601247`. `export.json` and `normalization.npz` are part of the model
contract, so a checkpoint paired with a different export is a silent error the
digest is there to catch.

### Three things that are easy to get wrong

- **The scene is a compiled binary (`.mjb`), not MJCF.** MuJoCo's XML writer
  rounds to six significant figures. The binary carries the compiled model
  verbatim, so the browser steps the same numbers Python does. The cost is that
  a binary is tied to the engine that wrote it: the manifest records the MuJoCo
  version and the page refuses to load against a different one. **Keep the
  `@mujoco/mujoco` npm version and the Python `mujoco` version equal.**
- **Deleting visual geometry moves the arm unless the inertials are frozen
  first.** Three bodies — `wrist_camera_mount`, `overhead_camera_mount`,
  `workspace_frame_frame` — infer their mass from geometry rather than declaring
  it, so stripping the meshes took two thirds of the wrist mount's mass off the
  arm. `physics_export.freeze_inertials` compiles the intact scene and writes
  what MuJoCo computed back into the spec before anything is removed;
  `py/tests/test_physics_export.py` pins it.
- **The tracking bias is scaled to zero at the nominal operating point.**
  `tracking_bias_rad` returns the fitted droop, but `PolicySimEnv` applies it
  through `tracking_bias_offsets`, and a scenario with no physics randomization
  has `tracking_bias_scale = 0.0`. Exporting the fitted value directly pushed
  every browser command a couple of degrees past its target. The export goes
  through the same scaling.

### What checks it

`ts/src/visualizations/live-policy/parity.test.ts` replays a rollout Python
actually ran — same scene, same noise draws — through the whole browser stack in
Node, and compares. It asserts two different things:

- **The first tick, tightly.** Both sides start from the same state and the same
  noise, so the only difference left is arithmetic: the observation matches
  exactly, the action to a few times 1e-5 degrees, the resulting state to 2e-7.
  This is what catches defects — the tracking-bias bug above sat at 2.8e-3 here.
- **The whole rollout, loosely.** Forty ticks of contact-rich closed-loop
  physics amplify a last-place difference into about 0.04 degrees of drift. That
  bound says the browser stayed in the same episode; nothing tighter is
  available and no tolerance would make it so.

### Precision

The exported graph is full precision by default: 17.4 MB, and it agrees with
PyTorch to 5e-7 in the normalized action space. `--precision fp16` halves it to
9.7 MB and moves the sampled endpoint by about 1.4e-3, which is a change in what
the policy *does*, not in how it is stored. **Score it on the development scenes
before promoting it** — the same way the 10-versus-100 integration-step question
was settled.

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
| `ts/public/` | All generated web assets, including the live policy page's compiled scene and policy weights. |
| `ts/test-fixtures/` | Generated test inputs that must not be served, currently the live policy parity rollout. |
| `datasets/` | Recorded and simulated LeRobot datasets. Tens of GB. |
| `outputs/`, `output/` | Training runs, checkpoints, diagnostics. |
| `intermediary-glb/`, `dist_assets/` | Mesh pipeline intermediates. |
| `config/camera_{intrinsics,extrinsics}/*.json` | Measured per-rig calibration. |

**Working notes, research write-ups, and dev-process material do not belong in
this repository.** Do not add code comments that reference an external
document for context; write the rationale into the comment itself so it
stands alone.

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
  responsibilities: one 1,068-line `main` holding thirteen nested closures that
  share state through `nonlocal`.
- Scripts hold about as much code as the package (~32k lines each). They used to
  hold considerably more; the balance moved as stable logic was pulled out of
  the recording scripts into `rollout/`.
- **`execute_episode`'s recording branch has no caller.** Its one production
  caller is `calibrate_joint_zeros.py`, which passes only the episode, follower,
  viewer and wrist camera. `recording`, `overhead_camera_cap`,
  `workspace_camera_cap`, `record_rest_to_rest`, `success_metadata`,
  `failed_trajectory_dir` and `joint_offsets_deg` are reachable from nothing —
  real-robot recording goes through
  `runtime/policy_real.run_physical_policy_episode` instead (`execute_episode`
  now lives in `rollout/real.py`). The branch is
  covered by tests but exercised by no command, so treat it as unproven against
  hardware and decide whether it earns its keep before building on it.
- No dependency lockfiles are committed for either language.
- Python and TypeScript reimplement the same kinematics, grasp selection and
  trajectory logic. The first three are held together by the parity fixtures
  (see below); the two trajectory builders have genuinely diverged and are not.
- The browser entry point eagerly imports every visualization, producing an
  ~870 KB main chunk.
