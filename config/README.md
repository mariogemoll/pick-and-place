<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->
<!-- SPDX-License-Identifier: 0BSD -->

# Local Configuration

This directory holds machine-local configuration files.

Camera intrinsics can be placed in `camera_intrinsics/` as JSON files named
after the MuJoCo camera:

- `camera_intrinsics/wrist_camera.json`
- `camera_intrinsics/overhead_camera.json`

These JSON files are ignored by git. When present, `python -m pick_and_place.export`
loads them automatically and uses them instead of the nominal camera defaults.

## Camera intrinsics calibration

Generate the project's standard OpenCV ChArUco board as a vector A4 PDF. Print
it at 100% scale (with no printer scaling), and verify a square with a ruler:

```sh
cd py
python scripts/generate_charuco_board.py
```

Then run the interactive Python calibrator. It automatically captures stable,
distinct views and writes the requested JSON file:

```sh
cd py
python scripts/calibrate_camera_intrinsics.py \
  --camera 0 \
  --output ../config/camera_intrinsics/overhead_camera.json
```

Move the board through the whole image at several distances and moderate tilts.
Press `u` to undo a view, `x` to remove the worst-reprojection-error view, `d`
to reset, and `s` to save. Capture 20--30 views where possible.

The calibration command is independent of the scene's camera names. For
example, an iPhone overview camera can be stored separately:

```sh
cd py
python scripts/calibrate_camera_intrinsics.py \
  --camera 0 \
  --output ../config/camera_intrinsics/iphone_overview.json
```

The file becomes part of a scene or replay workflow only when that workflow is
configured to use it.

The iPhone may rotate its webcam stream while being moved. By default the
calibrator accepts only landscape frames and visibly ignores portrait frames
or any different resolution after capture starts. Use `--orientation portrait`
for a portrait-only calibration, or `--orientation any` only when the camera
stream's dimensions are known to remain fixed.

Camera extrinsics solved from the workspace-frame AprilTags can be placed in
`camera_extrinsics/` as JSON files named after the MuJoCo camera:

- `camera_extrinsics/overhead_camera.json`

These JSON files are also ignored by git. When present, exports apply them over
the nominal authored camera pose.

Solve the overhead camera extrinsics from a frame where workspace-frame tags
12-15 are visible:

```sh
cd py
python -m pick_and_place.cam_align_solve \
  --camera 0 \
  --intrinsics ../config/camera_intrinsics/overhead_camera.json
```

The command reports reprojection error and the delta from the nominal authored
camera pose before saving the measured pose.

## Robot Dynamics

Recorded LeRobot datasets can be used to fit the follower arm's actuator
response. The fitted file lives at:

- `robot_dynamics/so101_follower.json`

Generate it from a dataset root:

```sh
PYTHONPATH=py/src python3 py/scripts/calibrate_robot_dynamics.py \
  datasets-512/combined \
  --output config/robot_dynamics/so101_follower.json
```

The calibrator fits per-joint delayed first-order response from the recorded
`action` and `observation.state` streams, then writes actuator time constants.
`build_robot()` / `build_scene()` apply those time constants to the composed
MuJoCo actuators by default, so generated MJCF exports and sim/replay tools use
the calibrated response automatically.

Use raw upstream actuator dynamics for comparison:

```sh
PYTHONPATH=py/src mjpython py/scripts/pick_and_place/sim.py --no-robot-dynamics
```

```sh
PYTHONPATH=py/src mjpython py/scripts/replay_dataset_episode.py \
  datasets-512/combined 0 \
  --no-robot-dynamics
```
