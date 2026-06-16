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
