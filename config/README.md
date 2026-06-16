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
