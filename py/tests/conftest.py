# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Global pytest fixtures and configurations."""

from pathlib import Path

import pupil_apriltags

from pick_and_place import camera_pose_envelope

#: Stand-in overhead calibration, holding the nominal intrinsics from
#: ``pick_and_place.camera_intrinsics``. The rig's own calibration lives in
#: ``config/camera_intrinsics`` and is machine-local, so frame-tag visibility --
#: and with it every preset that samples an overhead pose -- would otherwise be
#: untestable anywhere but the rig's own machine.
FIXTURE_CAMERA_INTRINSICS_DIR = Path(__file__).parent / "fixtures" / "camera_intrinsics"

camera_pose_envelope.LOCAL_CAMERA_INTRINSICS_DIR = FIXTURE_CAMERA_INTRINSICS_DIR

# pupil_apriltags 1.0.4.post1 has a known bug in Python 3.13 where the Detector.__del__
# method causes a segmentation fault during garbage collection. 
# We monkey-patch it here to avoid crashing the pytest runner.
# This leaks a small amount of memory per detector, which is harmless in tests.
pupil_apriltags.Detector.__del__ = lambda self: None
