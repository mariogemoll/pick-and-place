# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A particular rig's solved overhead camera, expressed as a domain-randomization draw.

Calibration solves for an *absolute* camera pose; randomization works in
*displacements* from the pose the scene authors. They describe the same thing in
two coordinate systems, and this converts between them: a solved
``cam_pos``/``cam_quat``/``fovy`` becomes the :class:`CameraJitter` that would
have produced it.

That conversion is what lets one simulator serve both purposes. Reading a rig's
calibration directly into the scene makes every pixel depend on gitignored local
files, so a scored run stops reproducing from a clone -- which is why
:func:`~pick_and_place.runtime.policy_sim.build_policy_sim_model` refuses to do
it. A draw carries the same information as ordinary data: it can be written into
a scenario, committed, diffed, and handed to someone with a different rig.

The envelope is the other half of the story. Randomization already samples a box
around the authored pose, and if a rig sits inside that box then "my rig" is not
a separate world at all -- it is one point randomization was already covering.
:func:`envelope_usage` answers that for a given preset, so the claim can be
checked rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.core.camera_calibration import (
    LOCAL_CAMERA_EXTRINSICS_DIR,
    LOCAL_CAMERA_INTRINSICS_DIR,
    load_local_camera_extrinsics,
    load_local_camera_intrinsics,
)
from pick_and_place.sim.camera_pose_envelope import CameraBase, CameraJitter
from pick_and_place.sim.domain_randomization import DomainRandomizationPreset


def rig_camera_jitter(
    base: CameraBase,
    *,
    solved_pos: np.ndarray | tuple[float, float, float],
    solved_quat: np.ndarray | tuple[float, float, float, float],
    solved_fovy_deg: float | None = None,
) -> CameraJitter:
    """Return the jitter that moves ``base``'s authored camera onto a solved pose.

    The exact inverse of what
    :func:`~pick_and_place.sim.camera_pose_envelope.apply_camera_jitter` does:
    the position is a plain offset, the rotation is the ``xyz`` Euler
    decomposition of the delta pre-multiplied onto the authored orientation, and
    the focal scale inverts ``tan(fovy/2)``. ``solved_quat`` is MuJoCo's wxyz
    order, matching what the extrinsics sidecar stores.
    """
    base_pos = np.asarray(base.pos, dtype=float)
    solved_pos = np.asarray(solved_pos, dtype=float)
    position_m = solved_pos - base_pos

    base_rotation = Rotation.from_quat(np.asarray(base.quat, dtype=float)[[1, 2, 3, 0]])
    solved_rotation = Rotation.from_quat(np.asarray(solved_quat, dtype=float)[[1, 2, 3, 0]])
    rotation_deg = (solved_rotation * base_rotation.inv()).as_euler("xyz", degrees=True)

    focal_scale = 1.0
    if solved_fovy_deg is not None:
        focal_scale = math.tan(math.radians(base.fovy) / 2.0) / math.tan(
            math.radians(solved_fovy_deg) / 2.0
        )

    return CameraJitter(
        position_m=tuple(float(value) for value in position_m),
        rotation_deg=tuple(float(value) for value in rotation_deg),
        focal_scale=float(focal_scale),
    )


def load_rig_camera_jitter(
    base: CameraBase,
    *,
    camera_name: str = "overhead_camera",
    extrinsics_dir: Path = LOCAL_CAMERA_EXTRINSICS_DIR,
    intrinsics_dir: Path = LOCAL_CAMERA_INTRINSICS_DIR,
) -> CameraJitter | None:
    """Read this box's solved calibration for ``camera_name``, or ``None`` if absent.

    Absent is the ordinary case away from the rig, and not an error: a machine
    with no calibration renders the authored pose, which is what a scored run
    wants anyway.
    """
    extrinsics = load_local_camera_extrinsics(extrinsics_dir).get(camera_name)
    if extrinsics is None:
        return None
    intrinsics = load_local_camera_intrinsics(intrinsics_dir).get(camera_name, {})
    return rig_camera_jitter(
        base,
        solved_pos=extrinsics["pos"],
        solved_quat=extrinsics["quat"],
        solved_fovy_deg=intrinsics.get("fovy_deg"),
    )


@dataclass(frozen=True)
class EnvelopeUsage:
    """How much of a preset's overhead-camera box a particular jitter uses.

    Each fraction is per axis against that axis's half-width, so ``1.0`` sits
    exactly on the boundary and anything above it is a pose the preset could
    never have drawn.
    """

    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    focal: float

    @property
    def worst(self) -> float:
        return max(*self.position, *self.rotation, self.focal)

    @property
    def inside(self) -> bool:
        return self.worst <= 1.0


def envelope_usage(jitter: CameraJitter, preset: DomainRandomizationPreset) -> EnvelopeUsage:
    """Score ``jitter`` against ``preset``'s overhead-camera box.

    Box membership only. The preset also rejects poses that would lose a
    workspace-frame tag off the sensor, so a jitter inside the box is not
    necessarily one the preset would draw; put it through
    :func:`~pick_and_place.sim.camera_pose_envelope.overhead_pose_filter` for
    that.
    """
    position_mm = preset.scalars["overhead_camera_position_mm"]
    rotation_deg = preset.scalars["overhead_camera_rotation_deg"]
    focal_pct = preset.scalars["overhead_camera_focal_pct"]

    def fraction(value: float, half_width: float) -> float:
        return math.inf if half_width == 0.0 else abs(value) / half_width

    return EnvelopeUsage(
        position=tuple(
            fraction(value * 1000.0, position_mm) for value in jitter.position_m
        ),
        rotation=tuple(fraction(value, rotation_deg) for value in jitter.rotation_deg),
        focal=fraction((jitter.focal_scale - 1.0) * 100.0, focal_pct),
    )


def jitter_payload(jitter: CameraJitter) -> dict[str, Any]:
    """The jitter as the three ``DomainSample`` fields that carry it."""
    return {
        "overhead_camera_position_m": list(jitter.position_m),
        "overhead_camera_rotation_deg": list(jitter.rotation_deg),
        "overhead_camera_focal_scale": jitter.focal_scale,
    }
