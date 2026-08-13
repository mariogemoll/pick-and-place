# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place-specific conditional-flow training adapters."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.core.rotations import quat_wxyz_to_rotation_6d
from pick_and_place.spec.controller import STATE_FEATURE

CUBE_ROTATION_NAMES = (
    "cube_rotation_column_0_x",
    "cube_rotation_column_0_y",
    "cube_rotation_column_0_z",
    "cube_rotation_column_1_x",
    "cube_rotation_column_1_y",
    "cube_rotation_column_1_z",
)


def _make_cube_symmetries() -> torch.Tensor:
    """Return the 24 orientation-preserving signed permutation matrices."""
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros(3, 3)
            for column, row in enumerate(permutation):
                matrix[row, column] = signs[column]
            if torch.linalg.det(matrix) > 0.0:
                matrices.append(matrix)
    return torch.stack(matrices)


CUBE_SYMMETRIES = _make_cube_symmetries()

NORMALIZATION_NAMES = frozenset(
    {"observation_min", "observation_max", "endpoint_min", "endpoint_max"}
)


def load_export(export_dir: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read a state flow-policy export's manifest and its normalization bounds."""
    export_dir = Path(export_dir)
    with (export_dir / "export.json").open() as file:
        manifest = json.load(file)
    with np.load(export_dir / "normalization.npz", allow_pickle=False) as archive:
        bounds = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    if set(bounds) != NORMALIZATION_NAMES:
        raise ValueError(f"normalization must contain exactly {sorted(NORMALIZATION_NAMES)}")
    return manifest, bounds


def normalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Map each dimension onto ``[-1, 1]``, leaving a degenerate one at zero."""
    span = maximum - minimum
    return np.where(
        span > 1e-6, 2 * (values - minimum) / np.where(span > 1e-6, span, 1) - 1, 0
    ).astype(np.float32)


def unnormalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize`."""
    span = maximum - minimum
    return np.where(span > 1e-6, (values + 1) / 2 * span + minimum, minimum).astype(np.float32)


def pack_observation(observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
    """Pack state in the order declared by this project's flow-policy export.

    Six robot coordinates, the cube's position, the first two columns of its
    rotation matrix, and the target's planar position. The cube pose comes from
    the simulator rather than from a camera, so this observation is privileged
    and the policy that reads it is a simulation policy until something
    estimates the same quantities on the rig.
    """
    task = info["task_state"]
    return np.concatenate(
        (
            np.asarray(observation[STATE_FEATURE], dtype=np.float32),
            np.asarray(task["cube_position_m"], dtype=np.float32),
            np.asarray(quat_wxyz_to_rotation_6d(task["cube_orientation_wxyz"]), dtype=np.float32),
            np.asarray(task["target_xy_m"], dtype=np.float32),
        )
    )


@dataclass(frozen=True)
class CubeSymmetryAugmentation:
    """Location of the cube's rotation-6D values in a flattened observation history."""

    observation_steps: int
    observation_dim: int
    rotation_start: int

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        """Right-multiply each observed cube rotation by a sampled cube symmetry."""
        expected_dim = self.observation_steps * self.observation_dim
        if observations.ndim != 2 or observations.shape[1] != expected_dim:
            raise ValueError(f"observations must have shape (examples, {expected_dim})")

        values = observations.reshape(-1, self.observation_steps, self.observation_dim).clone()
        rotation_6d = values[..., self.rotation_start : self.rotation_start + 6]
        column_0 = rotation_6d[..., :3]
        column_1 = rotation_6d[..., 3:]
        rotations = torch.stack((column_0, column_1, torch.linalg.cross(column_0, column_1)), dim=-1)
        symmetry_indices = torch.randint(
            len(CUBE_SYMMETRIES), (len(observations),), device=observations.device
        )
        symmetries = CUBE_SYMMETRIES.to(device=observations.device, dtype=observations.dtype)[
            symmetry_indices
        ]
        augmented = rotations @ symmetries[:, None]
        values[..., self.rotation_start : self.rotation_start + 6] = torch.cat(
            (augmented[..., :, 0], augmented[..., :, 1]), dim=-1
        )
        return values.reshape_as(observations)


def load_cube_symmetry_augmentation(dataset_path: str | Path) -> CubeSymmetryAugmentation:
    """Create cube augmentation after validating a pick-and-place state export."""
    export_dir = Path(dataset_path).resolve().parent
    with (export_dir / "export.json").open() as file:
        manifest = json.load(file)
    observation_names = tuple(manifest["observation_names"])
    rotation_start = observation_names.index(CUBE_ROTATION_NAMES[0])
    if observation_names[rotation_start : rotation_start + 6] != CUBE_ROTATION_NAMES:
        raise ValueError("cube rotation-6D components must be consecutive and column-major")
    if manifest.get("cube_rotation_representation") != (
        "first two rotation-matrix columns, concatenated column-by-column"
    ):
        raise ValueError("export does not declare the expected cube rotation representation")
    with np.load(export_dir / "normalization.npz", allow_pickle=False) as archive:
        rotation_min = archive["observation_min"][rotation_start : rotation_start + 6]
        rotation_max = archive["observation_max"][rotation_start : rotation_start + 6]
    if not np.allclose(rotation_min, -1.0) or not np.allclose(rotation_max, 1.0):
        raise ValueError("cube rotation columns must use their physical [-1, 1] bounds")
    return CubeSymmetryAugmentation(
        observation_steps=int(manifest["observation_steps"]),
        observation_dim=int(manifest["observation_dim"]),
        rotation_start=rotation_start,
    )
