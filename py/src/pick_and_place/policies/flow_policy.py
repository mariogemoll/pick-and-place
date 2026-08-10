# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Pick-and-place-specific conditional-flow training adapters."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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
