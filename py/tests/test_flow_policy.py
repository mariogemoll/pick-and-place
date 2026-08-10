# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import torch

from pick_and_place.policies.flow_policy import CUBE_SYMMETRIES, CubeSymmetryAugmentation


def test_cube_symmetries_are_the_24_proper_signed_permutations() -> None:
    assert CUBE_SYMMETRIES.shape == (24, 3, 3)
    torch.testing.assert_close(
        CUBE_SYMMETRIES.transpose(1, 2) @ CUBE_SYMMETRIES,
        torch.eye(3).expand(24, -1, -1),
    )
    torch.testing.assert_close(torch.linalg.det(CUBE_SYMMETRIES), torch.ones(24))


def test_cube_symmetry_augmentation_right_multiplies_both_rotations_only(monkeypatch) -> None:
    augmentation = CubeSymmetryAugmentation(observation_steps=2, observation_dim=8, rotation_start=1)
    observations = torch.tensor(
        [[10, 1, 0, 0, 0, 1, 0, 20, 30, 1, 0, 0, 0, 1, 0, 40]], dtype=torch.float32
    )
    quarter_turn = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    symmetry_index = torch.nonzero(torch.all(CUBE_SYMMETRIES == quarter_turn, dim=(1, 2))).item()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([symmetry_index], device=kwargs["device"]),
    )

    augmented = augmentation(observations)

    torch.testing.assert_close(
        augmented,
        torch.tensor(
            [[10, 0, 1, 0, -1, 0, 0, 20, 30, 0, 1, 0, -1, 0, 0, 40]], dtype=torch.float32
        ),
    )
