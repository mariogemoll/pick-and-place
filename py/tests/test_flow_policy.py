# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pathlib import Path

import torch

from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D
from pick_and_place.policies.flow_matching import generate, load_model, model_checkpoint_config
from pick_and_place.policies.flow_policy import CUBE_SYMMETRIES, CubeSymmetryAugmentation


def test_cube_symmetries_are_the_24_proper_signed_permutations() -> None:
    assert CUBE_SYMMETRIES.shape == (24, 3, 3)
    torch.testing.assert_close(
        CUBE_SYMMETRIES.transpose(1, 2) @ CUBE_SYMMETRIES,
        torch.eye(3).expand(24, -1, -1),
    )
    torch.testing.assert_close(torch.linalg.det(CUBE_SYMMETRIES), torch.ones(24))


def test_cube_symmetry_augmentation_right_multiplies_both_rotations_only(monkeypatch) -> None:
    augmentation = CubeSymmetryAugmentation(
        observation_steps=2, observation_dim=8, rotation_start=1
    )
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
        torch.tensor([[10, 0, 1, 0, -1, 0, 0, 20, 30, 0, 1, 0, -1, 0, 0, 40]], dtype=torch.float32),
    )


def test_conditional_unet_preserves_temporal_action_shape_and_backpropagates() -> None:
    model = FlowConditionalUnet1D(
        action_dim=3,
        observation_dim=10,
        prediction_steps=8,
        time_embedding_dim=8,
        down_dims=(8, 16, 32),
        groups=4,
    )
    actions = torch.randn(2, 8, 3)

    velocity = model(actions, torch.rand(2, 1), torch.randn(2, 10))
    velocity.square().mean().backward()

    assert velocity.shape == actions.shape
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_conditional_unet_checkpoint_round_trip_and_seeded_generation(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = FlowConditionalUnet1D(
        action_dim=2,
        observation_dim=6,
        prediction_steps=4,
        time_embedding_dim=8,
        down_dims=(8, 16),
        groups=4,
    )
    model_type, model_config = model_checkpoint_config(model)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "model_type": model_type, "model_config": model_config},
        checkpoint,
    )
    loaded = load_model(checkpoint)
    observations = torch.randn(3, 6)

    torch.manual_seed(12)
    expected = generate(model, observations, num_steps=3)
    torch.manual_seed(12)
    actual = generate(loaded, observations, num_steps=3)

    torch.testing.assert_close(actual, expected)
