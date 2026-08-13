# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pathlib import Path

import pytest
import torch

from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D
from pick_and_place.policies.flow_matching import (
    flow_sde_transition,
    generate,
    load_model,
    model_checkpoint_config,
)
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


def _gaussian_condot_velocity(values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Exact CondOT velocity carrying N(0, 1) noise to the target N(2, 0.5^2).

    With ``x_t = t z + (1 - t) e`` and Gaussian ``z``, both the marginal and the
    conditional expectation are closed forms, so this is the velocity a
    perfectly trained model would predict -- which makes the sampler's own error
    the only thing a test over it can measure.
    """
    mean, deviation = 2.0, 0.5
    variance = (time * deviation) ** 2 + (1 - time) ** 2
    expected_endpoint = mean + time * deviation**2 * (values - time * mean) / variance
    return (expected_endpoint - values) / (1 - time)


def _integrate(num_steps: int, noise_scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(20_000, 1, generator=generator)
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        time = torch.full((len(values), 1), step * step_size)
        mean, deviation = flow_sde_transition(
            values,
            _gaussian_condot_velocity(values, time),
            time,
            step_size=step_size,
            noise_scale=noise_scale,
        )
        values = mean + deviation * torch.randn(values.shape, generator=generator)
    return values


def test_the_flow_sde_transports_the_same_distribution_as_the_ode() -> None:
    # The score correction is what makes this true: dropping it leaves the noise
    # uncompensated and the sample distribution too wide.
    deterministic = _integrate(num_steps=200, noise_scale=0.0, seed=0)
    stochastic = _integrate(num_steps=200, noise_scale=0.5, seed=0)

    for samples in (deterministic, stochastic):
        assert samples.mean().item() == pytest.approx(2.0, abs=0.02)
        assert samples.std().item() == pytest.approx(0.5, abs=0.02)


def test_flow_sde_noise_vanishes_as_the_chain_ends() -> None:
    values = torch.zeros(3, 2, 2)
    velocity = torch.ones(3, 2, 2)
    times = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1)

    mean, deviation = flow_sde_transition(
        values, velocity, times, step_size=0.1, noise_scale=0.4
    )

    torch.testing.assert_close(
        deviation[:, 0, 0], 0.4 * torch.sqrt(torch.tensor([0.1, 0.05, 0.0]))
    )
    # With no noise the step is plain Euler, and the correction scales with the
    # square of the noise level rather than being clipped away.
    plain, silent = flow_sde_transition(
        values, velocity, times, step_size=0.1, noise_scale=0.0
    )
    torch.testing.assert_close(plain, values + velocity * 0.1)
    assert torch.all(silent == 0.0)
    torch.testing.assert_close(mean - plain, 0.5 * 0.4**2 * times * velocity * 0.1)
