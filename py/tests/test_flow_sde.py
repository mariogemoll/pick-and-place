# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The step arithmetic of the SDE that gives a flow policy a per-step density.

That the SDE transports the same distribution as the ODE -- the property that
makes the substitution legitimate at all -- is checked in ``test_flow_policy``
by integrating a Gaussian target. What is checked here is the step itself: the
identity the conversion rests on, the two limits, and the shape of the noise
schedule.
"""

import math

import pytest
import torch

from pick_and_place.policies.flow_matching import flow_sde_transition

# A target distribution whose exact conditional-flow velocity is available in
# closed form, so the sampler can be checked against the truth rather than
# against another approximation.
TARGET_MEAN = 1.5
TARGET_STD = 0.5


def exact_velocity(values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """The true CondOT velocity field for a Gaussian target.

    With ``x_t = t z + (1 - t) e``, ``z ~ N(mean, std^2)`` and ``e ~ N(0, 1)``,
    everything is jointly Gaussian and ``E[z - e | x_t]`` is exact.
    """
    scale, remaining = time, 1.0 - time
    variance = scale**2 * TARGET_STD**2 + remaining**2
    centered = values - scale * TARGET_MEAN
    return TARGET_MEAN + (scale * TARGET_STD**2 - remaining) / variance * centered


def exact_score(values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """The true score of the marginal at ``time``, which is Gaussian."""
    variance = time**2 * TARGET_STD**2 + (1.0 - time) ** 2
    return -(values - time * TARGET_MEAN) / variance


def test_the_score_is_recovered_from_the_velocity():
    """The identity the whole conversion rests on, checked against the truth."""
    values = torch.linspace(-3.0, 3.0, 41).reshape(-1, 1)
    for time_value in (0.05, 0.3, 0.5, 0.8, 0.95):
        time = torch.full_like(values, time_value)
        velocity = exact_velocity(values, time)
        recovered = (time * velocity - values) / (1.0 - time)
        assert recovered == pytest.approx(exact_score(values, time), abs=1e-4)


def test_zero_noise_is_plain_euler_integration():
    """At no noise the transition is exactly the deterministic ODE step."""
    values = torch.randn(7, 4, 3)
    velocity = torch.randn(7, 4, 3)
    time = torch.full((7, 1, 1), 0.25)
    mean, std = flow_sde_transition(
        values, velocity, time, step_size=0.1, noise_scale=0.0
    )
    assert mean == pytest.approx(values + velocity * 0.1, abs=1e-6)
    assert torch.count_nonzero(std) == 0


def test_the_noise_schedule_empties_the_last_step():
    """Exploration is largest at the start of the chain and vanishes at its end.

    This is the property that keeps the noise off the emitted action. A constant
    floor applied at every step, which is DPPO's default, lands undiminished on
    the action the environment executes.
    """
    values = torch.zeros(1, 2, 2)
    velocity = torch.zeros(1, 2, 2)
    widths = []
    for time_value in (0.0, 0.5, 0.9, 1.0):
        _, std = flow_sde_transition(
            values,
            velocity,
            torch.full((1, 1, 1), time_value),
            step_size=0.1,
            noise_scale=0.2,
        )
        widths.append(std.max().item())
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == pytest.approx(0.2 * math.sqrt(0.1))
    assert widths[-1] == pytest.approx(0.0)


def test_a_degenerate_step_or_negative_noise_is_rejected():
    values = torch.zeros(2, 2, 2)
    time = torch.zeros(2, 1, 1)
    with pytest.raises(ValueError):
        flow_sde_transition(values, values, time, step_size=0.0, noise_scale=0.1)
    with pytest.raises(ValueError):
        flow_sde_transition(values, values, time, step_size=0.1, noise_scale=-0.1)
    with pytest.raises(ValueError):
        flow_sde_transition(values, torch.zeros(2, 3, 2), time, step_size=0.1, noise_scale=0.1)
