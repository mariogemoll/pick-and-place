# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The flow policy's PPO model. Skipped unless third_party/dppo is on the path."""

import pytest
import torch

pytest.importorskip("model.diffusion.diffusion_ppo", reason="third_party/dppo is not on the path")

from pick_and_place.dppo_rl.flow_actor import FlowActor  # noqa: E402
from pick_and_place.dppo_rl.flow_ppo import LOGPROB_CLAMP, FlowPPO  # noqa: E402
from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D  # noqa: E402

OBSERVATION_DIM = 17
ACTION_DIM = 6
PREDICTION_STEPS = 4
FLOW_STEPS = 6
FINETUNED_STEPS = 3
COND_STEPS = 2


class _StubCritic(torch.nn.Module):
    def forward(self, cond, no_augment=False):
        del no_augment
        return torch.zeros(len(cond["state"]), 1)


def _model(**overrides):
    torch.manual_seed(0)
    network = FlowConditionalUnet1D(
        action_dim=ACTION_DIM,
        observation_dim=COND_STEPS * OBSERVATION_DIM,
        prediction_steps=PREDICTION_STEPS,
        down_dims=(8, 16),
    )
    settings = {
        "flow_steps": FLOW_STEPS,
        "sampling_noise_scale": 0.1,
        "min_logprob_denoising_std": 0.06,
        "ft_denoising_steps": FINETUNED_STEPS,
        "horizon_steps": PREDICTION_STEPS,
        "obs_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "gamma_denoising": 0.99,
        "clip_ploss_coef": 0.01,
        "device": "cpu",
        "final_action_clip_value": None,
    }
    settings.update(overrides)
    return FlowPPO(actor=FlowActor(network), critic=_StubCritic(), **settings)


def _cond(batch=3):
    return {"state": torch.randn(batch, COND_STEPS, OBSERVATION_DIM)}


def test_the_chain_holds_every_finetuned_transition_and_nothing_else():
    model = _model()

    sample = model(cond=_cond(), deterministic=False, return_chain=True)

    assert sample.trajectories.shape == (3, PREDICTION_STEPS, ACTION_DIM)
    # One entry per fine-tuned step, plus the point entering the first of them.
    assert sample.chains.shape == (3, FINETUNED_STEPS + 1, PREDICTION_STEPS, ACTION_DIM)
    torch.testing.assert_close(sample.chains[:, -1], sample.trajectories)


def test_deterministic_sampling_is_the_euler_integration_the_evaluation_uses():
    model = _model()
    cond = _cond()

    torch.manual_seed(1)
    first = model(cond=cond, deterministic=True).trajectories
    torch.manual_seed(1)
    second = model(cond=cond, deterministic=True).trajectories
    torch.manual_seed(1)
    noisy = model(cond=cond, deterministic=False).trajectories

    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, noisy)


def test_log_probabilities_score_the_transitions_the_sampler_actually_took():
    model = _model()
    cond = _cond()

    sample = model(cond=cond, deterministic=False, return_chain=True)
    logprobs = model.get_logprobs(cond, sample.chains)

    assert logprobs.shape == (3 * FINETUNED_STEPS, PREDICTION_STEPS, ACTION_DIM)
    assert torch.all(torch.isfinite(logprobs))
    # The floor is what the density is evaluated with, so a transition sampled
    # from a much narrower SDE sits close to its mean and scores near the peak.
    peak = -torch.log(torch.tensor(0.06) * (2 * torch.pi) ** 0.5)
    assert torch.all(logprobs <= peak + 1e-4)
    # Nothing reaches the clamp upstream applies before the importance ratio; if
    # it did, old and new policies would score identically whatever changed.
    assert torch.all((logprobs > LOGPROB_CLAMP[0]) & (logprobs < LOGPROB_CLAMP[1]))


def test_a_subsampled_transition_matches_the_full_chains_log_probability():
    model = _model()
    cond = _cond()

    sample = model(cond=cond, deterministic=False, return_chain=True)
    full = model.get_logprobs(cond, sample.chains).reshape(
        3, FINETUNED_STEPS, PREDICTION_STEPS, ACTION_DIM
    )
    indices = torch.tensor([2, 0, 1])
    subsampled = model.get_logprobs_subsample(
        cond,
        sample.chains[torch.arange(3), indices],
        sample.chains[torch.arange(3), indices + 1],
        indices,
    )

    torch.testing.assert_close(subsampled, full[torch.arange(3), indices])


def test_only_the_finetuned_tail_of_the_chain_follows_the_updated_weights():
    model = _model()
    cond = _cond()
    with torch.no_grad():
        for parameter in model.actor_ft.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)

    torch.manual_seed(2)
    steered = model(cond=cond, deterministic=True, return_chain=True)
    torch.manual_seed(2)
    base = model(cond=cond, deterministic=True, return_chain=True, use_base_policy=True)

    # The chains enter the fine-tuned window at the same point, because the
    # frozen network produced everything before it, and diverge from there.
    torch.testing.assert_close(steered.chains[:, 0], base.chains[:, 0])
    assert not torch.allclose(steered.trajectories, base.trajectories)


def test_each_transition_is_scored_at_the_flow_time_it_was_sampled_at():
    """Pins the time grid, which every other test here is blind to.

    A Gaussian evaluated at its own draws has mean log-density
    ``-log(sigma sqrt(2 pi)) - 1/2`` per dimension, and the SDE fixes sigma at
    each flow time. Shifting the grid by one step -- sampling at ``i / K`` and
    scoring at ``(i + 1) / K`` -- leaves shapes, finiteness and the base/tail
    split exactly as they are, and lands the mean somewhere else.
    """
    noise_scale = 0.3
    # The true widths, not the numerical floor, so the arithmetic is exact.
    model = _model(sampling_noise_scale=noise_scale, min_logprob_denoising_std=0.0)
    cond = _cond(batch=512)

    sample = model(cond=cond, deterministic=False, return_chain=True)
    logprobs = model.get_logprobs(cond, sample.chains).reshape(
        512, FINETUNED_STEPS, PREDICTION_STEPS, ACTION_DIM
    )

    for index in range(FINETUNED_STEPS):
        time = (model.first_finetuned_step + index) / FLOW_STEPS
        std = noise_scale * ((1.0 - time) / FLOW_STEPS) ** 0.5
        expected = -torch.log(torch.tensor(std) * (2 * torch.pi) ** 0.5) - 0.5
        assert logprobs[:, index].mean().item() == pytest.approx(expected.item(), abs=0.05)


def test_fine_tuning_every_step_keeps_the_starting_noise_in_the_chain():
    model = _model(ft_denoising_steps=FLOW_STEPS)

    sample = model(cond=_cond(), deterministic=False, return_chain=True)

    assert sample.chains.shape == (3, FLOW_STEPS + 1, PREDICTION_STEPS, ACTION_DIM)
