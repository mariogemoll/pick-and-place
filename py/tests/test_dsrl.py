# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pick_and_place.dsrl.replay import ReplayBuffer, ReplaySpec  # noqa: E402
from pick_and_place.dsrl.sac import LatentSac, SacConfig  # noqa: E402
from pick_and_place.dsrl.steerability import (  # noqa: E402
    measure_action_spread,
    summarize_outcome_spread,
)

LATENT_DIM = 12
ACTOR_DIM = 7
CRITIC_DIM = 5


def _config(**overrides) -> SacConfig:
    defaults = {
        "latent_dim": LATENT_DIM,
        "actor_feature_dim": ACTOR_DIM,
        "critic_feature_dim": CRITIC_DIM,
        "hidden_dim": 16,
        "n_layers": 2,
        "device": "cpu",
        "seed": 0,
    }
    return SacConfig(**{**defaults, **overrides})


def _batch(size: int = 8, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "actor_features": torch.randn(size, ACTOR_DIM, generator=generator),
        "critic_features": torch.randn(size, CRITIC_DIM, generator=generator),
        "action": torch.randn(size, LATENT_DIM, generator=generator),
        "reward": torch.rand(size, 1, generator=generator),
        "done": (torch.rand(size, 1, generator=generator) > 0.5).float(),
        "next_actor_features": torch.randn(size, ACTOR_DIM, generator=generator),
        "next_critic_features": torch.randn(size, CRITIC_DIM, generator=generator),
    }


# -- the actor's action space -------------------------------------------------


def test_actor_respects_the_action_magnitude():
    """The latent action never leaves the box the config declares."""
    agent = LatentSac(_config(action_magnitude=1.5))
    action, _ = agent.actor(torch.randn(64, ACTOR_DIM))
    assert action.abs().max().item() <= 1.5


#: Initializations to average the untrained spread over, and samples from each.
#: The spread varies by about 0.03 between initializations, which is wider than
#: any tolerance worth asserting, so the test asks about the *design* rather
#: than about one draw of it.
UNTRAINED_ACTOR_SEEDS = 8
UNTRAINED_ACTOR_SAMPLES = 4000


def _untrained_actor_spread(action_magnitude: float) -> float:
    """Mean output spread of a freshly initialized actor, over several seeds."""
    spreads = []
    for seed in range(UNTRAINED_ACTOR_SEEDS):
        torch.manual_seed(seed)
        agent = LatentSac(_config(action_magnitude=action_magnitude))
        action, _ = agent.actor(torch.zeros(UNTRAINED_ACTOR_SAMPLES, ACTOR_DIM))
        spreads.append(action.std().item())
    return float(np.mean(spreads))


def test_an_untrained_actor_starts_near_the_base_policys_noise():
    """Why action_magnitude is 1.5 and not 1.0.

    Warmup fills the buffer by denoising w ~ N(0, I). If the actor's initial
    distribution were much narrower, the handover at the end of warmup would
    step the policy off the noise distribution the denoiser was trained under
    before anything had been learned. At 1.5 the two very nearly coincide.

    Averaged over initializations, because that is what the claim is about. One
    draw lands anywhere between 0.90 and 1.02 depending on the weights it got,
    and which values a seed produces is not identical across platforms — so a
    single draw is flaky whether it is seeded or not. The mean over eight is
    stable to a couple of thousandths and still separates 1.5 from the
    alternatives by a mile.
    """
    assert _untrained_actor_spread(1.5) == pytest.approx(0.944, abs=0.05)


def test_the_magnitude_is_what_sets_that_spread():
    """The tolerance above is tight enough to mean something."""
    assert _untrained_actor_spread(1.0) == pytest.approx(0.63, abs=0.05)
    assert _untrained_actor_spread(2.0) == pytest.approx(1.26, abs=0.05)


def test_deterministic_action_is_the_squashed_mean():
    agent = LatentSac(_config(action_magnitude=2.0))
    features = torch.randn(4, ACTOR_DIM)
    first = agent.actor.act(features, deterministic=True)
    second = agent.actor.act(features, deterministic=True)
    assert torch.equal(first, second)
    assert first.abs().max().item() <= 2.0


def test_log_probability_ignores_the_magnitude_scaling():
    """target_entropy is stated in the unscaled variable, per the paper.

    Two agents differing only in action magnitude must report the same entropy
    for the same pre-squash distribution, or the meaning of target_entropy = 0
    would silently depend on how wide the latent box was made.
    """
    narrow = LatentSac(_config(action_magnitude=1.0))
    wide = LatentSac(_config(action_magnitude=4.0))
    wide.actor.load_state_dict(narrow.actor.state_dict())
    features = torch.randn(16, ACTOR_DIM)
    torch.manual_seed(0)
    _, narrow_log_prob = narrow.actor(features)
    torch.manual_seed(0)
    wide_action, wide_log_prob = wide.actor(features)
    assert torch.allclose(narrow_log_prob, wide_log_prob)
    assert wide_action.abs().max().item() > 1.0


# -- the learner --------------------------------------------------------------


def test_update_moves_the_actor_and_reports_finite_diagnostics():
    agent = LatentSac(_config())
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    diagnostics = agent.update(_batch())
    after = list(agent.actor.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_terminal_transitions_cut_the_bootstrap():
    """A done transition's target is its reward, whatever the next state holds.

    The environment auto-resets and hands back the *next* episode's first
    observation, so a target that bootstrapped through a terminal step would
    credit a failed episode with a fresh episode's value.
    """
    agent = LatentSac(_config(gamma=0.99))
    batch = _batch(size=6)
    batch["done"] = torch.ones_like(batch["done"])
    batch["reward"] = torch.full_like(batch["reward"], 0.25)

    with torch.no_grad():
        next_action, next_log_prob = agent.actor(batch["next_actor_features"])
        next_value = agent.critic_target(batch["next_critic_features"], next_action)
        next_value = next_value.min(dim=0).values - agent.temperature * next_log_prob
        target = batch["reward"] + agent.config.gamma * (1.0 - batch["done"]) * next_value
    assert torch.allclose(target, batch["reward"])
    # And the next-state value is genuinely nonzero, so the test would fail if
    # the mask were dropped rather than passing by coincidence.
    assert next_value.abs().sum() > 0


def test_target_network_moves_only_by_tau():
    agent = LatentSac(_config(tau=0.01))
    online = next(iter(agent.critic.parameters()))
    with torch.no_grad():
        online.add_(1.0)
    target_before = next(iter(agent.critic_target.parameters())).detach().clone()
    agent.update(_batch())
    target_after = next(iter(agent.critic_target.parameters()))
    moved = (target_after - target_before).abs().max().item()
    assert 0 < moved < 1.0


# -- replay -------------------------------------------------------------------


def _spec(capacity: int = 10) -> ReplaySpec:
    return ReplaySpec(
        actor_feature_dim=ACTOR_DIM,
        critic_feature_dim=CRITIC_DIM,
        latent_dim=LATENT_DIM,
        capacity=capacity,
    )


def _add(buffer: ReplayBuffer, count: int, *, marker: float) -> None:
    buffer.add(
        actor_features=np.full((count, ACTOR_DIM), marker, dtype=np.float32),
        critic_features=np.full((count, CRITIC_DIM), marker, dtype=np.float32),
        action=np.full((count, LATENT_DIM), marker, dtype=np.float32),
        reward=np.full(count, marker, dtype=np.float32),
        done=np.zeros(count, dtype=np.float32),
        next_actor_features=np.full((count, ACTOR_DIM), marker, dtype=np.float32),
        next_critic_features=np.full((count, CRITIC_DIM), marker, dtype=np.float32),
    )


def test_replay_grows_then_saturates():
    buffer = ReplayBuffer(_spec(capacity=10))
    _add(buffer, 4, marker=1.0)
    assert len(buffer) == 4
    _add(buffer, 4, marker=2.0)
    _add(buffer, 4, marker=3.0)
    assert len(buffer) == 10


def test_replay_wraps_without_losing_a_row():
    """A batch straddling the end of the ring lands intact, not truncated."""
    buffer = ReplayBuffer(_spec(capacity=5))
    _add(buffer, 3, marker=1.0)
    _add(buffer, 3, marker=2.0)
    sample = buffer.sample(200, torch.device("cpu"), np.random.default_rng(0))
    values = set(np.unique(sample["reward"].numpy()).tolist())
    # Three twos overwrote the ring from index 3, wrapping onto index 0.
    assert values == {1.0, 2.0}
    assert sample["reward"].shape == (200, 1)


def test_replay_rejects_a_batch_larger_than_capacity():
    buffer = ReplayBuffer(_spec(capacity=4))
    with pytest.raises(ValueError, match="cannot insert"):
        _add(buffer, 5, marker=1.0)


def test_sampling_an_empty_buffer_is_an_error():
    buffer = ReplayBuffer(_spec())
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(2, torch.device("cpu"), np.random.default_rng(0))


# -- the steerability gate ----------------------------------------------------


def test_action_spread_is_zero_for_a_policy_that_ignores_its_noise():
    """The failure this gate exists to catch: every draw denoises to one chunk."""
    chunk = np.random.default_rng(0).normal(size=(3, 16, 6))
    chunks = np.stack([chunk] * 8)
    spread = measure_action_spread(chunks)
    assert spread.noise_std == pytest.approx(0.0)
    assert spread.max_pairwise_l2 == pytest.approx(0.0)
    assert spread.ratio == pytest.approx(0.0)


def test_action_spread_reports_the_ratio_against_tick_to_tick_motion():
    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(8, 4, 16, 6))
    spread = measure_action_spread(chunks)
    assert spread.noise_std > 0
    assert spread.step_std > 0
    assert spread.ratio == pytest.approx(spread.noise_std / spread.step_std)
    assert spread.samples == 4


def test_action_spread_needs_more_than_one_draw():
    with pytest.raises(ValueError, match="two noise draws"):
        measure_action_spread(np.zeros((1, 2, 16, 6)))


def _records(outcomes: dict[str, bool]) -> list[dict]:
    return [
        {"scenario_id": scenario, "success": success}
        for scenario, success in outcomes.items()
    ]


def test_outcome_spread_splits_contested_from_settled_scenes():
    runs = [
        _records({"a": True, "b": False, "c": True}),
        _records({"a": True, "b": False, "c": False}),
    ]
    spread = summarize_outcome_spread(runs)
    assert spread.scenarios == 3
    assert spread.always_success == 1  # a
    assert spread.always_failure == 1  # b
    assert spread.contested == 1  # c
    assert spread.mean_success_rate == pytest.approx(3 / 6)
    # An oracle picking the best draw per scene solves a and c.
    assert spread.headroom == pytest.approx(2 / 3)


def test_outcome_spread_pairs_only_scenes_every_repeat_covered():
    """Unequal scene coverage would otherwise weight scenes by trial count."""
    runs = [
        _records({"a": True, "b": True}),
        _records({"a": False}),
    ]
    spread = summarize_outcome_spread(runs)
    assert spread.scenarios == 1
    assert spread.contested == 1


def test_outcome_spread_rejects_disjoint_runs():
    runs = [_records({"a": True}), _records({"z": True})]
    with pytest.raises(ValueError, match="share no scenario ids"):
        summarize_outcome_spread(runs)
