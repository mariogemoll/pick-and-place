# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Soft actor-critic over the diffusion policy's latent-noise action space.

This is the paper's DSRL-SAC instantiation: ordinary SAC, applied to the
transformed MDP whose action is the noise ``w`` handed to a frozen diffusion
policy. Nothing here knows that ``w`` is noise -- the denoising happens in
:mod:`~pick_and_place.dsrl.trainer` between choosing ``w`` and stepping the
environment -- which is exactly the black-boxing that makes the method cheap.

Written rather than vendored. The reference implementation forks
stable-baselines3, and this project removed that dependency once already when
the reverse-curriculum strand was deleted; SAC over small MLPs is a few hundred
lines and the alternative is carrying a fork to reach one algorithm.

Two conventions worth stating because they are easy to get subtly wrong:

- **The latent action is stored in raw noise units.** The actor emits
  ``magnitude * tanh(u)``, and the warmup rollouts that fill the buffer draw
  ``w ~ N(0, I)`` unclipped, exactly as the paper's initial rollouts do. Both
  land in the buffer on the same scale, so the critic never sees two different
  parameterizations of the same quantity. The log-probability, however, is the
  density of the *pre-scaling* ``tanh(u)`` in ``[-1, 1]``: keeping the constant
  ``-d log(magnitude)`` out of it is what makes ``target_entropy = 0`` mean what
  it means in the paper regardless of how ``magnitude`` is set.
- **The actor and the critic may read different features.** In simulation the
  critic can have privileged state while the actor sees only what deploys. That
  costs nothing at deployment -- the value function exists only during training
  -- and it is the same asymmetry
  :class:`~pick_and_place.dppo_rl.privileged_critic.PrivilegedCritic` already
  buys on the DPPO strand, where it tripled the usable advantage signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

# Bounds on the predicted log standard deviation. The wide end is the usual SAC
# value; the narrow end stops the entropy term from diverging once the actor
# becomes confident, which on a 96-dimensional action it otherwise does.
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class SacConfig:
    """Everything that shapes the latent-space learner.

    Defaults follow the paper's common online hyperparameters (Table 3) and its
    Robomimic image tasks (Table 4), which are the closest published setting to
    this one: a visual diffusion policy with chunked actions and a sparse
    reward.
    """

    latent_dim: int
    actor_feature_dim: int
    critic_feature_dim: int
    # Largest absolute value the latent-noise action may take. The paper's
    # ``b_W``; 1.5 across its Robomimic tasks.
    #
    # 1.5 is not arbitrary, and the reason decides what happens at the moment
    # warmup ends. An untrained actor emits ``magnitude * tanh(u)`` for
    # ``u ~ N(0, 1)``, whose per-dimension standard deviation is 0.628 *
    # magnitude -- so at 1.5 it is 0.942, within 6% of the standard normal the
    # base policy denoises from. Training therefore starts approximately *at*
    # the base policy, and the handover from warmup is continuous. At 1.0 the
    # actor would begin a third narrower than the noise distribution the
    # diffusion policy was trained under, which is off-distribution for the
    # denoiser before anything has been learned.
    action_magnitude: float = 1.5
    hidden_dim: int = 2048
    n_layers: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    # The paper sets this to 0 rather than the customary ``-latent_dim``. On a
    # high-dimensional latent that difference is large: -96 would drive the
    # actor toward the widest distribution the tanh allows.
    target_entropy: float = 0.0
    init_temperature: float = 1.0
    # Whether the temperature is tuned toward ``target_entropy`` or held where it
    # starts. Auto-tuning is standard SAC and is what the paper implies, but on
    # this 96-dimensional latent it has been measured chasing its tail: entropy
    # swung 33 -> -0.6 -> 3.4 -> -22 -> 20.6 nats across one run while the
    # temperature tracked it, which changes the sampling distribution faster
    # than the critic can follow. Holding it removes one feedback loop.
    auto_temperature: bool = True
    n_critics: int = 2
    # Tanh, per the paper's Table 3, not the ReLU that SAC implementations
    # usually default to.
    activation: str = "tanh"
    device: str = "cuda:0"
    seed: int = 0
    # Diagnostics only; never read by the actor.
    tags: tuple[str, ...] = field(default_factory=tuple)


def _activation(name: str) -> nn.Module:
    try:
        return {"tanh": nn.Tanh, "relu": nn.ReLU, "mish": nn.Mish}[name]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}") from None


def _mlp(input_dim: int, output_dim: int, config: SacConfig) -> nn.Sequential:
    """``n_layers`` hidden layers of ``hidden_dim``, then a linear head."""
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(config.n_layers):
        layers += [nn.Linear(width, config.hidden_dim), _activation(config.activation)]
        width = config.hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class LatentActor(nn.Module):
    """Maps observation features to a distribution over latent-noise actions."""

    def __init__(self, config: SacConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = _mlp(config.actor_feature_dim, 2 * config.latent_dim, config)

    def _mean_log_std(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.trunk(features).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a latent-noise action and its log-probability.

        Returns:
            ``(w, log_prob)`` where ``w`` is ``(B, latent_dim)`` in raw noise
            units and ``log_prob`` is ``(B, 1)``, the density of the squashed
            variable *before* scaling by ``action_magnitude`` -- see the module
            docstring for why the scale constant is deliberately excluded.
        """
        mean, log_std = self._mean_log_std(features)
        normal = torch.randn_like(mean)
        pre_tanh = mean + normal * log_std.exp()
        squashed = torch.tanh(pre_tanh)
        # Gaussian density, then the tanh change of variables. The
        # log(1 - tanh^2) term is written through softplus so it stays finite
        # when the sample saturates.
        log_prob = (
            -0.5 * normal.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        ).sum(-1, keepdim=True)
        log_prob = log_prob - (
            2 * (math.log(2) - pre_tanh - nn.functional.softplus(-2 * pre_tanh))
        ).sum(-1, keepdim=True)
        return squashed * self.config.action_magnitude, log_prob

    @torch.no_grad()
    def act(self, features: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """The latent-noise action to play, ``(B, latent_dim)``.

        ``deterministic`` takes the distribution's mode, which is what the
        paired oracle scores -- the base policy it is compared against is itself
        evaluated without sampling noise.
        """
        if not deterministic:
            return self(features)[0]
        mean, _ = self._mean_log_std(features)
        return torch.tanh(mean) * self.config.action_magnitude


class TwinCritic(nn.Module):
    """``n_critics`` independent Q functions over ``(features, w)``."""

    def __init__(self, config: SacConfig) -> None:
        super().__init__()
        self.nets = nn.ModuleList(
            _mlp(config.critic_feature_dim + config.latent_dim, 1, config)
            for _ in range(config.n_critics)
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``(n_critics, B, 1)`` values."""
        joined = torch.cat([features, action], dim=-1)
        return torch.stack([net(joined) for net in self.nets])


class LatentSac:
    """The learner: actor, twin critics, and an auto-tuned temperature."""

    def __init__(self, config: SacConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        # Deliberately does not seed torch. The caller has already done so, and
        # reseeding here would also reset the stream the warmup noise and the
        # denoiser draw from -- a constructor quietly deciding what the rest of
        # the run samples. ``config.seed`` is carried for the record only.
        self.actor = LatentActor(config).to(self.device)
        self.critic = TwinCritic(config).to(self.device)
        self.critic_target = TwinCritic(config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.log_temperature = torch.tensor(
            math.log(config.init_temperature), device=self.device, requires_grad=True
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=config.temperature_lr
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on each of critic, actor and temperature.

        Args:
            batch: ``actor_features``, ``critic_features``, ``action``,
                ``reward``, ``done``, ``next_actor_features``,
                ``next_critic_features``, all already on the device. ``done``
                marks the end of an episode, at which point the bootstrap is
                cut and the ``next_*`` features are not read -- see
                :mod:`~pick_and_place.dsrl.replay` for why that is the correct
                reading of this environment's step budget rather than a
                convenient one.

        Returns:
            Scalar diagnostics, all detached.
        """
        temperature = self.temperature.detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor(batch["next_actor_features"])
            next_value = self.critic_target(batch["next_critic_features"], next_action)
            next_value = next_value.min(dim=0).values - temperature * next_log_prob
            target = batch["reward"] + self.config.gamma * (1.0 - batch["done"]) * next_value

        values = self.critic(batch["critic_features"], batch["action"])
        critic_loss = sum(
            nn.functional.mse_loss(value, target) for value in values
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # The critic is frozen for the actor step so the actor's gradient does
        # not also drag Q toward whatever it happens to prefer.
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        action, log_prob = self.actor(batch["actor_features"])
        actor_value = self.critic(batch["critic_features"], action).min(dim=0).values
        actor_loss = (temperature * log_prob - actor_value).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        if self.config.auto_temperature:
            temperature_loss = -(
                self.log_temperature * (log_prob.detach() + self.config.target_entropy)
            ).mean()
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()

        with torch.no_grad():
            for online, target_parameter in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                target_parameter.lerp_(online, self.config.tau)

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "temperature": float(temperature),
            "entropy": float(-log_prob.detach().mean()),
            "q_mean": float(values.detach().mean()),
            "target_mean": float(target.mean()),
            "action_abs_mean": float(action.detach().abs().mean()),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "config": vars(self.config),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
