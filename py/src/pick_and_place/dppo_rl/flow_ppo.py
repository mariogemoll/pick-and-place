# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""PPO fine-tuning of a conditional flow-matching policy.

DPPO's contribution is to read a diffusion policy's denoising chain as a small
MDP: each step is an action drawn from a known Gaussian, so the whole sampler
has a likelihood and PPO can differentiate it. Everything downstream of that --
the clipped surrogate, the denoising discount, the advantage handling -- is
generic in the transition kernel, which is why this class changes only the
kernel and inherits :class:`PPODiffusion`'s loss unmodified.

A flow policy samples by integrating an ODE, which has no per-step density at
all. :func:`~pick_and_place.policies.flow_matching.flow_sde_transition` supplies
one: the SDE whose marginals match the trained ODE's, discretized by
Euler-Maruyama over the same grid the closed-loop runner integrates on. The
chain is then structurally what DPPO already fine-tunes -- ``flow_steps``
Gaussian transitions, of which the last ``ft_denoising_steps`` carry gradient --
and the pretrained weights are used exactly as behavior cloning left them.

Two properties of the resulting chain differ from the diffusion one and matter:

- the per-step standard deviation is the flow SDE's, largest at the start of
  the chain and vanishing as it ends, so the step that emits the action is the
  quietest rather than the loudest;
- the observation is privileged task state rather than pixels, so the actor is
  small and the rollout does not render.
"""

from __future__ import annotations

import logging

import torch
from model.diffusion.diffusion import Sample
from model.diffusion.diffusion_ppo import PPODiffusion
from torch.distributions import Normal

from pick_and_place.policies.flow_matching import flow_sde_transition

log = logging.getLogger(__name__)

# The clamp `PPODiffusion.loss` applies to every log-probability before the
# importance ratio. It is upstream's, and it is invisible until the transition
# standard deviation is small: a Gaussian with std 0.05 has a peak log-density
# of 2.08, so below roughly that width most dimensions clamp, both ratio terms
# take the same constant, and the update quietly reads as "nothing changed".
LOGPROB_CLAMP = (-5.0, 2.0)


class FlowPPO(PPODiffusion):
    """DPPO over the flow policy's stochastic sampler."""

    def __init__(
        self,
        *,
        flow_steps: int,
        sampling_noise_scale: float,
        **kwargs,
    ) -> None:
        if flow_steps < 1:
            raise ValueError("flow_steps must be positive")
        if sampling_noise_scale < 0.0:
            raise ValueError("sampling_noise_scale must be non-negative")
        super().__init__(
            denoising_steps=flow_steps,
            use_ddim=False,
            # Inherited annealing and logging read the sampling noise through
            # this attribute. For the flow sampler it is a scale rather than a
            # floor: the SDE's own schedule multiplies it.
            min_sampling_denoising_std=sampling_noise_scale,
            **kwargs,
        )
        self.flow_steps = flow_steps
        self.step_size = 1.0 / flow_steps
        # Integration times of the chain: step i maps x(t_i) to x(t_i + dt).
        self.flow_times = torch.arange(flow_steps, device=self.device) * self.step_size
        self.first_finetuned_step = flow_steps - self.ft_denoising_steps

    # ---------- the transition kernel ----------#

    def _transition(
        self,
        values: torch.Tensor,
        step_indices: torch.Tensor,
        cond: dict,
        *,
        network: torch.nn.Module,
        noise_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation of one Euler-Maruyama step per row."""
        times = self.flow_times.to(values.device)[step_indices]
        velocity = network(values, times, cond)
        return flow_sde_transition(
            values,
            velocity,
            times.reshape(-1, 1, 1),
            step_size=self.step_size,
            noise_scale=noise_scale,
        )

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ) -> Sample:
        """Integrate the flow, and return the tail of the chain PPO fine-tunes.

        With ``deterministic`` the noise scale is zero, which is the plain Euler
        integration the closed-loop evaluation uses -- the same relationship
        DDIM sampling has to its noisy training-time counterpart.
        """
        sample_data = cond["state"]
        batch = len(sample_data)
        device = sample_data.device
        noise_scale = 0.0 if deterministic else self.get_min_sampling_denoising_std()
        values = torch.randn(
            (batch, self.horizon_steps, self.action_dim), device=device
        )
        chain = [] if return_chain else None
        if return_chain and self.ft_denoising_steps == self.flow_steps:
            chain.append(values)
        for step in range(self.flow_steps):
            finetuned = step >= self.first_finetuned_step
            network = self.actor if (use_base_policy or not finetuned) else self.actor_ft
            mean, std = self._transition(
                values,
                torch.full((batch,), step, dtype=torch.long, device=device),
                cond,
                network=network,
                noise_scale=noise_scale,
            )
            noise = torch.randn_like(values).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            values = mean + std * noise
            if self.final_action_clip_value is not None and step == self.flow_steps - 1:
                # The closed-loop runner clips the generated chunk onto the
                # normalized action bounds before unnormalizing it, so training
                # rollouts have to see the same actions the deployed policy
                # commands.
                values = torch.clamp(
                    values, -self.final_action_clip_value, self.final_action_clip_value
                )
            if return_chain and step >= self.first_finetuned_step - 1:
                chain.append(values)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(values, chain)

    # ---------- RL training ----------#

    def _logprobs(
        self,
        cond: dict,
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        step_indices: torch.Tensor,
        use_base_policy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._transition(
            chains_prev,
            step_indices,
            cond,
            network=self.actor if use_base_policy else self.actor_ft,
            noise_scale=self.get_min_sampling_denoising_std(),
        )
        # The likelihood floor is deliberately not the sampling schedule. It
        # exists for numerical stability: the SDE's last steps are far narrower
        # than the log-probability clamp above can represent, and a density that
        # narrow turns every ratio into a clamped constant.
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        return Normal(mean, std).log_prob(chains_next), std

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """Log-probabilities of every fine-tuned transition in ``chains``."""
        steps = self.ft_denoising_steps
        cond = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }
        step_indices = torch.arange(
            self.first_finetuned_step, self.flow_steps, device=self.device
        ).repeat(len(chains))
        log_prob, std = self._logprobs(
            cond,
            chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim),
            chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim),
            step_indices,
            use_base_policy,
        )
        # Once per iteration, and the one diagnostic that says whether the
        # importance ratio can move at all: a clamped log-probability is
        # identical for the old and new policies, so a large fraction here means
        # the trust region is reading zero by construction.
        clamped = (
            (log_prob < LOGPROB_CLAMP[0]) | (log_prob > LOGPROB_CLAMP[1])
        ).float().mean()
        log.info(
            f"flow chain: transition std {std.min().item():.4f}-{std.max().item():.4f}, "
            f"log-probabilities clamped {clamped.item():.3f}"
        )
        if get_ent:
            return log_prob, std
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """Log-probabilities of one sampled transition per batch row."""
        log_prob, std = self._logprobs(
            cond,
            chains_prev,
            chains_next,
            denoising_inds + self.first_finetuned_step,
            use_base_policy,
        )
        if get_ent:
            # Stands in for DPPO's learnable DDIM eta. The flow sampler has no
            # such parameter, so this reports the width the likelihood used and
            # contributes no gradient to the entropy term.
            return log_prob, std
        return log_prob
