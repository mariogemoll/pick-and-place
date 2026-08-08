# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The pretrained Diffusion Policy as a deterministic function of its input noise.

DSRL's premise is that ``a = pi_dp(s, w)`` can be folded into the environment:
the policy's weights never move, and RL only picks which point of the
latent-noise space to denoise. That requires the map to be deterministic given
``w``, and DPPO's DDIM sampler already is -- ``DiffusionModel.forward`` pins
``ddim_eta = 0`` and then zeroes the per-step standard deviation, so the single
source of randomness in a rollout is the draw of ``x_T``. This module makes that
draw an argument instead of a side effect.

The ``base_eta: 1`` in the fine-tuning config is unrelated: it configures
``PPODiffusion``'s learnable exploration noise over the fine-tuned denoising
steps, which the evaluation path never reaches. DDPM sampling (``use_ddim:
False``) *does* inject fresh noise at every step and cannot be steered by
``x_T`` alone, so :func:`denoise` refuses it rather than silently learning
against a stochastic map.

Two things are exposed:

- :func:`denoise` -- the sampling loop, given the noise.
- :func:`visual_features` -- the conditioning vector the checkpoint's own U-Net
  is handed, which is what the latent-noise actor reads. Reusing it means the
  actor needs no vision encoder of its own and trains no pixels: the frozen ViT
  is run once per environment step for the rollout and its output is cached.
  The paper does the same thing, feeding "the image features learned by
  ``pi_dp``'s pretrained ResNet encoder" to both actor and critic.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch


def latent_shape(model: Any) -> tuple[int, int]:
    """The shape of one latent-noise action, ``(horizon_steps, action_dim)``.

    This is the whole predicted chunk, not the executed prefix: the denoiser
    produces all ``horizon_steps`` from one ``x_T``, so the noise that decides
    the executed actions has to cover the full horizon.
    """
    return int(model.horizon_steps), int(model.action_dim)


def _actor_network(model: Any) -> Any:
    """The ``VisionUnet1D`` carrying the pretrained weights.

    ``DiffusionModel`` calls it ``network``; the VPG/PPO subclasses copy it to
    ``actor`` and keep a trainable ``actor_ft`` beside it. A DSRL run steers a
    *pretrained* checkpoint, where the two are identical, but ``actor`` is the
    frozen one by construction so it is the one to read features from.
    """
    network = getattr(model, "actor", None)
    return model.network if network is None else network


@torch.no_grad()
def visual_features(model: Any, cond: dict[str, torch.Tensor]) -> torch.Tensor:
    """The conditioning vector the checkpoint's U-Net sees, ``(B, feature_dim)``.

    Mirrors ``VisionUnet1D.forward`` up to the point where it hands
    ``cat([visual feature, state])`` to the denoising blocks, so this is the
    diffusion policy's own learned summary of the observation rather than a
    second, separately-trained encoder.

    Args:
        model: An instantiated DPPO diffusion model.
        cond: ``{"state": (B, To, Do), "rgb": (B, To, C, H, W)}`` as the
            environment emits it, already on the model's device.

    Returns:
        ``(B, spatial_emb * num_img + cond_dim)`` float32 features.
    """
    from pick_and_place.policies.diffusion_policy_pretrain import batched_vision_forward

    network = _actor_network(model)
    if getattr(network, "num_img", 0) != 2 or not hasattr(network, "compress1"):
        raise ValueError(
            "visual_features expects the two-camera spatial-embedding VisionUnet1D "
            "this project pretrains; got "
            f"num_img={getattr(network, 'num_img', None)}. A different encoder needs "
            "its own feature path rather than this one silently returning the wrong "
            "conditioning."
        )
    rgb = cond["rgb"]
    batch = rgb.shape[0]
    height, width = rgb.shape[-2:]
    rgb = rgb[:, -network.img_cond_steps :]
    steps = rgb.shape[1]
    # History is concatenated along channels, per camera -- the same regrouping
    # the network does, kept here rather than imported because the upstream copy
    # lives inside a closure.
    rgb = rgb.reshape(batch, steps, network.num_img, 3, height, width)
    rgb = rgb.permute(0, 2, 1, 3, 4, 5).reshape(
        batch, network.num_img, steps * 3, height, width
    )
    state = cond["state"].reshape(batch, -1)
    if hasattr(network, "cond_mlp"):
        state = network.cond_mlp(state)
    feature = batched_vision_forward(network, rgb.float(), state)
    return torch.cat([feature, state], dim=-1)


def feature_dim(model: Any) -> int:
    """Width of :func:`visual_features` for this checkpoint.

    Read off the network rather than recomputed from the config, so a mismatch
    between the two is impossible. ``VisionUnet1D`` keeps no ``cond_dim``
    attribute, but it hands that value to ``SpatialEmb`` as ``prop_dim`` and the
    projection width is the last axis of the embedding's weight.
    """
    network = _actor_network(model)
    if hasattr(network, "cond_mlp"):
        state_dim = int(network.cond_mlp[-1].out_features)
    else:
        state_dim = int(network.compress1.prop_dim)
    visual_dim = int(network.compress1.weight.shape[-1]) * int(network.num_img)
    return visual_dim + state_dim


@torch.no_grad()
def denoise(model: Any, cond: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    """Run the checkpoint's DDIM sampler from a supplied ``x_T``.

    A transcription of ``DiffusionModel.forward`` with two changes: the initial
    ``torch.randn`` becomes the ``noise`` argument, and the per-step noise term
    is dropped rather than multiplied by a zero standard deviation. Everything
    else -- the timestep schedule, ``p_mean_var`` (and therefore the denoised
    clipping the config sets), the final action clip -- is the model's own, so
    ``denoise(model, cond, torch.randn(...))`` reproduces the checkpoint's
    ordinary behavior exactly.

    Args:
        model: An instantiated DPPO diffusion model with ``use_ddim`` set.
        cond: ``{"state": ..., "rgb": ...}`` on the model's device.
        noise: ``(B, horizon_steps, action_dim)`` latent-noise action.

    Returns:
        ``(B, horizon_steps, action_dim)`` normalized action chunk.
    """
    from model.diffusion.sampling import make_timesteps

    if not getattr(model, "use_ddim", False):
        raise ValueError(
            "DSRL needs a deterministic map from noise to action, and DDPM sampling "
            "draws fresh noise at every denoising step. Set use_ddim: True (the "
            "convention this project's checkpoints are evaluated under) or extend "
            "the latent action to cover every step's draw."
        )
    batch = noise.shape[0]
    expected = (batch, *latent_shape(model))
    if tuple(noise.shape) != expected:
        raise ValueError(f"noise must be {expected}, got {tuple(noise.shape)}")

    # ``deterministic`` is not cosmetic here, and omitting it does not raise.
    # Under DDIM, ``DiffusionVPG.p_mean_var`` sets ``etas`` to zero when it is
    # true and to ``self.eta(cond)`` when it is false -- and ``etas`` decides
    # ``sigma``, which enters the *mean* through
    # ``dir_xt = (1 - alpha_prev - sigma^2).sqrt() * noise``. With the config's
    # ``base_eta: 1`` the two settings therefore trace different trajectories,
    # and the wrong one would be a silently different policy rather than an
    # error. True is what ``check_dppo_rl_env.py`` scores with, so it is what
    # the latent space has to be defined against.
    #
    # The base ``DiffusionModel`` accepts no such argument and returns a pair,
    # while the VPG and PPO subclasses take it and return ``(mu, logvar, etas)``.
    # Both are supported by reading the signature rather than the class.
    signature = inspect.signature(model.p_mean_var)
    extra = {"deterministic": True} if "deterministic" in signature.parameters else {}

    x = noise.to(device=model.betas.device, dtype=torch.float32)
    timesteps = model.ddim_t
    for index, timestep in enumerate(timesteps):
        predicted = model.p_mean_var(
            x=x,
            t=make_timesteps(batch, timestep, x.device),
            cond=cond,
            index=make_timesteps(batch, index, x.device),
            **extra,
        )
        # DDIM with eta pinned to zero: the posterior standard deviation is
        # identically zero, so the sample is its mean and the chunk is a
        # function of x_T alone.
        x = predicted[0]
        if model.final_action_clip_value is not None and index == len(timesteps) - 1:
            x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
    return x
