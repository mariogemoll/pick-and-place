# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""DSRL: steering the pretrained Diffusion Policy through its latent-noise space.

Where the DPPO strand fine-tunes the diffusion policy's weights with PPO, DSRL
(`arXiv:2506.15799 <https://arxiv.org/abs/2506.15799>`_) leaves them frozen and
learns *which noise to denoise*. A diffusion policy turns a draw ``w ~ N(0, I)``
into an action chunk; fixing the weights and choosing ``w`` deliberately steers
the chunk toward whichever mode of the demonstrated action distribution pays
best. The map ``a = pi_dp(s, w)`` becomes part of the environment, and what is
learned is a small policy over ``w``.

Two properties of that framing answer the two things that went wrong on the
DPPO strand, which is why it is worth a second RL attempt on the same task:

- **Nothing back-propagates through the denoising chain.** Twelve consecutive
  collapses were bought off with brakes (``update_epochs`` 2, gradient clipping)
  rather than fixed, and the surviving configuration was a lottery over seeds.
  Here the diffusion policy is called only forward, so there is no chain to
  destabilize and the deployed weights cannot degrade at all: the worst outcome
  of a bad DSRL run is a latent policy no better than ``w ~ N(0, I)``.
- **The critic gets an off-policy TD target instead of a sparse-return
  regression.** The image critic correlated with realized returns at r = 0.33,
  and the resulting advantages were ~89% noise. A latent-space Q function is an
  MLP over frozen features and a 96-dimensional noise vector, trained by
  bootstrapping off a replay buffer rather than by regressing whole episodes.

The pieces:

- :mod:`~pick_and_place.dsrl.noise_policy` presents a pretrained DPPO diffusion
  checkpoint as the deterministic function ``a = pi_dp(s, w)``, plus the frozen
  visual features its own U-Net conditions on.
- :mod:`~pick_and_place.dsrl.sac` is soft actor-critic over that latent-noise
  action space, with an optionally asymmetric critic.
- :mod:`~pick_and_place.dsrl.replay` stores transitions as *cached features*
  rather than pixels, which is what keeps an off-policy buffer affordable here.
- :mod:`~pick_and_place.dsrl.trainer` is the loop that joins them to
  :class:`~pick_and_place.dppo_rl.vector_env.DppoVectorEnv`.
- :mod:`~pick_and_place.dsrl.steerability` measures the precondition the whole
  method rests on: that ``w`` moves the action at all.

The environment is reused from :mod:`pick_and_place.dppo_rl` unchanged. It
already presents an action chunk as one step in the checkpoint's own
observation convention, which is exactly the interface DSRL wants -- the paper
likewise treats a chunk as a single action and ignores the observations inside
it.

Only :mod:`~pick_and_place.dsrl.noise_policy` imports DPPO itself, so the rest
stays importable and testable without ``third_party/dppo`` on the path.
"""
