# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""DPPO reinforcement-learning fine-tuning of this project's pretrained policies.

The pieces here exist to feed the vendored DPPO fine-tuner (``third_party/dppo``)
this project's simulated pick-and-place task:

- :mod:`pick_and_place.runtime.training_scenes` draws training resets from the same
  declared distribution as the frozen evaluation manifests, on a disjoint seed
  stream.
- :mod:`pick_and_place.dppo_rl.observations` says what one observation timestep
  is for each policy family, and what a normalized action means.
- :mod:`pick_and_place.dppo_rl.env` presents one episode in DPPO's observation
  and action convention, with actions in ``[-1, 1]``.
- :mod:`pick_and_place.dppo_rl.vector_env` runs many of those in worker
  processes and exposes the small interface DPPO's training agent calls.
- :mod:`pick_and_place.dppo_rl.agent` binds that vectorized env into
  ``TrainPPOImgDiffusionAgent`` in place of the vendored robomimic/d4rl builder.

Two policy families are fine-tuned through it. The visual Diffusion Policy uses
DPPO's own diffusion model unmodified. The state flow policy substitutes a
transition kernel -- :mod:`pick_and_place.dppo_rl.flow_ppo`, over the actor
adapter in :mod:`pick_and_place.dppo_rl.flow_actor` -- because integrating a
flow ODE has no per-step likelihood for PPO to differentiate, while the SDE with
the same marginals does.

Only the model, actor and agent modules import DPPO itself, so the env pieces
stay importable (and testable) without ``third_party/dppo`` on the path.
"""
