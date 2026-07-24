# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""DPPO reinforcement-learning fine-tuning of the pretrained Diffusion Policy.

The pieces here exist to feed the vendored DPPO fine-tuner (``third_party/dppo``)
this project's simulated pick-and-place task:

- :mod:`pick_and_place.dppo_rl.scenes` draws training resets from the same
  declared distribution as the frozen evaluation manifests, on a disjoint seed
  stream.
- :mod:`pick_and_place.dppo_rl.env` presents one episode in DPPO's observation
  and action convention: normalized state, stacked ``uint8`` camera tensors, and
  actions in ``[-1, 1]``.
- :mod:`pick_and_place.dppo_rl.vector_env` runs many of those in worker
  processes and exposes the small interface DPPO's training agent calls.
- :mod:`pick_and_place.dppo_rl.agent` binds that vectorized env into
  ``TrainPPOImgDiffusionAgent`` in place of the vendored robomimic/d4rl builder.

Only :mod:`~pick_and_place.dppo_rl.agent` imports DPPO itself, so the env pieces
stay importable (and testable) without ``third_party/dppo`` on the path.
"""
