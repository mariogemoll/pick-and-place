# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""An asymmetric critic: privileged simulator state in, value out.

The image critic DPPO ships has to regress a sparse return from two 96x96
cameras, and on this task it does not manage it. Measured on a checkpoint whose
critic had already been warmed up, its predictions correlate with realized
discounted returns at r = 0.33 -- about 11% of the variance explained even after
allowing an arbitrary rescaling. Advantages computed from it are mostly noise,
and PPO does not merely stall on noisy advantages: it exploits whatever the
critic mis-values, which is why every fine-tuning run degraded a strong
behavior-cloned policy monotonically, at a rate set only by the learning rate.

Giving the critic privileged state is the standard remedy and costs nothing at
deployment: the value function exists only during training, and the actor
(``VisionUnet1D``) reads only ``rgb`` and ``state``, so the policy that ships is
unchanged and still sees cameras and proprioception alone. The project's plan
allows exactly this -- simulator ground truth may inform reward, termination and
value, but never the policy observation.
"""

from __future__ import annotations

import torch
from model.common.critic import CriticObs


class PrivilegedCritic(CriticObs):
    """Value function over the environment's privileged observation.

    Consumes ``cond["privileged"]`` -- cube pose, target, their difference and
    distance, gripper-to-cube distance, contact/grasp flags and elapsed episode
    fraction -- flattened over the observation history.
    """

    def __init__(self, privileged_dim: int, cond_steps: int = 1, **kwargs):
        super().__init__(cond_dim=privileged_dim * cond_steps, **kwargs)
        self.privileged_dim = privileged_dim
        self.cond_steps = cond_steps

    def forward(self, cond: dict, no_augment: bool = False) -> torch.Tensor:
        del no_augment  # no images, so nothing to augment
        privileged = cond["privileged"]
        return super().forward(privileged.reshape(privileged.shape[0], -1))
