# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A DPPO policy that fine-tunes the arm joints and leaves the gripper alone.

The gripper command is not a smooth control like the five arm joints. Measured
over the training export, it has two attractors -- jaws closed on the cube at
about -0.75 normalized (38% of samples) and fully open at +1.0 (29%) -- with the
remaining third in transit between them as the jaws open or close over several
control ticks.

A denoising step models each action dimension as a Gaussian. A Gaussian whose
mean is pushed between two modes puts its mass in the valley between them, which
for this dimension means a half-closed gripper that touches the cube without
gripping it. That is exactly the measured failure of every fine-tuning run so
far: against the pretrained policy's 0.758 contact / 0.758 lift, the degraded
policy scores contact 0.891 -- *higher* -- and lift 0.203. It reaches better and
grasps not at all.

Zeroing the gripper's log-probability removes it from the importance ratio, so
it cancels exactly between the old and new policy and carries no gradient. The
behavior-cloned gripper policy is left intact while PPO still shapes the reach,
carry and release. The dimension is masked rather than sliced away so every
buffer in the vendored trainer keeps its declared ``action_dim`` shape.
"""

from __future__ import annotations

import torch
from model.diffusion.diffusion_ppo import PPODiffusion

# The six-dimensional action is five arm joints followed by the gripper.
ARM_ACTION_DIMS = 5


def _mask_gripper(log_prob: torch.Tensor, arm_dims: int) -> torch.Tensor:
    """Zero the trailing (gripper) action dimensions of a log-probability."""
    masked = log_prob.clone()
    masked[..., arm_dims:] = 0.0
    return masked


class ArmOnlyPPODiffusion(PPODiffusion):
    """``PPODiffusion`` whose policy ratio ignores the gripper dimension."""

    def __init__(self, *args, arm_action_dims: int = ARM_ACTION_DIMS, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm_action_dims = arm_action_dims

    def get_logprobs(self, *args, get_ent: bool = False, **kwargs):
        result = super().get_logprobs(*args, get_ent=get_ent, **kwargs)
        if get_ent:
            log_prob, entropy = result
            return _mask_gripper(log_prob, self.arm_action_dims), entropy
        return _mask_gripper(result, self.arm_action_dims)

    def get_logprobs_subsample(self, *args, get_ent: bool = False, **kwargs):
        result = super().get_logprobs_subsample(*args, get_ent=get_ent, **kwargs)
        if get_ent:
            log_prob, entropy = result
            return _mask_gripper(log_prob, self.arm_action_dims), entropy
        return _mask_gripper(result, self.arm_action_dims)
