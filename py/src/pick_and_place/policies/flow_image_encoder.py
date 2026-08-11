# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Vision conditioning for the flow policy, over the unmodified 1D U-Net.

The state flow policy conditions on privileged simulator facts -- the cube's
position and orientation, and the target -- which no camera hands a real robot.
This module replaces that conditioning with what the two cameras see, and
changes nothing else: the CondOT path, the velocity target, the action horizons
and the temporal backbone are all the state policy's.

Each camera timestep is encoded by a ResNet18 trunk whose BatchNorm is replaced
by GroupNorm, projected to a small set of channels, and reduced to keypoint
coordinates by a spatial softmax. That is Diffusion Policy's encoder, chosen
because spatially-argmaxed keypoints keep the millimetre-scale positional
precision a grasp needs, which a pooled classification feature discards.

Both cameras and both observation timesteps are folded into the batch axis for
a single trunk call. The backbone has no cross-sample operations, so this is the
same arithmetic in a quarter of the launches -- the same trick already measured
bit-identical for the Diffusion Policy pre-training loop.
"""

from __future__ import annotations

import torch
from torch import nn

from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D


class SpatialSoftmax(nn.Module):
    """Expected image-plane position of each feature channel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        attention = torch.softmax(features.reshape(batch, channels, height * width), dim=-1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        grid = torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=-1)
        return (attention @ grid).reshape(batch, channels * 2)


def _replace_batch_norm(module: nn.Module) -> None:
    """Swap every BatchNorm2d for GroupNorm, in place.

    Batch statistics are wrong for this data: a batch holds temporally adjacent
    frames from few episodes, so its mean and variance are not the dataset's.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = min(16, channels)
            while channels % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            _replace_batch_norm(child)


#: Output channels of the ResNet18 residual stages, indexed by stage count.
_STAGE_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 512}


class CameraEncoder(nn.Module):
    """One shared ResNet18 trunk reduced to spatial-softmax keypoints.

    ``trunk_stages`` drops residual stages from the end. Stopping after
    ``layer3`` doubles the keypoint map -- 14x14 rather than 7x7 at 224 px --
    which is the grid the spatial softmax localizes over. It removes 75% of the
    trunk's weights but only ~9% of its time, ResNet stages being designed to
    cost roughly equal compute; the finer map is the reason to do it, not speed.
    """

    def __init__(
        self, keypoints: int = 32, pretrained: bool = False, trunk_stages: int = 4
    ) -> None:
        super().__init__()
        if trunk_stages not in _STAGE_CHANNELS:
            raise ValueError(f"trunk_stages must be one of {sorted(_STAGE_CHANNELS)}")
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        _replace_batch_norm(backbone)
        # children(): conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc.
        self.trunk_stages = trunk_stages
        kept = list(backbone.children())[: 4 + trunk_stages]
        self.trunk = nn.Sequential(*kept)
        self.project = nn.Conv2d(_STAGE_CHANNELS[trunk_stages], keypoints, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(keypoints)
        self.feature_dim = keypoints * 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.spatial_softmax(self.project(self.trunk(images)))


class FlowImageUnet1D(nn.Module):
    """Velocity field conditioned on camera images and proprioception.

    ``images`` arrive as ``(batch, observation_steps, cameras * 3, height,
    width)`` in the export's channel layout, and ``states`` as ``(batch,
    observation_steps, state_dim)``. The concatenated vision keypoints and
    flattened robot state become the U-Net's global condition.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        prediction_steps: int,
        observation_steps: int = 2,
        cameras: int = 2,
        keypoints: int = 32,
        pretrained_backbone: bool = False,
        trunk_stages: int = 4,
        time_embedding_dim: int = 32,
        down_dims: tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if min(action_dim, state_dim, prediction_steps, observation_steps, cameras) < 1:
            raise ValueError("model dimensions must be positive")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.prediction_steps = prediction_steps
        self.observation_steps = observation_steps
        self.cameras = cameras
        self.keypoints = keypoints
        self.pretrained_backbone = pretrained_backbone
        self.trunk_stages = trunk_stages

        self.encoder = CameraEncoder(
            keypoints=keypoints, pretrained=pretrained_backbone, trunk_stages=trunk_stages
        )
        vision_dim = self.encoder.feature_dim * cameras * observation_steps
        self.observation_dim = vision_dim + state_dim * observation_steps

        self.unet = FlowConditionalUnet1D(
            action_dim=action_dim,
            observation_dim=self.observation_dim,
            prediction_steps=prediction_steps,
            time_embedding_dim=time_embedding_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            groups=groups,
        )
        # The flat-boundary attributes the sampler and checkpoint loader read.
        self.time_embedding_dim = time_embedding_dim
        self.output_dim = prediction_steps * action_dim
        self.input_dim = self.output_dim + time_embedding_dim + self.observation_dim

    def encode_observation(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Reduce raw camera and proprioceptive history to the global condition."""
        batch = len(images)
        expected = (batch, self.observation_steps, self.cameras * 3)
        if images.shape[:3] != expected:
            raise ValueError(f"images must start with shape {expected}, got {tuple(images.shape)}")
        if states.shape != (batch, self.observation_steps, self.state_dim):
            raise ValueError(
                f"states must have shape {(batch, self.observation_steps, self.state_dim)}"
            )
        height, width = images.shape[-2:]
        # (batch, steps, cameras * 3, h, w) -> (batch * steps * cameras, 3, h, w)
        folded = images.reshape(batch * self.observation_steps * self.cameras, 3, height, width)
        features = self.encoder(folded)
        vision = features.reshape(batch, -1)
        return torch.cat((vision, states.reshape(batch, -1)), dim=1)

    def forward(
        self,
        values: torch.Tensor,
        time: torch.Tensor,
        images: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(values, time, self.encode_observation(images, states))


def model_config(model: FlowImageUnet1D) -> dict[str, object]:
    """Constructor arguments recorded in a checkpoint."""
    return {
        "action_dim": model.action_dim,
        "state_dim": model.state_dim,
        "prediction_steps": model.prediction_steps,
        "observation_steps": model.observation_steps,
        "cameras": model.cameras,
        "keypoints": model.keypoints,
        "pretrained_backbone": model.pretrained_backbone,
        "trunk_stages": model.trunk_stages,
        "time_embedding_dim": model.time_embedding_dim,
        "down_dims": model.unet.down_dims,
        "kernel_size": model.unet.kernel_size,
        "groups": model.unet.groups,
    }
