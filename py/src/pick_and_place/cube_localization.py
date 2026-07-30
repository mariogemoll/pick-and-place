# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""ViT encoder and regression head for supervised cube-position localization.

Ceiling probe for docs/DPPO_CLOSED_LOOP_STALL_HANDOFF.md's open question: can
the cube be localized to grasp precision from a 96x96 image at all? This
reimplements the DPPO visual encoder's architecture (patch embed "embed2",
patch size 8, depth 1, embed dim 128, 4 heads) from
``third_party/dppo/model/common/vit.py``, rather than importing that module,
because the vendored DPPO submodule pins an incompatible Torch/Gym stack (see
docs/PYTHON_ENVIRONMENT_BOUNDARIES.md). The two are architecturally equivalent
— same patch embedding, depth and width — but this one is trained from scratch
for a single regression objective, so exact numerical parity with the policy's
attention implementation does not matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.init import trunc_normal_


@dataclass(frozen=True)
class ViTConfig:
    patch_size: int = 8
    depth: int = 1
    embed_dim: int = 128
    num_heads: int = 4


class _PatchEmbed(nn.Module):
    """The "embed2" patch embedding: two strided convs, no learned norm."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        if patch_size != 8:
            raise NotImplementedError("only the embed2 8x8 patch embedding is implemented")
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed(x)
        return y.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)


class _TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ViTEncoder(nn.Module):
    """Patch embed + transformer stack; output is one embedding per patch.

    Input images are ``uint8`` or float pixels in ``[0, 255]``; normalization
    to ``[-0.5, 0.5]`` happens inside :meth:`forward`, matching the policy
    encoder's convention.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int = 96,
        config: ViTConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or ViTConfig()
        self.patch_embed = _PatchEmbed(in_channels, config.embed_dim, config.patch_size)
        with torch.no_grad():
            probe = self.patch_embed(torch.zeros(1, in_channels, image_size, image_size))
        num_patches = probe.shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        self.layers = nn.ModuleList(
            [_TransformerLayer(config.embed_dim, config.num_heads) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.num_patches = num_patches
        self.embed_dim = config.embed_dim

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float() / 255.0 - 0.5
        x = self.patch_embed(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CubeLocalizationHead(nn.Module):
    """The encoder above plus a two-layer MLP regressing cube position."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        image_size: int = 96,
        output_dim: int = 3,
        vit_config: ViTConfig | None = None,
    ) -> None:
        super().__init__()
        vit_config = vit_config or ViTConfig()
        self.encoder = ViTEncoder(in_channels=in_channels, image_size=image_size, config=vit_config)
        embed_dim = vit_config.embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )
        self.mlp.apply(_init_weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: ``(batch, in_channels, image_size, image_size)``."""
        pooled = self.encoder(images).mean(dim=1)
        return self.mlp(pooled)
