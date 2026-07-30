# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import torch

from pick_and_place.cube_localization import CubeLocalizationHead, ViTConfig, ViTEncoder


def test_encoder_patch_count_matches_dppo_embed2_geometry():
    # third_party/dppo/model/common/vit.py's PatchEmbed2 gives 11x11 = 121
    # patches for a 96x96 image at patch_size 8; this reimplementation must
    # match, since it stands in for the same architecture.
    encoder = ViTEncoder(in_channels=6, image_size=96, config=ViTConfig())
    assert encoder.num_patches == 121


def test_head_regresses_xyz_from_a_batch_of_stacked_camera_images():
    model = CubeLocalizationHead(in_channels=6, image_size=96, output_dim=3)
    images = torch.randint(0, 256, (5, 6, 96, 96), dtype=torch.uint8)

    output = model(images)

    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()


def test_gradients_flow_to_both_encoder_and_head():
    model = CubeLocalizationHead(in_channels=6, image_size=96, output_dim=3)
    images = torch.randint(0, 256, (2, 6, 96, 96), dtype=torch.uint8)

    model(images).sum().backward()

    assert model.encoder.pos_embed.grad is not None
    assert model.mlp[0].weight.grad is not None
