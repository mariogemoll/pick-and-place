# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The image flow model's boundary: shapes, gradients, and camera separation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pick_and_place.policies.flow_image_encoder import (  # noqa: E402
    CameraEncoder,
    FlowImageUnet1D,
    SpatialSoftmax,
    model_config,
)

ACTION_DIM = 6
STATE_DIM = 6
PREDICTION_STEPS = 16
OBSERVATION_STEPS = 2
CAMERAS = 2
SIZE = 64


def build() -> FlowImageUnet1D:
    return FlowImageUnet1D(
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        prediction_steps=PREDICTION_STEPS,
        observation_steps=OBSERVATION_STEPS,
        cameras=CAMERAS,
        keypoints=8,
    )


def sample(batch: int = 2):
    images = torch.randint(
        0, 255, (batch, OBSERVATION_STEPS, CAMERAS * 3, SIZE, SIZE), dtype=torch.uint8
    ).float()
    states = torch.randn(batch, OBSERVATION_STEPS, STATE_DIM)
    values = torch.randn(batch, PREDICTION_STEPS, ACTION_DIM)
    time = torch.rand(batch, 1)
    return values, time, images, states


def test_spatial_softmax_recovers_a_peak_position() -> None:
    features = torch.full((1, 1, 5, 5), -20.0)
    features[0, 0, 0, 4] = 20.0  # top-right corner
    keypoints = SpatialSoftmax(1)(features)
    assert keypoints.shape == (1, 2)
    assert keypoints[0, 0].item() == pytest.approx(1.0, abs=1e-3)
    assert keypoints[0, 1].item() == pytest.approx(-1.0, abs=1e-3)


def test_forward_returns_the_action_chunk_shape() -> None:
    model = build()
    values, time, images, states = sample()
    output = model(values, time, images, states)
    assert output.shape == (2, PREDICTION_STEPS, ACTION_DIM)


def test_every_parameter_receives_gradient() -> None:
    model = build()
    values, time, images, states = sample()
    model(values, time, images, states).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"


def test_observation_dimension_matches_the_unet_condition() -> None:
    model = build()
    _, _, images, states = sample()
    condition = model.encode_observation(images, states)
    assert condition.shape == (2, model.observation_dim)
    assert model.unet.observation_dim == model.observation_dim


def test_cameras_are_encoded_separately_not_blended() -> None:
    """Swapping the two cameras must change the condition.

    A folding bug that mixed the camera and channel axes would make the encoder
    blind to which stream a pixel came from, and this is the cheapest way to
    catch it.
    """
    model = build()
    _, _, images, states = sample(batch=1)
    swapped = torch.cat((images[:, :, 3:], images[:, :, :3]), dim=2)
    with torch.no_grad():
        original = model.encode_observation(images, states)
        exchanged = model.encode_observation(swapped, states)
    assert not torch.allclose(original, exchanged)


def test_state_history_reaches_the_condition() -> None:
    model = build()
    _, _, images, states = sample(batch=1)
    altered = states.clone()
    altered[0, 0, 0] += 5.0
    with torch.no_grad():
        assert not torch.allclose(
            model.encode_observation(images, states),
            model.encode_observation(images, altered),
        )


def test_checkpoint_config_round_trips() -> None:
    model = build()
    restored = FlowImageUnet1D(**model_config(model))
    restored.load_state_dict(model.state_dict())
    values, time, images, states = sample()
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(
            model(values, time, images, states), restored(values, time, images, states)
        )


def test_truncated_trunk_doubles_the_keypoint_map() -> None:
    """Stopping after layer3 is what buys the finer grid; check it, not the parameter count."""
    full = CameraEncoder(keypoints=8, trunk_stages=4)
    truncated = CameraEncoder(keypoints=8, trunk_stages=3)
    images = torch.randn(2, 3, SIZE, SIZE)
    assert full.trunk(images).shape[-2:] == (SIZE // 32, SIZE // 32)
    assert truncated.trunk(images).shape[-2:] == (SIZE // 16, SIZE // 16)
    # Same keypoint count out, so the U-Net's condition width is unchanged.
    assert truncated(images).shape == full(images).shape
    assert sum(p.numel() for p in truncated.parameters()) < sum(
        p.numel() for p in full.parameters()
    )


def test_truncated_trunk_round_trips_through_the_checkpoint_config() -> None:
    model = FlowImageUnet1D(
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        prediction_steps=PREDICTION_STEPS,
        observation_steps=OBSERVATION_STEPS,
        cameras=CAMERAS,
        keypoints=8,
        trunk_stages=3,
    )
    config = model_config(model)
    assert config["trunk_stages"] == 3
    FlowImageUnet1D(**config).load_state_dict(model.state_dict())


def test_checkpoints_without_trunk_stages_load_as_the_full_trunk() -> None:
    """The 300,000-update artifact predates the flag; its config must still construct."""
    config = model_config(build())
    del config["trunk_stages"]
    assert FlowImageUnet1D(**config).trunk_stages == 4


def test_unknown_trunk_stage_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        CameraEncoder(keypoints=8, trunk_stages=5)


def test_wrong_image_shape_is_rejected() -> None:
    model = build()
    _, _, images, states = sample()
    with pytest.raises(ValueError):
        model.encode_observation(images[:, :1], states)
    with pytest.raises(ValueError):
        model.encode_observation(images, states[:, :1])
