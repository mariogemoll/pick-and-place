# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

import numpy as np
import pytest
import torch

from pick_and_place.policies.dataset_export import (
    load_bounds,
    load_manifest,
    normalize,
    resolve_recording_hw,
    unnormalize,
)
from pick_and_place.policies.flow_image_policy import (
    FlowImagePolicyController,
    carry_noise,
    stack_cameras,
    summarize_smoothness,
)
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE

ACTION_DIM = 6
STATE_DIM = 6
PREDICTION_STEPS = 4
OBSERVATION_STEPS = 2
INTEGRATION_STEPS = 3
IMAGE_HW = (8, 8)


class StubUnet:
    """A velocity field that drives every sample onto a fixed constant.

    The controller integrates ``values += velocity / integration_steps``, so
    scaling the remaining distance by that many steps lands on ``target`` after
    the first step and holds there, whatever noise was drawn.
    """

    def __init__(self, target: float, integration_steps: int) -> None:
        self.target = target
        self.integration_steps = integration_steps
        self.calls = 0

    def __call__(self, values, time, condition):
        del time, condition
        self.calls += 1
        return (torch.full_like(values, self.target) - values) * self.integration_steps


class StubModel:
    """Stands in for FlowImageUnet1D without loading a vision backbone."""

    def __init__(self, target: float = 0.0, integration_steps: int = INTEGRATION_STEPS) -> None:
        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
        self.prediction_steps = PREDICTION_STEPS
        self.observation_steps = OBSERVATION_STEPS
        self.unet = StubUnet(target, integration_steps)
        self.encoded: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def encode_observation(self, images, states):
        self.encoded.append((tuple(images.shape), tuple(states.shape)))
        return torch.zeros(1, 1)


def make_bounds() -> dict[str, np.ndarray]:
    return {
        "obs_min": np.zeros(STATE_DIM, dtype=np.float32),
        "obs_max": np.full(STATE_DIM, 100.0, dtype=np.float32),
        "action_min": np.zeros(ACTION_DIM, dtype=np.float32),
        "action_max": np.full(ACTION_DIM, 10.0, dtype=np.float32),
    }


def make_controller(*, act_steps: int = 2, target: float = 0.0) -> FlowImagePolicyController:
    return FlowImagePolicyController(
        StubModel(target),
        make_bounds(),
        act_steps=act_steps,
        integration_steps=INTEGRATION_STEPS,
        device=torch.device("cpu"),
        seed=0,
        policy_hz=10.0,
        image_hw=IMAGE_HW,
    )


def make_observation(value: int = 0) -> dict[str, np.ndarray]:
    height, width = IMAGE_HW
    return {
        STATE_FEATURE: np.full(STATE_DIM, 50.0, dtype=np.float32),
        OVERHEAD_FEATURE: np.full((height, width, 3), value, dtype=np.uint8),
        WRIST_FEATURE: np.full((height, width, 3), value + 1, dtype=np.uint8),
    }


def test_normalization_round_trips_through_the_export_bounds():
    minimum = np.array([-1.0, 0.0], dtype=np.float32)
    maximum = np.array([1.0, 100.0], dtype=np.float32)
    values = np.array([0.5, 25.0], dtype=np.float32)
    assert np.allclose(unnormalize(normalize(values, minimum, maximum), minimum, maximum), values)


def test_normalization_pins_degenerate_dimensions_to_the_minimum():
    minimum = np.array([7.0], dtype=np.float32)
    maximum = np.array([7.0], dtype=np.float32)
    assert normalize(np.array([7.0], dtype=np.float32), minimum, maximum) == 0
    assert unnormalize(np.array([0.0], dtype=np.float32), minimum, maximum) == 7.0


def test_stack_cameras_puts_overhead_before_wrist_on_the_channel_axis():
    stacked = stack_cameras(make_observation(value=10))
    assert stacked.shape == (6, *IMAGE_HW)
    assert np.all(stacked[:3] == 10)
    assert np.all(stacked[3:] == 11)


def test_act_steps_must_fit_within_the_predicted_horizon():
    with pytest.raises(ValueError, match="act_steps"):
        make_controller(act_steps=PREDICTION_STEPS + 1)
    with pytest.raises(ValueError, match="act_steps"):
        make_controller(act_steps=0)


def test_a_horizon_is_generated_only_when_the_queue_runs_dry():
    controller = make_controller(act_steps=2)
    for _ in range(2):
        controller.act(make_observation())
    assert controller.model.unet.calls == INTEGRATION_STEPS  # one query

    controller.act(make_observation())
    assert controller.model.unet.calls == 2 * INTEGRATION_STEPS  # the queue emptied


def test_only_the_generating_tick_reports_a_prediction():
    controller = make_controller(act_steps=2)
    controller.act(make_observation())
    assert controller.latest_prediction is not None
    assert controller.latest_prediction.shape == (PREDICTION_STEPS, ACTION_DIM)

    controller.act(make_observation())
    assert controller.latest_prediction is None


def test_the_generating_tick_reports_the_whole_integration_path():
    controller = make_controller(act_steps=2)
    controller.act(make_observation())

    path = controller.latest_path
    assert path is not None
    # The noise it started from, then one state per Euler step.
    assert path.shape == (INTEGRATION_STEPS + 1, PREDICTION_STEPS, ACTION_DIM)

    controller.act(make_observation())
    assert controller.latest_path is None


def test_the_path_starts_at_the_noise_and_ends_at_the_reported_prediction():
    controller = make_controller(act_steps=2, target=0.25)
    controller.act(make_observation())
    path, prediction = controller.latest_path, controller.latest_prediction
    assert path is not None and prediction is not None

    # The stub drives every sample onto its target after the first Euler step,
    # so the first row is still the raw draw and the rest are already there.
    assert not np.allclose(path[0], 0.25)
    assert np.allclose(path[-1], 0.25)
    # The last row is what the controller clips and unnormalizes into commands.
    bounds = make_bounds()
    expected = unnormalize(np.clip(path[-1], -1, 1), bounds["action_min"], bounds["action_max"])
    assert np.allclose(prediction, expected)


def test_the_first_observation_fills_the_missing_history():
    controller = make_controller()
    controller.act(make_observation())
    images_shape, states_shape = controller.model.encoded[0]
    assert images_shape == (1, OBSERVATION_STEPS, 6, *IMAGE_HW)
    assert states_shape == (1, OBSERVATION_STEPS, STATE_DIM)


def test_actions_come_back_in_the_export_units():
    # The stub integrates to +1 in normalized space, the top of the action range.
    controller = make_controller(act_steps=1, target=1.0)
    action = controller.act(make_observation())
    assert action.shape == (ACTION_DIM,)
    assert np.allclose(action, 10.0, atol=1e-5)


def test_out_of_range_samples_are_clipped_and_counted():
    controller = make_controller(act_steps=1, target=2.0)
    controller.act(make_observation())
    assert controller.clipped_fraction == 1.0


def test_reset_drops_the_queue_and_the_history():
    controller = make_controller(act_steps=2)
    controller.act(make_observation())
    controller.reset()
    assert not controller.actions
    assert not controller.images
    assert controller.latest_prediction is None


def test_resolve_recording_hw_reads_the_export_beside_the_normalization(tmp_path):
    (tmp_path / "export.json").write_text(json.dumps({"source_video_hw": [720, 960]}))
    assert resolve_recording_hw(tmp_path / "normalization.npz") == (720, 960)


def test_resolve_recording_hw_prefers_an_explicit_override(tmp_path):
    assert resolve_recording_hw(tmp_path / "normalization.npz", (480, 640)) == (480, 640)


def test_resolve_recording_hw_rejects_a_missing_export(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_recording_hw(tmp_path / "normalization.npz")


def test_export_readers_load_the_manifest_and_the_bounds(tmp_path):
    (tmp_path / "export.json").write_text(json.dumps({"fps": 10, "image_size": [224, 224]}))
    bounds = make_bounds()
    np.savez(tmp_path / "normalization.npz", **bounds)

    assert load_manifest(tmp_path)["image_size"] == [224, 224]
    loaded = load_bounds(tmp_path)
    assert set(loaded) == set(bounds)
    assert np.array_equal(loaded["action_max"], bounds["action_max"])


def test_carried_noise_keeps_the_overlapping_steps_and_refreshes_the_tail():
    previous = torch.arange(PREDICTION_STEPS, dtype=torch.float32).reshape(1, -1, 1)
    previous = previous.expand(1, PREDICTION_STEPS, ACTION_DIM).contiguous()
    fresh = torch.full_like(previous, -1.0)
    independent = torch.full_like(previous, 99.0)
    shift = 2

    carried = carry_noise(previous, fresh, independent, shift=shift, correlation=1.0)

    # Step i of the new horizon is step i + shift of the old one, so the two
    # queries start from the same latent wherever they describe the same time.
    assert torch.allclose(carried[0, : PREDICTION_STEPS - shift, 0], previous[0, shift:, 0])
    assert torch.allclose(carried[0, PREDICTION_STEPS - shift :, 0], independent[0, -shift:, 0])


def test_uncorrelated_noise_is_the_fresh_draw():
    previous = torch.ones(1, PREDICTION_STEPS, ACTION_DIM)
    fresh = torch.zeros(1, PREDICTION_STEPS, ACTION_DIM)
    independent = torch.full((1, PREDICTION_STEPS, ACTION_DIM), 5.0)

    assert torch.equal(carry_noise(previous, fresh, independent, shift=2, correlation=0.0), fresh)
    assert torch.equal(carry_noise(None, fresh, independent, shift=2, correlation=1.0), fresh)


def test_partly_carried_noise_stays_a_standard_normal_sample():
    torch.manual_seed(0)
    shape = (1, PREDICTION_STEPS, ACTION_DIM)
    variances = []
    for _ in range(400):
        carried = carry_noise(
            torch.randn(*shape),
            torch.randn(*shape),
            torch.randn(*shape),
            shift=2,
            correlation=0.6,
        )
        variances.append(float(carried.var()))
    # The spherical mix preserves unit variance, which is what the velocity
    # field was trained to transport.
    assert abs(np.mean(variances) - 1.0) < 0.05


def test_noise_correlation_must_be_a_fraction():
    with pytest.raises(ValueError, match="noise_correlation"):
        FlowImagePolicyController(
            StubModel(),
            make_bounds(),
            act_steps=2,
            integration_steps=INTEGRATION_STEPS,
            device=torch.device("cpu"),
            seed=0,
            policy_hz=10.0,
            image_hw=IMAGE_HW,
            noise_correlation=1.5,
        )


def test_smoothness_separates_seam_steps_from_interior_steps():
    # Two joints, flat within each chunk and jumping 10 degrees at the seam.
    commands = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    requeried = np.array([True, False, True, False])

    summary = summarize_smoothness(commands, requeried, joints=2)

    assert summary["interior_step_deg"] == 0.0
    assert summary["boundary_step_deg"] == 10.0
    assert summary["max_step_deg"] == 10.0
