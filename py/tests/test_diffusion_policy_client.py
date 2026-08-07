# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tests for the Diffusion Policy client, its server protocol, and normalization."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.policies import diffusion_policy_client
from pick_and_place.data.diffusion_policy_dataset import normalize_min_max
from pick_and_place.policies.diffusion_policy_client import DiffusionPolicyController, resolve_recording_hw
from pick_and_place.spec.action_encoding import ActionEncoding
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE

SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diffusion_policy_server.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location("diffusion_policy_server", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = _load_server_module()


def test_client_and_server_framing_are_interoperable() -> None:
    arrays = {
        "state": np.arange(6, dtype=np.float32),
        "overhead": np.zeros((4, 4, 3), dtype=np.uint8),
        "label": np.asarray("hello"),
    }
    for writer, reader in ((diffusion_policy_client, server), (server, diffusion_policy_client)):
        buffer = io.BytesIO()
        writer.write_message(buffer, arrays)
        buffer.seek(0)
        decoded = reader.read_message(buffer)
        assert decoded is not None
        assert set(decoded) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(decoded[key], value)
        assert reader.read_message(buffer) is None


def test_read_message_rejects_truncation() -> None:
    buffer = io.BytesIO()
    diffusion_policy_client.write_message(buffer, {"x": np.zeros(3)})
    truncated = io.BytesIO(buffer.getvalue()[:-1])
    with pytest.raises(EOFError):
        diffusion_policy_client.read_message(truncated)


def test_server_normalization_inverts_the_dataset_export() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(-90.0, 90.0, size=(64, 6)).astype(np.float32)
    normalized, minimum, maximum = normalize_min_max(raw)
    round_tripped = server.unnormalize_actions(normalized, minimum, maximum)
    np.testing.assert_allclose(round_tripped, raw, atol=1e-4)
    forward = server.normalize_state(raw, minimum, maximum)
    np.testing.assert_allclose(forward, normalized, atol=1e-6)


def test_recording_resolution_comes_from_export_or_override(tmp_path: Path) -> None:
    normalization = tmp_path / "normalization.npz"
    normalization.touch()
    (tmp_path / "export.json").write_text('{"source_video_hw": [720, 960]}')

    assert resolve_recording_hw(normalization) == (720, 960)
    assert resolve_recording_hw(normalization, (480, 640)) == (480, 640)
    with pytest.raises(ValueError, match="positive"):
        resolve_recording_hw(normalization, (0, 640))


FAKE_SERVER = """
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, {server_dir!r})
from diffusion_policy_server import read_message, write_message

stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
write_message(stdout, {{
    "horizon_steps": np.asarray(4),
    "act_steps": np.asarray(2),
    "cond_steps": np.asarray(2),
    "img_cond_steps": np.asarray(2),
    "policy_hz": np.asarray(10.0),
    "obs_dim": np.asarray(6),
    "action_dim": np.asarray(6),
    "image_height": np.asarray(8),
    "image_width": np.asarray(8),
    "denoising_steps": np.asarray(100),
    "sampler": np.asarray("ddpm-100"),
    "epoch": np.asarray(500),
    "device": np.asarray("cpu"),
    "seed": np.asarray(0),
    "torch_version": np.asarray("0.0-fake"),
    "action_encoding": np.asarray({encoding!r}),
}})
queries = 0
while True:
    request = read_message(stdin)
    if request is None:
        break
    assert request["state"].shape == (2, 6)
    assert request["overhead"].shape == (2, 8, 8, 3)
    assert request["wrist"].shape == (2, 8, 8, 3)
    queries += 1
    actions = np.full((4, 6), float(queries), dtype=np.float32)
    actions += np.arange(4, dtype=np.float32)[:, None] / 10.0
    actions[0, 1] = request["state"][0, 0]
    actions[0, 2] = request["state"][1, 0]
    write_message(stdout, {{"actions": actions}})
"""


def _fake_server_command(tmp_path: Path, encoding: str = "absolute") -> list[str]:
    script = tmp_path / f"fake_server_{encoding}.py"
    script.write_text(
        FAKE_SERVER.format(server_dir=str(SERVER_SCRIPT.parent), encoding=encoding)
    )
    return [sys.executable, str(script)]


@pytest.fixture
def fake_server_command(tmp_path: Path) -> list[str]:
    return _fake_server_command(tmp_path)


def _observation(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        STATE_FEATURE: np.full(6, value, dtype=np.float32),
        OVERHEAD_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
        WRIST_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_controller_serves_chunks_and_requeries(fake_server_command: list[str]) -> None:
    controller = DiffusionPolicyController(fake_server_command, act_steps=2)
    try:
        assert controller.horizon_steps == 4
        assert controller.cond_steps == 2
        assert controller.policy_hz == 10.0
        assert controller.image_hw == (8, 8)
        assert controller.handshake["epoch"] == 500
        # Two actions per query: the integer part encodes the query count and
        # the fractional part the position within the returned horizon.
        first = controller.act(_observation())
        assert controller.latest_prediction is not None
        assert controller.latest_prediction.shape == (4, 6)
        second = controller.act(_observation())
        assert controller.latest_prediction is None
        values = [
            first[0],
            second[0],
            controller.act(_observation())[0],
            controller.act(_observation())[0],
        ]
        assert values == pytest.approx([1.0, 1.1, 2.0, 2.1])
    finally:
        controller.close()


def test_controller_integrates_a_delta_onto_the_tick_it_is_commanded_on(tmp_path: Path) -> None:
    controller = DiffusionPolicyController(
        _fake_server_command(tmp_path, "delta"), act_steps=2
    )
    try:
        assert controller.action_encoding is ActionEncoding.DELTA
        # Query 1 returns rows 1.0 and 1.1, offsets rather than commands.
        first = controller.act(_observation(1.0))
        # The chunk was predicted from a state of 1.0; by the second tick the
        # arm reads 5.0, and that is what the queued offset belongs to.
        second = controller.act(_observation(5.0))
    finally:
        controller.close()

    assert first[0] == pytest.approx(2.0)
    assert second[0] == pytest.approx(6.1)
    # Not 1.0 + 1.1: integrating a whole chunk from where it was predicted
    # would leave the arm a chunk's worth of motion behind.
    assert second[0] != pytest.approx(2.1)


def test_controller_reads_a_delta_horizon_open_loop_for_diagnostics(tmp_path: Path) -> None:
    controller = DiffusionPolicyController(
        _fake_server_command(tmp_path, "delta"), act_steps=2
    )
    try:
        commanded = controller.act(_observation(3.0))
        prediction = controller.latest_prediction
    finally:
        controller.close()

    assert prediction is not None
    # latest_prediction is in joint units, so the first row is exactly what was
    # commanded and the rest reads as if the arm never moved.
    assert prediction[0][0] == pytest.approx(commanded[0])
    assert prediction[1][0] == pytest.approx(3.0 + 1.1)


def test_controller_rejects_a_server_whose_encoding_it_does_not_know(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown action encoding"):
        DiffusionPolicyController(_fake_server_command(tmp_path, "relative"))


def test_controller_reset_discards_queued_actions(fake_server_command: list[str]) -> None:
    controller = DiffusionPolicyController(fake_server_command)
    try:
        assert controller.act_steps == 2
        first = controller.act(_observation())[0]
        controller.reset()
        second = controller.act(_observation())[0]
        assert (first, second) == pytest.approx((1.0, 2.0))
    finally:
        controller.close()


def test_controller_predict_horizon_uses_repeated_observation(
    fake_server_command: list[str],
) -> None:
    controller = DiffusionPolicyController(fake_server_command)
    try:
        actions = controller.predict_horizon(_observation(7.0), sampling_seed=42)
        assert actions.shape == (4, 6)
        assert actions[0, 1:3].tolist() == pytest.approx([7.0, 7.0])
    finally:
        controller.close()


def test_controller_tracks_observations_while_executing_queue(
    fake_server_command: list[str],
) -> None:
    controller = DiffusionPolicyController(fake_server_command, act_steps=2)
    try:
        first = controller.act(_observation(10.0))
        controller.act(_observation(20.0))
        third = controller.act(_observation(30.0))
        assert first[1:3].tolist() == pytest.approx([10.0, 10.0])
        assert third[1:3].tolist() == pytest.approx([20.0, 30.0])
    finally:
        controller.close()


def test_controller_rejects_invalid_act_steps(fake_server_command: list[str]) -> None:
    with pytest.raises(ValueError, match="act_steps"):
        DiffusionPolicyController(fake_server_command, act_steps=5)


def test_controller_reports_server_death(fake_server_command: list[str]) -> None:
    controller = DiffusionPolicyController(fake_server_command)
    try:
        controller._process.terminate()
        controller._process.wait()
        with pytest.raises(RuntimeError, match="server exited"):
            controller.act(_observation())
    finally:
        controller.close()


def test_controller_requires_all_observation_features(fake_server_command: list[str]) -> None:
    controller = DiffusionPolicyController(fake_server_command)
    try:
        observation = _observation()
        del observation[WRIST_FEATURE]
        with pytest.raises(KeyError):
            controller.act(observation)
    finally:
        controller.close()
