# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tests for the DPPO policy client, its server protocol, and normalization."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from pick_and_place import dppo_policy
from pick_and_place.diffusion_policy_dataset import normalize_min_max
from pick_and_place.dppo_policy import DppoPolicyController
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
)

SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dppo_policy_server.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location("dppo_policy_server", SERVER_SCRIPT)
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
    for writer, reader in ((dppo_policy, server), (server, dppo_policy)):
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
    dppo_policy.write_message(buffer, {"x": np.zeros(3)})
    truncated = io.BytesIO(buffer.getvalue()[:-1])
    with pytest.raises(EOFError):
        dppo_policy.read_message(truncated)


def test_server_normalization_inverts_the_dataset_export() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(-90.0, 90.0, size=(64, 6)).astype(np.float32)
    normalized, minimum, maximum = normalize_min_max(raw)
    round_tripped = server.unnormalize_actions(normalized, minimum, maximum)
    np.testing.assert_allclose(round_tripped, raw, atol=1e-4)
    forward = server.normalize_state(raw, minimum, maximum)
    np.testing.assert_allclose(forward, normalized, atol=1e-6)


FAKE_SERVER = '''
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, {server_dir!r})
from dppo_policy_server import read_message, write_message

stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
write_message(stdout, {{
    "horizon_steps": np.asarray(4),
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
}})
queries = 0
while True:
    request = read_message(stdin)
    if request is None:
        break
    queries += 1
    actions = np.full((4, 6), float(queries), dtype=np.float32)
    actions += np.arange(4, dtype=np.float32)[:, None] / 10.0
    write_message(stdout, {{"actions": actions}})
'''


@pytest.fixture
def fake_server_command(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER.format(server_dir=str(SERVER_SCRIPT.parent)))
    return [sys.executable, str(script)]


def _observation() -> dict[str, np.ndarray]:
    return {
        STATE_FEATURE: np.zeros(6, dtype=np.float32),
        OVERHEAD_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
        WRIST_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_controller_serves_chunks_and_requeries(fake_server_command: list[str]) -> None:
    controller = DppoPolicyController(fake_server_command, act_steps=2)
    try:
        assert controller.horizon_steps == 4
        assert controller.image_hw == (8, 8)
        assert controller.handshake["epoch"] == 500
        # Two actions per query: the integer part encodes the query count and
        # the fractional part the position within the returned horizon.
        values = [controller.act(_observation())[0] for _ in range(4)]
        assert values == pytest.approx([1.0, 1.1, 2.0, 2.1])
    finally:
        controller.close()


def test_controller_reset_discards_queued_actions(fake_server_command: list[str]) -> None:
    controller = DppoPolicyController(fake_server_command)
    try:
        assert controller.act_steps == 4
        first = controller.act(_observation())[0]
        controller.reset()
        second = controller.act(_observation())[0]
        assert (first, second) == pytest.approx((1.0, 2.0))
    finally:
        controller.close()


def test_controller_rejects_invalid_act_steps(fake_server_command: list[str]) -> None:
    with pytest.raises(ValueError, match="act_steps"):
        DppoPolicyController(fake_server_command, act_steps=5)


def test_controller_reports_server_death(fake_server_command: list[str]) -> None:
    controller = DppoPolicyController(fake_server_command)
    try:
        controller._process.terminate()
        controller._process.wait()
        with pytest.raises(RuntimeError, match="server exited"):
            controller.act(_observation())
    finally:
        controller.close()


def test_controller_requires_all_observation_features(fake_server_command: list[str]) -> None:
    controller = DppoPolicyController(fake_server_command)
    try:
        observation = _observation()
        del observation[WRIST_FEATURE]
        with pytest.raises(KeyError):
            controller.act(observation)
    finally:
        controller.close()
