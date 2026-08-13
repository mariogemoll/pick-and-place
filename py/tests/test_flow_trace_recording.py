# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The flow trace binary format round-trips and rejects inconsistent inputs."""

from __future__ import annotations

import numpy as np
import pytest

from pick_and_place.analysis.flow_trace_recording import (
    FlowTraceRecording,
    decode,
    encode,
)

FRAMES = 24
NQ = 13
CHUNKS = 3
EULER = 10
STEPS = 16
JOINTS = 6


def build(**overrides) -> FlowTraceRecording:
    rng = np.random.default_rng(0)
    fields = {
        "fps": 10.0,
        "act_steps": 8,
        "target_xy": (0.21, -0.05),
        "qpos": rng.standard_normal((FRAMES, NQ)).astype(np.float32),
        "chunk_ticks": np.array([0, 8, 16], dtype=np.uint32),
        "path": rng.standard_normal((CHUNKS, EULER + 1, STEPS, JOINTS)).astype(np.float32),
        "commands": rng.standard_normal((CHUNKS, STEPS, JOINTS)).astype(np.float32),
    }
    return FlowTraceRecording(**(fields | overrides))


def test_round_trip_preserves_every_field() -> None:
    original = build()
    restored = decode(encode(original))

    assert restored.fps == original.fps
    assert restored.act_steps == original.act_steps
    assert restored.target_xy == pytest.approx(original.target_xy)
    np.testing.assert_array_equal(restored.qpos, original.qpos)
    np.testing.assert_array_equal(restored.chunk_ticks, original.chunk_ticks)
    np.testing.assert_array_equal(restored.path, original.path)
    np.testing.assert_array_equal(restored.commands, original.commands)


def test_the_payload_is_header_plus_exactly_its_arrays() -> None:
    recording = build()
    arrays = FRAMES * NQ + CHUNKS * (EULER + 1) * STEPS * JOINTS + CHUNKS * STEPS * JOINTS

    assert len(encode(recording)) == 48 + 4 * arrays + 4 * CHUNKS


def test_a_realistic_rollout_stays_small() -> None:
    """The whole point of leaving images out: one episode fits in ~100 KB."""
    rng = np.random.default_rng(0)
    recording = build(
        qpos=rng.standard_normal((150, NQ)).astype(np.float32),
        chunk_ticks=np.arange(0, 150, 8, dtype=np.uint32),
        path=rng.standard_normal((19, EULER + 1, STEPS, JOINTS)).astype(np.float32),
        commands=rng.standard_normal((19, STEPS, JOINTS)).astype(np.float32),
    )

    assert len(encode(recording)) < 110_000


def test_the_magic_header_is_checked() -> None:
    payload = bytearray(encode(build()))
    payload[:4] = b"XXXX"

    with pytest.raises(ValueError, match="magic"):
        decode(bytes(payload))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"commands": np.zeros((CHUNKS, STEPS, JOINTS + 1), dtype=np.float32)}, "commands"),
        ({"chunk_ticks": np.array([0, 8], dtype=np.uint32)}, "one entry per chunk"),
        ({"act_steps": STEPS + 1}, "act_steps"),
        ({"chunk_ticks": np.array([0, 8, FRAMES], dtype=np.uint32)}, "recorded frame"),
    ],
)
def test_inconsistent_recordings_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build(**overrides)
