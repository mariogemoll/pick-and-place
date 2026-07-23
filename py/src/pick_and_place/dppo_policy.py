# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Client for the out-of-process DPPO diffusion-policy server.

The DPPO stack lives in its own virtual environment (incompatible Torch/Gym/AV
pins), so inference runs in a subprocess speaking a small binary protocol on
its standard streams: every message is a 4-byte big-endian length prefix
followed by an ``.npz`` payload. ``scripts/dppo_policy_server.py`` implements
the other end and documents the message contract.

:class:`DppoPolicyController` adapts that server to the evaluator's
``PolicyController`` protocol. The policy predicts a short action horizon per
query; the controller queues the first ``act_steps`` actions and serves one per
control tick, re-querying when the queue empties.
"""

from __future__ import annotations

import io
import struct
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    PolicyObservation,
)

SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dppo_policy_server.py"


def write_message(stream: BinaryIO, arrays: dict[str, np.ndarray]) -> None:
    """Write one length-prefixed npz message."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def read_message(stream: BinaryIO) -> dict[str, np.ndarray] | None:
    """Read one length-prefixed npz message, or None on a clean end-of-file."""
    header = stream.read(4)
    if len(header) == 0:
        return None
    if len(header) != 4:
        raise EOFError("truncated message header")
    (length,) = struct.unpack(">I", header)
    payload = stream.read(length)
    if len(payload) != length:
        raise EOFError(f"truncated message payload: expected {length} bytes, got {len(payload)}")
    with np.load(io.BytesIO(payload)) as data:
        return {key: data[key] for key in data.files}


class DppoPolicyController:
    """Run a DPPO diffusion-policy checkpoint behind a subprocess boundary."""

    def __init__(self, command: list[str], *, act_steps: int | None = None) -> None:
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        try:
            handshake = self._receive()
        except Exception:
            self.close()
            raise
        self.handshake: dict[str, Any] = {
            key: value.item() for key, value in handshake.items()
        }
        self.horizon_steps = int(self.handshake["horizon_steps"])
        self.obs_dim = int(self.handshake["obs_dim"])
        self.action_dim = int(self.handshake["action_dim"])
        self.image_hw = (
            int(self.handshake["image_height"]),
            int(self.handshake["image_width"]),
        )
        if act_steps is None:
            act_steps = self.horizon_steps
        if not 1 <= act_steps <= self.horizon_steps:
            self.close()
            raise ValueError(
                f"act_steps must be in [1, {self.horizon_steps}], got {act_steps}"
            )
        self.act_steps = act_steps
        self._queue: deque[np.ndarray] = deque()

    @classmethod
    def launch(
        cls,
        *,
        python: str | Path,
        checkpoint: str | Path,
        config: str | Path,
        normalization: str | Path,
        device: str = "auto",
        seed: int = 0,
        act_steps: int | None = None,
        ddim_steps: int | None = None,
        server_script: str | Path = SERVER_SCRIPT,
    ) -> "DppoPolicyController":
        """Start the server with the DPPO environment's interpreter."""
        command = [
            str(python),
            str(server_script),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--normalization",
            str(normalization),
            "--device",
            device,
            "--seed",
            str(seed),
        ]
        if ddim_steps is not None:
            command += ["--ddim-steps", str(ddim_steps)]
        return cls(command, act_steps=act_steps)

    def _server_exited(self) -> RuntimeError:
        code = self._process.poll()
        return RuntimeError(
            f"dppo policy server exited unexpectedly (return code {code}); "
            "its stderr output has the details"
        )

    def _receive(self) -> dict[str, np.ndarray]:
        assert self._process.stdout is not None
        message = read_message(self._process.stdout)
        if message is None:
            raise self._server_exited()
        return message

    def reset(self) -> None:
        self._queue.clear()

    def act(self, observation: PolicyObservation) -> np.ndarray:
        for feature in (STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE):
            if feature not in observation:
                raise KeyError(f"observation is missing {feature!r}")
        if not self._queue:
            assert self._process.stdin is not None
            try:
                write_message(
                    self._process.stdin,
                    {
                        "state": np.asarray(observation[STATE_FEATURE], dtype=np.float32),
                        "overhead": np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8),
                        "wrist": np.asarray(observation[WRIST_FEATURE], dtype=np.uint8),
                    },
                )
            except BrokenPipeError:
                raise self._server_exited() from None
            reply = self._receive()
            actions = np.asarray(reply["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] < self.act_steps:
                raise ValueError(f"server returned malformed actions with shape {actions.shape}")
            self._queue.extend(actions[: self.act_steps])
        return self._queue.popleft().copy()

    def close(self) -> None:
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()
