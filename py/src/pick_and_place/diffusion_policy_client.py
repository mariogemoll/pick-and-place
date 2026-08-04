# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Client for the out-of-process Diffusion Policy server.

The DPPO stack the policy is built on lives in its own virtual environment
(incompatible Torch/Gym/AV pins), so inference runs in a subprocess speaking a
small binary protocol on its standard streams: every message is a 4-byte
big-endian length prefix followed by an ``.npz`` payload.
``scripts/diffusion_policy_server.py`` implements the other end and documents
the message contract.

:class:`DiffusionPolicyController` adapts that server to the evaluator's
``PolicyController`` protocol. The policy predicts an action horizon per query;
the controller queues the first ``act_steps`` actions and serves one per control
tick, re-querying when the queue empties. Observation history is retained on
every control tick, including while queued actions are being executed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
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

SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diffusion_policy_server.py"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "diffusion_policy"
    / "pretrain_so101_unet_img.yaml"
)


def add_diffusion_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common Diffusion Policy server arguments to a runner CLI."""
    parser.add_argument(
        "--diffusion-policy-python",
        type=Path,
        default=os.environ.get("DIFFUSION_POLICY_PYTHON"),
        help="interpreter of the DPPO virtual environment (default: $DIFFUSION_POLICY_PYTHON)",
    )
    parser.add_argument(
        "--diffusion-policy-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="training configuration YAML (default: the pretraining configuration)",
    )
    parser.add_argument(
        "--diffusion-policy-normalization",
        type=Path,
        help="normalization.npz written by the Diffusion Policy dataset export",
    )
    parser.add_argument(
        "--recording-hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "resolution the training videos were recorded at; defaults to "
            "source_video_hw in export.json beside --diffusion-policy-normalization"
        ),
    )
    parser.add_argument(
        "--diffusion-policy-act-steps",
        type=int,
        default=None,
        help="executed actions per policy query (default: the training configuration)",
    )
    parser.add_argument(
        "--diffusion-policy-seed",
        type=int,
        default=0,
        help="Torch seed for DDPM action sampling (default: 0)",
    )
    parser.add_argument(
        "--diffusion-policy-ddim-steps",
        type=int,
        default=None,
        help=(
            "sample with DDIM using this many steps instead of the trained DDPM "
            "schedule (try 10 for live control)"
        ),
    )


def resolve_recording_hw(
    normalization: str | Path,
    override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve the intermediate video size used by a Diffusion Policy dataset export."""
    if override is not None:
        height, width = override
        if height < 1 or width < 1:
            raise ValueError("recording height and width must be positive")
        return (height, width)

    export_path = Path(normalization).parent / "export.json"
    if not export_path.exists():
        raise FileNotFoundError(
            f"no export.json beside {normalization}; pass the recording height and width"
        )
    with export_path.open() as file:
        export = json.load(file)
    if "source_video_hw" not in export:
        raise ValueError(
            f"{export_path} predates source_video_hw; pass the resolution its source "
            "dataset's videos were recorded at"
        )
    height, width = export["source_video_hw"]
    return (int(height), int(width))


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


class DiffusionPolicyController:
    """Run a Diffusion Policy checkpoint behind a subprocess boundary."""

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
        self.handshake: dict[str, Any] = {key: value.item() for key, value in handshake.items()}
        self.horizon_steps = int(self.handshake["horizon_steps"])
        self.cond_steps = int(self.handshake["cond_steps"])
        self.policy_hz = float(self.handshake["policy_hz"])
        if self.policy_hz <= 0:
            self.close()
            raise ValueError(f"policy_hz must be positive, got {self.policy_hz}")
        self.obs_dim = int(self.handshake["obs_dim"])
        self.action_dim = int(self.handshake["action_dim"])
        self.image_hw = (
            int(self.handshake["image_height"]),
            int(self.handshake["image_width"]),
        )
        if act_steps is None:
            act_steps = int(self.handshake["act_steps"])
        if not 1 <= act_steps <= self.horizon_steps:
            self.close()
            raise ValueError(f"act_steps must be in [1, {self.horizon_steps}], got {act_steps}")
        self.act_steps = act_steps
        self._queue: deque[np.ndarray] = deque()
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self.cond_steps)
        self.latest_prediction: np.ndarray | None = None

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
    ) -> DiffusionPolicyController:
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

    @classmethod
    def launch_from_args(
        cls,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *,
        override_hw: tuple[int, int] | None,
        default_checkpoint: str,
    ) -> tuple[DiffusionPolicyController, tuple[int, int]]:
        """Validate common runner arguments and start a configured controller."""
        if args.checkpoint == default_checkpoint:
            parser.error(
                "--checkpoint (a state_*.pt file) is required for the "
                "diffusion-policy controller"
            )
        if args.diffusion_policy_python is None:
            parser.error(
                "--diffusion-policy-python (or $DIFFUSION_POLICY_PYTHON) is required "
                "for the diffusion-policy controller"
            )
        if args.diffusion_policy_normalization is None:
            parser.error(
                "--diffusion-policy-normalization is required for the "
                "diffusion-policy controller"
            )
        try:
            recording_hw = resolve_recording_hw(args.diffusion_policy_normalization, args.recording_hw)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

        print(f"Starting the Diffusion Policy server for {args.checkpoint}...")
        controller = cls.launch(
            python=args.diffusion_policy_python,
            checkpoint=args.checkpoint,
            config=args.diffusion_policy_config,
            normalization=args.diffusion_policy_normalization,
            device=args.device,
            seed=args.diffusion_policy_seed,
            act_steps=args.diffusion_policy_act_steps,
            ddim_steps=args.diffusion_policy_ddim_steps,
        )
        if override_hw is not None and override_hw != controller.image_hw:
            trained_hw = controller.image_hw
            controller.close()
            parser.error(
                f"--image-height/--image-width {override_hw} do not match the "
                f"model's trained image size {trained_hw}"
            )
        return controller, recording_hw

    def _server_exited(self) -> RuntimeError:
        code = self._process.poll()
        return RuntimeError(
            f"diffusion policy server exited unexpectedly (return code {code}); "
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
        self._history.clear()
        self.latest_prediction = None

    def _current_observation(self, observation: PolicyObservation) -> dict[str, np.ndarray]:
        for feature in (STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE):
            if feature not in observation:
                raise KeyError(f"observation is missing {feature!r}")
        return {
            "state": np.asarray(observation[STATE_FEATURE], dtype=np.float32).copy(),
            "overhead": np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8).copy(),
            "wrist": np.asarray(observation[WRIST_FEATURE], dtype=np.uint8).copy(),
        }

    def _request_actions(
        self,
        history: list[dict[str, np.ndarray]],
        *,
        sampling_seed: int | None = None,
    ) -> np.ndarray:
        assert self._process.stdin is not None
        request = {
            key: np.stack([item[key] for item in history]) for key in ("state", "overhead", "wrist")
        }
        if sampling_seed is not None:
            request["sampling_seed"] = np.asarray(sampling_seed, dtype=np.int64)
        try:
            write_message(self._process.stdin, request)
        except BrokenPipeError:
            raise self._server_exited() from None
        return np.asarray(self._receive()["actions"], dtype=np.float32)

    def predict_horizon(
        self,
        observation: PolicyObservation,
        *,
        sampling_seed: int | None = None,
    ) -> np.ndarray:
        """Predict a full action horizon from repeated copies of one observation.

        ``sampling_seed`` makes diffusion sampling repeatable for counterfactual
        diagnostics. It is intentionally separate from :meth:`act`, whose
        history and queued-action behavior model closed-loop execution.
        """
        current = self._current_observation(observation)
        actions = self._request_actions([current] * self.cond_steps, sampling_seed=sampling_seed)
        if actions.shape != (self.horizon_steps, self.action_dim):
            raise ValueError(f"server returned malformed actions with shape {actions.shape}")
        return actions.copy()

    def predict_horizon_from_history(
        self,
        observations: list[PolicyObservation],
        *,
        sampling_seed: int | None = None,
    ) -> np.ndarray:
        """Predict from an explicit observation history for diagnostics."""
        if len(observations) != self.cond_steps:
            raise ValueError(f"expected {self.cond_steps} observations, got {len(observations)}")
        history = [self._current_observation(observation) for observation in observations]
        actions = self._request_actions(history, sampling_seed=sampling_seed)
        if actions.shape != (self.horizon_steps, self.action_dim):
            raise ValueError(f"server returned malformed actions with shape {actions.shape}")
        return actions.copy()

    def act(self, observation: PolicyObservation) -> np.ndarray:
        self.latest_prediction = None
        current = self._current_observation(observation)
        self._history.append(current)
        if not self._queue:
            history = [self._history[0]] * (self.cond_steps - len(self._history))
            history.extend(self._history)
            actions = self._request_actions(history)
            if actions.ndim != 2 or actions.shape[0] < self.act_steps:
                raise ValueError(f"server returned malformed actions with shape {actions.shape}")
            self.latest_prediction = actions.copy()
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
