# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Worker-process vectorization of :class:`DppoTaskEnv`.

DPPO's training agent asks very little of a vectorized environment: seed it,
``reset_arg`` it with per-environment options, and ``step`` it with one action
chunk per environment. That is the whole interface implemented here, which is
why this does not go through the vendored ``make_async`` -- that builder only
knows robomimic, d4rl, D3IL and furniture, and reaching it would drag in the
pinned ``gym`` 0.22 stack for no benefit.

Each worker owns a MuJoCo model, an offscreen renderer and its own stride of the
training scene stream. Workers are spawned rather than forked: an OpenGL context
does not survive ``fork``, which is the same reason DPPO builds a dummy env in
its parent process.

Observations cross the process boundary as ``uint8`` camera tensors (55 KB per
environment step, against 442 KB had they been floats), and the whole action
chunk is executed inside the worker, so one round trip covers ``act_steps``
control ticks.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
from collections.abc import Sequence
from typing import Any

import numpy as np

from pick_and_place.dppo_rl.env import DppoTaskEnv, EnvConfig

# Keys crossing the process boundary. "privileged" is present only when the
# asymmetric critic is enabled; the actor never reads it.
_OBS_KEYS = ("state", "rgb")
_PRIVILEGED_KEY = "privileged"


def _worker(
    connection,
    config: EnvConfig,
    offset: int,
    stride: int,
    mujoco_gl: str | None,
) -> None:
    # Ask the kernel to kill this worker if the trainer dies. Daemon processes
    # are only reaped through multiprocessing's exit handler, which a SIGKILL on
    # the parent skips -- leaving 64 workers alive holding their CUDA contexts,
    # which is how ~25 GB of this GPU went missing during one debugging session.
    with contextlib.suppress(Exception):
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

    if mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
    # Native math libraries multiply across workers otherwise; each worker is
    # one MuJoCo context and wants one thread.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    env: DppoTaskEnv | None = None
    try:
        while True:
            command, payload = connection.recv()
            if command == "make":
                env = DppoTaskEnv(config, offset=offset, stride=stride)
                connection.send(("ok", None))
            elif command == "reset":
                connection.send(("ok", env.reset(payload)))
            elif command == "rewind":
                # Put the worker's scene stream back at its first scene. Only
                # evaluation uses this: a training run wants the endless stream,
                # but scoring several policies in one process has to show each of
                # them the same scenes or the comparison is not paired.
                env.scene_stream.rewind()
                connection.send(("ok", None))
            elif command == "step":
                connection.send(("ok", env.step(payload)))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise RuntimeError(f"unknown command {command!r}")
    except Exception as error:  # surface the worker's failure to the trainer
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        if env is not None:
            env.close()
        connection.close()


class DppoVectorEnv:
    """``n_envs`` pick-and-place episodes stepped in parallel worker processes."""

    def __init__(
        self,
        config: EnvConfig,
        n_envs: int,
        *,
        mujoco_gl: str | None = None,
        start_method: str = "spawn",
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        self.config = config
        self.n_envs = n_envs
        context = mp.get_context(start_method)
        self._connections = []
        self._processes = []
        for index in range(n_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, config, index, n_envs, mujoco_gl),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        self._closed = False
        self._broadcast("make", [None] * n_envs)

    # -- plumbing --------------------------------------------------------

    def _broadcast(self, command: str, payloads: Sequence[Any]) -> list[Any]:
        for connection, payload in zip(self._connections, payloads, strict=True):
            connection.send((command, payload))
        results = []
        for index, connection in enumerate(self._connections):
            status, value = connection.recv()
            if status == "error":
                self.close()
                raise RuntimeError(f"environment worker {index} failed: {value}")
            results.append(value)
        return results

    @staticmethod
    def _stack(observations: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        keys = list(_OBS_KEYS)
        if _PRIVILEGED_KEY in observations[0]:
            keys.append(_PRIVILEGED_KEY)
        return {
            key: np.stack([observation[key] for observation in observations])
            for key in keys
        }

    # -- the interface DPPO's training agent calls -----------------------

    def seed(self, seeds: Sequence[int]) -> None:
        """Accepted for interface compatibility; resets are seeded by scene index.

        Every scene this run visits is determined by the configured seed base and
        the worker's stride, which is what makes a run's reset sequence
        reproducible. Per-worker RNG seeding would only add a second, weaker
        source of the same thing.
        """
        if len(list(seeds)) != self.n_envs:
            raise ValueError("expected one seed per environment")

    def rewind(self) -> None:
        """Put every worker's scene stream back at its first scene.

        Scoring several policies in one process is only a paired comparison if
        each one is shown the same scenes; without this the stream keeps
        advancing and the second policy is measured on harder or easier ground
        than the first, which looks exactly like a difference in policy.
        """
        self._broadcast("rewind", [None] * self.n_envs)

    def reset_arg(
        self, options_list: Sequence[dict[str, Any]] | None = None
    ) -> dict[str, np.ndarray]:
        options_list = list(options_list or [{} for _ in range(self.n_envs)])
        return self._stack(self._broadcast("reset", options_list))

    def reset_one_arg(self, env_ind: int, options: dict[str, Any] | None = None):
        connection = self._connections[env_ind]
        connection.send(("reset", options or {}))
        status, value = connection.recv()
        if status == "error":
            self.close()
            raise RuntimeError(f"environment worker {env_ind} failed: {value}")
        return value

    def step(
        self, action_venv: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        actions = np.asarray(action_venv, dtype=np.float32)
        if actions.shape[0] != self.n_envs:
            raise ValueError(
                f"expected {self.n_envs} action chunks, got {actions.shape[0]}"
            )
        results = self._broadcast("step", list(actions))
        observations = self._stack([result[0] for result in results])
        rewards = np.array([result[1] for result in results], dtype=np.float64)
        terminated = np.array([result[2] for result in results], dtype=bool)
        truncated = np.array([result[3] for result in results], dtype=bool)
        infos = [result[4] for result in results]
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        for connection in self._connections:
            connection.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
