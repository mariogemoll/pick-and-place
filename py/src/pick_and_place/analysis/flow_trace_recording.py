# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One flow-policy rollout, serialized for the web viewer.

The viewer replays the arm from ``qpos`` the way the episode-replay
visualization already does, and draws the integration path beside it: for each
generated horizon, the Gaussian noise the sampler started from, one state per
Euler step, and the sample at ``t = 1``.

Integration happens in the normalized action space the model was trained in, so
``path`` is kept in that space -- the states before the last are not joint
angles yet and converting them would misrepresent what the network does. Only
``commands`` is in degrees, clipped and unnormalized the way the arm receives
it.

Binary layout ("PPFT" format, little-endian)::

    magic       4 bytes   b"PPFT"
    version     u32       1
    fps         u32       policy control rate
    nframes     u32       policy ticks in the rollout
    nq          u32       floats per replay frame (6 joints + 7 cube pose = 13)
    njoints     u32       action dimensions per predicted step
    nsteps      u32       predicted steps per horizon
    nact        u32       steps of each horizon actually executed
    neuler      u32       Euler steps per horizon
    nchunks     u32       horizons generated
    target_x    f32       drop target position on the floor (meters)
    target_y    f32
    qpos        f32[nframes * nq]
    chunk_tick  u32[nchunks]
    path        f32[nchunks * (neuler + 1) * nsteps * njoints]
    commands    f32[nchunks * nsteps * njoints]

At the usual operating point -- 10 Hz, 150 ticks, 16 predicted steps, 8
executed, 10 Euler steps -- that is about 95 KB, nearly all of it ``path``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"PPFT"
VERSION = 1
HEADER_FORMAT = "<4sIIIIIIIII2f"


@dataclass(frozen=True)
class FlowTraceRecording:
    """A rollout's replay state and the integration path behind every horizon."""

    fps: float
    act_steps: int
    target_xy: tuple[float, float]
    # (nframes, nq)
    qpos: np.ndarray
    # (nchunks,) the tick each horizon was generated on
    chunk_ticks: np.ndarray
    # (nchunks, neuler + 1, nsteps, njoints) in normalized action space
    path: np.ndarray
    # (nchunks, nsteps, njoints) in degrees
    commands: np.ndarray

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2 or not len(self.qpos):
            raise ValueError("qpos must be a nonempty (frames, nq) array")
        if self.path.ndim != 4 or self.path.shape[1] < 2:
            raise ValueError("path must be (chunks, euler + 1, steps, joints)")
        chunks, _, steps, joints = self.path.shape
        if self.commands.shape != (chunks, steps, joints):
            raise ValueError("commands must match the path's chunks, steps and joints")
        if self.chunk_ticks.shape != (chunks,):
            raise ValueError("chunk_ticks must have one entry per chunk")
        if not 1 <= self.act_steps <= steps:
            raise ValueError("act_steps must be between 1 and the predicted steps")
        if self.chunk_ticks.size and self.chunk_ticks.max() >= len(self.qpos):
            raise ValueError("chunk_ticks must index a recorded frame")


def encode(recording: FlowTraceRecording) -> bytes:
    """Serialize a recording to the ``PPFT`` binary layout."""
    chunks, states, steps, joints = recording.path.shape
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        round(recording.fps),
        len(recording.qpos),
        recording.qpos.shape[1],
        joints,
        steps,
        recording.act_steps,
        states - 1,
        chunks,
        *recording.target_xy,
    )
    return b"".join(
        (
            header,
            np.ascontiguousarray(recording.qpos, dtype="<f4").tobytes(),
            np.ascontiguousarray(recording.chunk_ticks, dtype="<u4").tobytes(),
            np.ascontiguousarray(recording.path, dtype="<f4").tobytes(),
            np.ascontiguousarray(recording.commands, dtype="<f4").tobytes(),
        )
    )


def decode(payload: bytes) -> FlowTraceRecording:
    """Read back what :func:`encode` wrote."""
    size = struct.calcsize(HEADER_FORMAT)
    (
        magic,
        version,
        fps,
        nframes,
        nq,
        njoints,
        nsteps,
        nact,
        neuler,
        nchunks,
        target_x,
        target_y,
    ) = struct.unpack(HEADER_FORMAT, payload[:size])
    if magic != MAGIC:
        raise ValueError(f"unexpected magic header: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported flow trace version: {version}")

    offset = size

    def take(dtype: str, count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        width = np.dtype(dtype).itemsize * count
        values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        offset += width
        return values.reshape(shape)

    return FlowTraceRecording(
        fps=float(fps),
        act_steps=nact,
        target_xy=(target_x, target_y),
        qpos=take("<f4", nframes * nq, (nframes, nq)),
        chunk_ticks=take("<u4", nchunks, (nchunks,)),
        path=take(
            "<f4",
            nchunks * (neuler + 1) * nsteps * njoints,
            (nchunks, neuler + 1, nsteps, njoints),
        ),
        commands=take("<f4", nchunks * nsteps * njoints, (nchunks, nsteps, njoints)),
    )
