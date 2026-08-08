# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Memory-map the arrays inside an uncompressed NPZ.

``np.load`` on an NPZ always materializes an array in RAM, which is not an
option for an image tensor of several gigabytes when only a few episodes are
wanted. The archives written by the Diffusion Policy exporter store their
members uncompressed and unencrypted, so each array is a contiguous run of
bytes at a fixed offset in the file and can be mapped directly.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")


def _payload_offset(file, entry: zipfile.ZipInfo) -> int:
    """Return the absolute offset of ``entry``'s first payload byte."""
    file.seek(entry.header_offset)
    header = _LOCAL_HEADER.unpack(file.read(_LOCAL_HEADER.size))
    if header[0] != b"PK\x03\x04":
        raise ValueError(f"{entry.filename} has no local file header")
    name_length, extra_length = header[-2:]
    return entry.header_offset + _LOCAL_HEADER.size + name_length + extra_length


def memmap_stored_npz(path: Path) -> dict[str, np.memmap]:
    """Map every array of an uncompressed NPZ without reading its contents."""
    arrays: dict[str, np.memmap] = {}
    with zipfile.ZipFile(path) as archive, path.open("rb") as file:
        for entry in archive.infolist():
            if entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{entry.filename} is compressed; cannot memory-map it")
            offset = _payload_offset(file, entry)
            file.seek(offset)
            major, _ = np.lib.format.read_magic(file)
            read_header = (
                np.lib.format.read_array_header_1_0
                if major == 1
                else np.lib.format.read_array_header_2_0
            )
            shape, fortran_order, dtype = read_header(file)
            if fortran_order:
                raise ValueError(f"{entry.filename} is Fortran-ordered")
            name = entry.filename.removesuffix(".npy")
            arrays[name] = np.memmap(path, dtype=dtype, mode="r", offset=file.tell(), shape=shape)
    return arrays


def episode_bounds(traj_lengths: np.ndarray) -> np.ndarray:
    """Return the ``(start, stop)`` frame range of every episode."""
    stops = np.cumsum(np.asarray(traj_lengths, dtype=np.int64))
    starts = stops - traj_lengths
    return np.stack([starts, stops], axis=1)
