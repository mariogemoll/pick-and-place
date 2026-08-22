# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Put a frame on the projector, through the kernel framebuffer.

The projector is an ordinary KMS output, so on a headless box the cheapest way
to light it is to write pixels into ``/dev/fb0``. That needs no X server, no
root, and no compositor -- membership of the ``video`` group is enough.

**A frame must already be the framebuffer's size.** Nothing here scales, and the
refusal is deliberate rather than an omission: what gets projected is a
calibration board whose feature positions *are* the measurement, so resampling
it behind the caller's back would corrupt the solve while looking like it
worked. A caller that wants a smaller pattern composes it into a full-size frame
and keeps the arithmetic where it can be checked.

Two limitations worth knowing before this is trusted unattended:

* ``fbcon`` owns the console on the same framebuffer and repaints over whatever
  is there whenever the console has something to say. Fine for an interactive
  calibration; not fine for a long unattended run.
* The mode is whatever ``fbcon`` set, which is the EDID's preferred mode rather
  than the panel's native size (see :mod:`pick_and_place.spec.projector`).

Both are the same fix -- hold DRM master with a dumb buffer instead -- and both
are reasons to treat this as the interactive path rather than the only one.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

SYS_GRAPHICS = Path("/sys/class/graphics")


@dataclass(frozen=True)
class FramebufferInfo:
    """Geometry of one ``/dev/fb*`` device, as the kernel reports it."""

    device: Path
    width: int
    height: int
    bits_per_pixel: int
    stride: int

    @property
    def size(self) -> tuple[int, int]:
        """Frame size as ``(width, height)``."""
        return (self.width, self.height)


def read_framebuffer_info(index: int = 0) -> FramebufferInfo:
    """Read framebuffer ``index``'s geometry from sysfs."""
    sysfs = SYS_GRAPHICS / f"fb{index}"
    if not sysfs.exists():
        raise FileNotFoundError(
            f"no framebuffer at {sysfs}. On this rig fb0 appears only once the "
            "projector is connected and powered on, since it is the sole display."
        )
    width, height = (int(value) for value in _read(sysfs, "virtual_size").split(","))
    bits_per_pixel = int(_read(sysfs, "bits_per_pixel"))
    if bits_per_pixel != 32:
        raise ValueError(f"expected a 32 bpp framebuffer, got {bits_per_pixel}")
    return FramebufferInfo(
        device=Path(f"/dev/fb{index}"),
        width=width,
        height=height,
        bits_per_pixel=bits_per_pixel,
        stride=int(_read(sysfs, "stride")),
    )


def _read(sysfs: Path, name: str) -> str:
    return (sysfs / name).read_text().strip()


def pack_xrgb8888(frame_bgr: NDArray[np.uint8], info: FramebufferInfo) -> bytes:
    """Pack a BGR frame into the framebuffer's XRGB8888 rows.

    XRGB8888 is little-endian, so the bytes land in memory as B, G, R, X --
    which is why the caller's frame is BGR and not RGB.
    """
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"frame must be uint8, got {frame_bgr.dtype}")
    if frame_bgr.shape != (info.height, info.width, 3):
        raise ValueError(
            f"frame is {frame_bgr.shape}, but {info.device} is "
            f"{(info.height, info.width, 3)}. Nothing here scales -- compose the "
            "pattern into a full-size frame instead."
        )

    rows = np.empty((info.height, info.width, 4), dtype=np.uint8)
    rows[..., :3] = frame_bgr
    rows[..., 3] = 255

    packed = rows.reshape(info.height, info.width * 4)
    padding = info.stride - info.width * 4
    if padding < 0:
        raise ValueError(f"stride {info.stride} is shorter than one {info.width}px row")
    if padding:
        packed = np.hstack([packed, np.zeros((info.height, padding), dtype=np.uint8)])
    return packed.tobytes()


def show(frame_bgr: NDArray[np.uint8], info: FramebufferInfo | None = None) -> FramebufferInfo:
    """Write one BGR frame to the framebuffer and return the geometry used."""
    info = info if info is not None else read_framebuffer_info()
    with open(info.device, "wb") as framebuffer:
        framebuffer.write(pack_xrgb8888(frame_bgr, info))
        framebuffer.flush()
    return info


def blank(info: FramebufferInfo | None = None) -> FramebufferInfo:
    """Project black."""
    info = info if info is not None else read_framebuffer_info()
    return show(np.zeros((info.height, info.width, 3), dtype=np.uint8), info)


@contextmanager
def projecting(frame_bgr: NDArray[np.uint8]) -> Iterator[FramebufferInfo]:
    """Project one frame for the duration of the block, then blank."""
    info = show(frame_bgr)
    try:
        yield info
    finally:
        blank(info)
