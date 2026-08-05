# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What one recording run is: the scene it draws, its sizes, and where it lands.

These travel together and are decided once per run, so the recorder takes three
records rather than twenty loose arguments. They are also what a worker process
is handed, so they are plain frozen data — picklable, comparable, and printable
into a run's own configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Size the saved dataset frames are stored at, downsampled and cropped from the
# offscreen render. Smaller than the render so the reduction is a real one.
SAVED_IMAGE_WIDTH = 960
SAVED_IMAGE_HEIGHT = 720

# Offscreen render size the MuJoCo cameras are rendered at before that
# reduction. Also the default the live runners and the evaluation harness use.
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080


@dataclass(frozen=True)
class SceneDraw:
    """What each episode's scene is drawn from.

    Everything here is either pinned or left to be sampled per episode. Pinning
    the cube or the target makes a run reproducible in one place; the textures
    and the randomization preset decide how much the look varies across a
    dataset, which is what a policy has to generalize over.
    """

    source_xy: tuple[float, float] | None = None
    target_xy: tuple[float, float] | None = None
    background_panorama: Path | None = None
    table_texture: Path | None = None
    miscalibration: bool = False
    domain_randomization: Path | None = None


@dataclass(frozen=True)
class FrameSizes:
    """The offscreen render size and the size frames are saved at.

    Rendering above the saved size and reducing afterwards is what keeps the
    saved frames free of aliasing, so the render must never be the smaller of
    the two.
    """

    render_width: int = RENDER_WIDTH
    render_height: int = RENDER_HEIGHT
    image_width: int = SAVED_IMAGE_WIDTH
    image_height: int = SAVED_IMAGE_HEIGHT

    def __post_init__(self) -> None:
        if min(self.image_width, self.image_height) < 1:
            raise ValueError("image dimensions must be positive")
        if self.render_width < self.image_width or self.render_height < self.image_height:
            raise ValueError("the render size must be at least the saved image size")


@dataclass(frozen=True)
class DatasetOutput:
    """Where a recorded dataset lands and how its video is encoded.

    One encoding profile per dataset: the codec is pinned rather than probed, so
    every episode in a dataset is encoded the same way and a top-up run months
    later still matches what is already there.
    """

    root: Path
    repo_id: str
    task: str
    vcodec: str = "h264"
    streaming_encoding: bool = True
    image_writer_threads: int = 4
