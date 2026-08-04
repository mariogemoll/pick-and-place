# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Frame-shaping operations shared by everything that produces policy images.

A render, a recorded video frame and a dataset sample all have to reach the
same pixel grid, or a policy trained on one cannot be fed the other. Doing that
in one place is what keeps them identical.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def resize_and_center_crop(image: NDArray, output_height: int, output_width: int) -> NDArray:
    """Area-downsample an image to cover the output, then center-crop it."""
    if output_width < 1 or output_height < 1:
        raise ValueError("output width and height must be positive")
    height, width = image.shape[:2]
    scale = max(output_width / width, output_height / height)
    resized_width = max(output_width, round(width * scale))
    resized_height = max(output_height, round(height * scale))
    if (resized_width, resized_height) != (width, height):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
    left = (resized_width - output_width) // 2
    top = (resized_height - output_height) // 2
    return np.asarray(image[top : top + output_height, left : left + output_width]).copy()
