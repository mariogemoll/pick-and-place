# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Physical facts about the projector aimed at the workspace floor.

The unit is an AKIYO O1 pocket projector. Like the camera modules, these are
properties of the hardware rather than of any particular placement -- where it
stands is solved for, not declared.

**The panel is 1280x720 but the EDID advertises 1920x1080 as preferred.** A
1080p signal is therefore resampled by the projector's internal scaler before it
reaches the panel. That costs sharpness, and sharpness is feature-localization
accuracy when what is being projected is a calibration board, so the native size
is the one to drive when a mode can be chosen.

It does *not* invalidate a calibration solved at 1080p. The scaler is a fixed
linear map from input pixels to panel pixels, and a linear map composed with the
panel-to-floor homography is still a homography -- so a solve against *input*
pixel coordinates stays exact whichever mode is driven. Only the residuals get
worse. Anything nonlinear in the scaler (overscan, edge sharpening) would show
up there rather than silently biasing the fit.
"""

from __future__ import annotations

#: Native panel size in pixels. Drive this when the mode is ours to pick.
PROJECTOR_NATIVE_SIZE: tuple[int, int] = (1280, 720)

#: What the projector's EDID names as its preferred mode, which is not its panel.
PROJECTOR_PREFERRED_SIZE: tuple[int, int] = (1920, 1080)
