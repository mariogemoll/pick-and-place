# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Where the split between a frozen SmolVLA prefix and the rest of it falls."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pick_and_place.policies.smolvla_frozen_prefix import _frozen_prefix_length  # noqa: E402


def att_2d_masks(att_masks: torch.Tensor) -> torch.Tensor:
    """lerobot's own rule: attend where the cumulative mask is no larger."""
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    return cumsum[:, None, :] <= cumsum[:, :, None]


def prefix(frozen: int, attending: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    masks = torch.tensor([[0] * frozen + [1] * attending], dtype=torch.bool)
    return masks, att_2d_masks(masks)


def test_the_boundary_is_the_first_token_that_attends():
    # Images and language carry 0; the state token carries 1 and is the only
    # thing in the prefix that a gradient has to reach.
    masks, two_d = prefix(frozen=140)
    assert _frozen_prefix_length(masks, two_d) == 140


def test_a_prefix_that_is_all_frozen_is_refused():
    masks = torch.zeros(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="no state token"):
        _frozen_prefix_length(masks, att_2d_masks(masks))


def test_a_frozen_token_after_an_attending_one_is_refused():
    # Padding appended past the state token would land here, and splitting on
    # the first attending token would then leave a padded tail in the wrong half.
    masks = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="mixes frozen and attending"):
        _frozen_prefix_length(masks, att_2d_masks(masks))


def test_a_mask_that_lets_the_frozen_half_look_forward_is_refused():
    # The whole argument is that the frozen tokens cannot see `state_proj`. If
    # the mask handed to the model says otherwise, the split is not sound and
    # this has to fail rather than train something subtly different.
    masks, two_d = prefix(frozen=4)
    two_d = two_d.clone()
    two_d[:, 0, 4] = True
    with pytest.raises(ValueError, match="attend to later ones"):
        _frozen_prefix_length(masks, two_d)
