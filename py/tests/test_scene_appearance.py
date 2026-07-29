# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import pytest

from pick_and_place.scene_appearance import AS_RECORDED, SceneAppearance, parse_appearance


def test_named_presets_keep_their_name():
    assert parse_appearance("blue-cube") == ("blue-cube", SceneAppearance(cube="blue"))
    assert parse_appearance("as-recorded") == ("as-recorded", AS_RECORDED)
    assert parse_appearance("black-floor") == (
        "black-floor",
        SceneAppearance(floor="black", target="white"),
    )


def test_ad_hoc_specs_are_named_after_their_fields():
    name, appearance = parse_appearance("cube=blue,floor=dark-gray")
    assert appearance == SceneAppearance(floor="dark-gray", cube="blue")
    assert name == "cube-blue_floor-dark-gray"

    _, tagless = parse_appearance("frame_tags=off")
    assert tagless == SceneAppearance(frame_tags=False)


@pytest.mark.parametrize(
    "token",
    ["chartreuse-cube", "cube=chartreuse", "wall=blue"],
)
def test_unknown_names_fields_and_colours_are_rejected(token):
    with pytest.raises(ValueError):
        parse_appearance(token)
