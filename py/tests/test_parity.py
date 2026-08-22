# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The Python half of the cross-language parity check.

``fixtures/parity/`` is the oracle both implementations answer to. Here we
confirm Python still produces it; the tests in ``ts/src/parity/`` confirm
TypeScript reproduces it too. A change that moves the planner fails this test
first, and regenerating the fixtures to make it pass is exactly the moment to
notice that the TypeScript side now has to follow.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pick_and_place.cli import generate_parity_fixtures as generate

#: Absolute tolerance on every number in a fixture. The fixtures are written at
#: twelve significant digits; this leaves room for the last-place wobble a
#: different CPU or BLAS can introduce without letting a real change through.
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, dict[str, Any]]:
    """The fixtures as the current code would write them."""
    return generate.build_fixtures(generate.kinematics())


def _assert_matches(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, bool) or expected is None:
        assert actual == expected, path
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)), path
        assert abs(float(actual) - float(expected)) <= TOLERANCE, (
            f"{path}: {actual} != {expected}"
        )
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, (a, e) in enumerate(zip(actual, expected)):
            _assert_matches(a, e, f"{path}[{index}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_matches(actual[key], expected[key], f"{path}.{key}")
    else:
        assert actual == expected, path


COMMITTED = sorted(path.name for path in generate.FIXTURE_DIR.glob("*.json"))


@pytest.mark.parametrize("name", COMMITTED)
def test_committed_fixture_still_reproduces(name: str, rebuilt: dict[str, Any]) -> None:
    path = generate.FIXTURE_DIR / name
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert name in rebuilt, f"{name} is committed but no longer generated"
    _assert_matches(rebuilt[name], committed, name)


def test_every_generated_fixture_is_committed(rebuilt: dict[str, Any]) -> None:
    for name in rebuilt:
        assert (generate.FIXTURE_DIR / name).is_file(), f"{name} was never committed"


def test_fixtures_cover_the_unreachable_answers() -> None:
    """Parity on what *fails* matters as much as parity on what succeeds.

    A port that solves every reachable pose but quietly returns a wrong branch
    for an unreachable one is the failure mode these fixtures exist to catch, so
    refuse to let the unreachable cases be sampled away.
    """
    ik = json.loads((generate.FIXTURE_DIR / "simple_ik.json").read_text(encoding="utf-8"))
    unreachable = [case for case in ik["cases"] if not case["branches"]]
    assert unreachable, "no unreachable IK case in the fixtures"
    assert any("out of plane" in case["label"] for case in unreachable)

    grasps = json.loads((generate.FIXTURE_DIR / "grasp.json").read_text(encoding="utf-8"))
    assert any(case["selected"] is None for case in grasps["cases"])
