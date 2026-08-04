# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The rule the package layout is held to, and that the layout satisfies it."""

import importlib.util
from pathlib import Path

CHECKER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_package_layering.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_package_layering", CHECKER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_the_foundation_may_not_reach_up() -> None:
    assert checker.is_violation("core", "sim")
    assert checker.is_violation("spec", "core")
    assert not checker.is_violation("core", "spec")


def test_capability_branches_may_not_import_each_other() -> None:
    assert checker.is_violation("sim", "perception")
    assert checker.is_violation("planning", "policies")
    assert not checker.is_violation("sim", "core")


def test_only_the_convergence_tier_may_import_the_convergence_tier() -> None:
    assert checker.is_violation("sim", "runtime")
    assert not checker.is_violation("runtime", "sim")
    assert not checker.is_violation("analysis", "runtime")


def test_the_package_has_no_violating_edges() -> None:
    violations = {edge for edge in checker.edges() if checker.is_violation(*edge)}

    assert violations == set()
