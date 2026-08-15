#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fail if a ``pick_and_place`` module imports across the layering.

The package is a fan, not a stack:

* ``spec`` holds the physical facts and the contracts, and imports nothing else
  in the package; ``core`` is pure computation over them and imports only
  ``spec``.
* ``planning``, ``perception``, ``sim``, ``hardware``, ``data`` and ``policies``
  are capability branches. Each owns one heavy dependency and **none may import
  another** — they meet only above.
* ``runtime``, ``plant``, ``variants``, ``calibration``, ``analysis`` and ``cli``
  are where capabilities converge. They may import anything, including each other; nothing
  below them may import them.

That last rule is what gives the convergence tier a definition instead of
making it a leftover bucket: a module that needs two capabilities belongs there
by construction. A module that reaches sideways for a fact or a contract is
telling you the fact belongs in ``spec``.

``dppo_rl`` and ``dsrl``, the two RL fine-tuning strands, sit above everything
and are exempt.

Run it from ``py/``:

    python scripts/check_package_layering.py          # violations only
    python scripts/check_package_layering.py --all    # every cross-package edge
"""

from __future__ import annotations

import ast
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "pick_and_place"

FOUNDATION = ("spec", "core")
BRANCHES = ("planning", "perception", "sim", "hardware", "data", "policies")
CONVERGENCE = ("runtime", "plant", "variants", "calibration", "analysis", "cli")
EXEMPT = ("dppo_rl", "dsrl")


def is_violation(source: str, target: str) -> bool:
    """Whether importing ``target``'s package from ``source``'s breaks the layering."""
    if source == "spec":
        return True
    if source == "core":
        return target != "spec"
    if source in BRANCHES and target in BRANCHES:
        return True
    if source in CONVERGENCE:
        return False
    return target in CONVERGENCE


LAYERS = FOUNDATION + BRANCHES + CONVERGENCE


def package_of(module: str) -> str | None:
    """The layer a dotted module name belongs to, or ``None`` if it has none."""
    head = module.split(".")[0]
    return head if head in LAYERS else None


def internal_imports(path: pathlib.Path) -> set[str]:
    """Module names inside ``pick_and_place`` that ``path`` imports."""
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pick_and_place."):
                    targets.add(alias.name.removeprefix("pick_and_place."))
            continue
        else:
            continue
        if module.startswith("pick_and_place."):
            targets.add(module.removeprefix("pick_and_place."))
        elif module == "pick_and_place":
            for alias in node.names:
                targets.add(alias.name)
    return targets


def edges() -> dict[tuple[str, str], list[str]]:
    """Cross-package edges, each mapped to the module imports that create it."""
    found: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    unfiled: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        name = str(relative)[:-3].replace("/", ".").removesuffix(".__init__")
        source = relative.parts[0] if len(relative.parts) > 1 else None
        if source in EXEMPT or name == "__init__":
            continue
        if source not in LAYERS:
            unfiled.append(name)
            continue
        for target_module in internal_imports(path):
            target = package_of(target_module)
            if target is None:
                if target_module.split(".")[0] not in EXEMPT:
                    unfiled.append(target_module)
                continue
            if target != source:
                found[(source, target)].append(f"{name} -> {target_module}")
    if unfiled:
        print(
            "not part of any layer, so nothing constrains it: " + ", ".join(sorted(set(unfiled))),
            file=sys.stderr,
        )
        raise SystemExit(2)
    return found


def main() -> int:
    show_all = "--all" in sys.argv
    found = edges()
    shown = {edge: imports for edge, imports in found.items() if show_all or is_violation(*edge)}
    if not shown:
        print("no violations")
        return 0
    for (source, target), imports in sorted(shown.items()):
        mark = "" if show_all and not is_violation(source, target) else "  [violation]"
        print(f"{source} -> {target}{mark}")
        for line in sorted(set(imports)):
            print(f"    {line}")
    return 1 if any(is_violation(*edge) for edge in shown) else 0


if __name__ == "__main__":
    raise SystemExit(main())
