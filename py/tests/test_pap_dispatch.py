# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The command table is data, so something has to check it against the tree.

``pick_and_place/cli/commands.py`` names modules and script paths as strings,
which is what lets ``pap --help`` list every command without importing any of
them. The cost is that a wrong string is invisible until someone runs that one
command. These tests pay it back: every entry resolves, every entry's parser
builds, and every script is either registered or on the list of things that are
deliberately not commands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pick_and_place.cli.commands import COMMANDS, COMMANDS_BY_NAME
from pick_and_place.cli.dispatch import SCRIPTS_DIR, load_parser_owner
from pick_and_place.cli.suggest import SuggestingArgumentParser

#: Files under ``scripts/`` that are not commands, and why.
NOT_COMMANDS = {
    "pap.py": "the dispatcher itself",
    "check_package_layering.py": "a CI check, run by name from the build",
}

#: Scripts that still have to be given a ``build_parser()``/``run()`` seam and a
#: table entry. **This list only shrinks.** A script leaves it in the same commit
#: that registers it, so the countdown is visible in the history and the build is
#: green at every step of it. When it is empty, delete it and the test that reads
#: it -- at that point every script is either a command or on ``NOT_COMMANDS``.
PENDING = {
    "calibrate_camera_intrinsics.py",
    "calibrate_joint_zeros.py",
    "diagnose_flow_image_policy.py",
    "eval_scripted_parallel.py",
    "export_camera_calibrations.py",
    "pick_and_place/finalize_sim_dataset.py",
    "pick_and_place/real.py",
    "pick_and_place/record_sim.py",
    "run_flow_image_policy_sim.py",
    "run_policy_real.py",
    "run_policy_sim.py",
    "showcamfeed.py",
    "showcamfeeds.py",
    "train_flow_image_policy.py",
    "wrist_cam_align_solve.py",
}


def _scripts() -> list[Path]:
    return sorted(
        path
        for path in SCRIPTS_DIR.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    )


def test_every_command_name_is_unique() -> None:
    assert len(COMMANDS_BY_NAME) == len(COMMANDS)


def test_every_command_names_a_script_that_exists() -> None:
    missing = [c.name for c in COMMANDS if not (SCRIPTS_DIR / c.script).is_file()]
    assert missing == []


def test_every_script_is_a_command_or_deliberately_not() -> None:
    registered = {command.script for command in COMMANDS}
    unaccounted = [
        name
        for path in _scripts()
        if (name := str(path.relative_to(SCRIPTS_DIR))) not in registered
        and path.name not in NOT_COMMANDS
        and name not in PENDING
    ]
    assert unaccounted == [], (
        "add these to COMMANDS, or to NOT_COMMANDS with a reason: " + ", ".join(unaccounted)
    )


def test_pending_only_shrinks() -> None:
    """A script that has been registered must have been struck off PENDING.

    Without this the list would quietly become a list of things that were once
    unfinished, which is not the same thing and would never reach empty.
    """
    registered = {command.script for command in COMMANDS}
    assert PENDING & registered == set()
    assert PENDING <= {str(p.relative_to(SCRIPTS_DIR)) for p in _scripts()}


def test_command_names_are_hyphenated_lowercase() -> None:
    bad = [c.name for c in COMMANDS if c.name != c.name.lower() or "_" in c.name]
    assert bad == []


def test_summaries_are_one_sentence() -> None:
    bad = [c.name for c in COMMANDS if not c.summary or "\n" in c.summary]
    assert bad == []


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c.name)
def test_parser_builds_and_suggests(command) -> None:
    """Every command's parser exists, builds, and offers a hint on a typo."""
    parser = load_parser_owner(command).build_parser()
    assert isinstance(parser, SuggestingArgumentParser), (
        f"{command.name} builds a plain ArgumentParser, so a typo in it gets no hint"
    )
    parser.format_help()
