# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The command table is data, so something has to check it against the tree.

``pick_and_place/cli/commands.py`` names modules and script paths as strings,
which is what lets ``pap --help`` list every command without importing any of
them. The cost is that a wrong string is invisible until someone runs that one
command. These tests pay it back: every entry resolves, every entry's parser
builds, and every script is either registered or on the list of things that are
deliberately not commands.

That last one is what closes the tree: a new script under ``scripts/`` fails
this suite until it is either a ``pap`` command or explicitly declared not to
be. It carried a shrinking ``PENDING`` list while the commands were converted;
the list reached empty and went away with it.
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
    ]
    assert unaccounted == [], (
        "add these to COMMANDS, or to NOT_COMMANDS with a reason: " + ", ".join(unaccounted)
    )


def test_command_names_are_hyphenated_lowercase() -> None:
    bad = [c.name for c in COMMANDS if c.name != c.name.lower() or "_" in c.name]
    assert bad == []


def test_summaries_are_one_sentence() -> None:
    bad = [c.name for c in COMMANDS if not c.summary or "\n" in c.summary]
    assert bad == []


def test_every_command_has_a_group() -> None:
    """`pap --help` lists commands under their group, so an ungrouped one vanishes."""
    assert [c.name for c in COMMANDS if not c.group] == []


def test_groups_are_contiguous() -> None:
    """A group named twice would print twice, splitting its commands in the help."""
    seen: list[str] = []
    for command in COMMANDS:
        if not seen or seen[-1] != command.group:
            seen.append(command.group)
    assert len(seen) == len(set(seen)), f"group order is interleaved: {seen}"


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c.name)
def test_parser_builds_and_suggests(command) -> None:
    """Every command's parser exists, builds, and offers a hint on a typo."""
    owner = load_parser_owner(command)
    if command.typed_config:
        assert hasattr(owner, "parse_arguments"), (
            f"{command.name} is marked typed_config but exposes no parse_arguments"
        )
        return
    parser = owner.build_parser()
    assert isinstance(parser, SuggestingArgumentParser), (
        f"{command.name} builds a plain ArgumentParser, so a typo in it gets no hint"
    )
    parser.format_help()
