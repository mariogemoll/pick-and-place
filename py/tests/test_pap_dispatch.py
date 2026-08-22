# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The command table is data, so something has to check it against the tree.

``pick_and_place/cli/commands.py`` names modules as strings, which is what lets
``pap --help`` list every command without importing any of them. The cost is
that a wrong string is invisible until someone runs that one command. These
tests pay it back: every entry resolves, every entry's parser builds, and every
module in ``cli/`` is either registered or on the list of things that are
deliberately not commands.

That last one is what closes the tree: a new module under ``cli/`` fails this
suite until it is either a ``pap`` command or explicitly declared not to be. It
carried a shrinking ``PENDING`` list while the commands were converted; the list
reached empty and went away with it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pick_and_place.cli import commands as commands_module
from pick_and_place.cli.commands import COMMANDS, COMMANDS_BY_NAME
from pick_and_place.cli.dispatch import load_parser_owner
from pick_and_place.cli.suggest import SuggestingArgumentParser

CLI_DIR = Path(commands_module.__file__).resolve().parent

#: Modules under ``cli/`` that are not commands, and why.
NOT_COMMANDS = {
    "commands.py": "the table itself",
    "dispatch.py": "how the table's strings become code",
    "pap.py": "the dispatcher itself",
    "suggest.py": "the parser that offers a hint on a typo",
    # The shared argument groups. A flag declared here is declared once for
    # every command that takes it; see cli/common.py.
    "calibration.py": "shared argument group",
    "common.py": "shared argument group",
    "dataset.py": "shared argument group",
    "evaluation.py": "shared argument group",
    "policy.py": "shared argument group",
    "rig.py": "shared argument group",
    "scene.py": "shared argument group",
    "training.py": "shared argument group",
}


def _cli_modules() -> list[str]:
    return sorted(path.name for path in CLI_DIR.glob("*.py") if path.name != "__init__.py")


def test_every_command_name_is_unique() -> None:
    assert len(COMMANDS_BY_NAME) == len(COMMANDS)


def test_every_command_names_a_module_that_exists() -> None:
    """Found without being imported, so the check costs nothing a heavy command drags in."""
    missing = [c.name for c in COMMANDS if importlib.util.find_spec(c.module) is None]
    assert missing == []


def test_every_command_lives_under_cli() -> None:
    """A command elsewhere in the package would leave the tree below unclosed."""
    stray = [c.name for c in COMMANDS if not c.module.startswith("pick_and_place.cli.")]
    assert stray == []


def test_every_parser_module_is_named_for_its_command() -> None:
    """The second module a heavy command needs, so that it can be found from the first."""
    wrong = [
        c.name
        for c in COMMANDS
        if c.parser is not None and c.parser != f"{c.module}_parser"
    ]
    assert wrong == []


def test_every_cli_module_is_a_command_or_deliberately_not() -> None:
    registered = {
        f"{module.rsplit('.', 1)[1]}.py"
        for command in COMMANDS
        for module in (command.module, command.parser)
        if module
    }
    unaccounted = [
        name for name in _cli_modules() if name not in registered and name not in NOT_COMMANDS
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
