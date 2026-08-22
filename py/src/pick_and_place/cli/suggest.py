# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""An ``ArgumentParser`` that proposes a close match for a token it rejected.

A two-level tree is two places to mistype, and argparse's default is
``unrecognized arguments: --chekpoint x`` with no hint at all. Argparse already
accepts unambiguous prefixes, so ``--integration-step`` works on its own and
only real typos reach the error path.

Python 3.14 does this with ``ArgumentParser(suggest_on_error=True)``. The
project is on 3.13 and its dependency graph is not worth moving for a hint, so
this is the twenty lines that stand in until 3.14 is viable -- at which point
this module goes away and the flag replaces it.

The cutoff matters: at 0.6 an unrelated token draws no suggestion rather than a
confidently wrong one.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections.abc import Iterable
from typing import NoReturn

#: Below this ratio a candidate is not offered at all. Tuned so that a genuine
#: typo is caught and an unrelated flag is left alone.
CUTOFF = 0.6


class SuggestingArgumentParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that proposes a close match for a bad token."""

    def error(self, message: str) -> NoReturn:
        hint = self._suggest(message)
        super().error(f"{message}\n{hint}" if hint else message)
        raise AssertionError("unreachable: ArgumentParser.error exits")

    def _suggest(self, message: str) -> str | None:
        """Turn argparse's own message back into the token it rejected.

        Both a misspelled flag and a misspelled subcommand arrive through
        ``error()``, which is why one method covers leaf names and flags.
        """
        unknown = re.search(r"unrecognized arguments: (\S+)", message)
        if unknown and unknown.group(1).startswith("-"):
            return self._closest(unknown.group(1), self._option_strings_in_tree())
        bad_choice = re.search(r"invalid choice: '([^']*)'", message)
        if bad_choice:
            return self._closest(bad_choice.group(1), self._leaf_names())
        return None

    def _option_strings_in_tree(self) -> list[str]:
        """Every flag this parser or any leaf under it accepts.

        The tree walk is not thoroughness for its own sake. Argparse hands
        leftover tokens back to the *root*: a leaf parses what it recognizes and
        an unrecognized flag propagates up, so by the time ``error()`` runs, the
        parser reporting the typo is the one parser that has never heard of the
        flag it was a typo for. Searching only ``self`` would offer ``--help``
        or nothing at all for every misspelling below the first level.
        """
        names = list(self._option_string_actions)
        for leaf in self._leaf_parsers():
            names += list(leaf._option_string_actions)
        # Deduplicated, because a flag the leaves share -- which is most of them,
        # now that they come from cli/ -- would otherwise be offered once per
        # leaf: "did you mean: --manifest, --manifest, --manifest?"
        return list(dict.fromkeys(names))

    def _leaf_names(self) -> list[str]:
        return [
            str(choice)
            for action in self._actions
            if action.choices
            for choice in action.choices
        ]

    def _leaf_parsers(self) -> list[argparse.ArgumentParser]:
        return [
            parser
            for action in self._actions
            if isinstance(action, argparse._SubParsersAction)
            for parser in action.choices.values()
        ]

    @staticmethod
    def _closest(token: str, candidates: Iterable[str]) -> str | None:
        matches = difflib.get_close_matches(token, list(candidates), n=3, cutoff=CUTOFF)
        return "did you mean: " + ", ".join(matches) + "?" if matches else None
