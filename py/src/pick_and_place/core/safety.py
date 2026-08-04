# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Exception-keyed recovery for the real-arm continuous loop.

``recover_on`` is the one primitive: run a body, and if it raises one of the
given exception types, run the caller's own recovery action and swallow the
exception — the ``try/except`` a task script would otherwise hand-write at
every recovery point, made reusable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager


class EpisodeAborted(Exception):
    """Raised by a task body to abort the current episode and trigger
    recovery, when the failure isn't already a more specific exception."""


@contextmanager
def recover_on(*exceptions: type[BaseException], recover: Callable[[], None]) -> Iterator[None]:
    """Run the body; on any of ``exceptions``, run ``recover()`` and swallow it
    so control falls through to past the guarded block, as if it had
    completed normally."""
    try:
        yield
    except exceptions:
        recover()
