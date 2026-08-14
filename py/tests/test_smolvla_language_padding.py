# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Repadding a SmolVLA preprocessor's tokenizer step."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("lerobot")

from lerobot.processor.tokenizer_processor import TokenizerProcessorStep  # noqa: E402

from pick_and_place.policies.smolvla_language_padding import pad_language_to  # noqa: E402


def tokenizer_step(padding: str, max_length: int) -> TokenizerProcessorStep:
    """A step carrying only the two fields that matter, and no tokenizer.

    Built without `__init__` on purpose: the real one loads the SmolVLM tokenizer
    from the hub, and what is under test is which object gets found and rewritten.
    """
    step = object.__new__(TokenizerProcessorStep)
    step.padding = padding
    step.max_length = max_length
    return step


def preprocessor(*steps: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(steps=list(steps))


def test_repadding_reports_what_the_checkpoint_had():
    step = tokenizer_step("max_length", 48)
    previous = pad_language_to(preprocessor(object(), step, object()))
    assert previous == ("max_length", 48)
    assert step.padding == "longest"
    # Left alone: it is the truncation length as well, and "longest" ignores it.
    assert step.max_length == 48


def test_a_fixed_length_can_be_asked_for_too():
    step = tokenizer_step("max_length", 48)
    pad_language_to(preprocessor(step), "max_length", 24)
    assert (step.padding, step.max_length) == ("max_length", 24)


def test_a_preprocessor_without_a_tokenizer_is_an_error_rather_than_a_silent_pass():
    # Silently doing nothing would show up as a run that is mysteriously slow.
    with pytest.raises(ValueError, match="no tokenizer step"):
        pad_language_to(preprocessor(object()))
