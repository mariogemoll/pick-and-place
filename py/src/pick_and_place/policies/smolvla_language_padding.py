# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Stop a SmolVLA step running 36 language tokens that are all padding.

``SmolVLAConfig.pad_language_to`` is ``"longest"``, but the tokenizer step is not
built from that config: ``make_pre_post_processors`` loads the processor saved
beside the checkpoint, and ``smolvla_base``'s pins ``padding="max_length"`` with
``max_length=48``. This dataset's one task string is 12 tokens, so a step carries
48 and runs 36 of them through all 16 VLM layers and their backward.

They are masked out -- ``embed_prefix`` builds ``pad_masks`` from the attention
mask, ``make_att_2d_masks`` lets nothing attend to a padded position, and
``position_ids`` come from a cumulative sum that padding does not advance -- so
the tokens cost time and change nothing. Dropping them is worth **1.27x** on a
step whose vision tower is already cached, and it was invisible while the tower
was 65% of a step.

The loss moves by about 1e-3 relative, which is bfloat16 rounding rather than a
different computation: the VLM is loaded with ``torch_dtype="bfloat16"``, so
changing the sequence length changes the order things are accumulated in. A
sweep over padding lengths that are all mathematically identical moves the loss
by the same order, which is what ``check_smolvla_prefix_cache.py`` shows.

One caveat carries over from `torch.compile`: ``"longest"`` is a constant length
only because this dataset carries exactly one task string. A multi-task dataset
would give a length that varies per batch, which recompiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.processor.pipeline import PolicyProcessorPipeline


def pad_language_to(
    preprocessor: PolicyProcessorPipeline,
    padding: str = "longest",
    max_length: int | None = None,
) -> tuple[str, int]:
    """Change what one preprocessor's tokenizer step pads to.

    Returns what it was padding to and at what length, so a caller can say what
    it changed. ``max_length`` is left alone unless given, which matters only for
    ``padding="max_length"``.
    """
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

    for step in preprocessor.steps:
        if isinstance(step, TokenizerProcessorStep):
            previous = (step.padding, step.max_length)
            step.padding = padding
            if max_length is not None:
                step.max_length = max_length
            return previous
    raise ValueError("this preprocessor has no tokenizer step to change")
