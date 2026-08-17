# ABOUTME: Split the vendored LLMBar ChatML prompt into system/user messages and render one
# ABOUTME: comparison in a chosen presentation order.

"""Prompt rendering for the `llmbar` eval.

The vendored asset is upstream's own ChatML file (`<|im_start|>system … <|im_end|>`). It is
kept in that form so a diff against upstream stays trivial, and split here rather than at
authoring time so nothing about the vendored bytes has to be trusted.

Presentation order is a parameter because every item runs BOTH ways. Upstream's own rules
tell the evaluator that order "should NOT affect your judgment"; whether that holds is a
measurement, and it is the same order-flip metric `docs/in_domain_evals.md` specifies for
the courtroom eval — position bias is a known LLM-judge failure and it cannot be faked by
writing more.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.eval.deliberation.llmbar.data import Item

ASSET = Path(__file__).parent / "assets" / "vanilla.txt"

_BLOCK = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)


@lru_cache(maxsize=1)
def templates() -> tuple[str, str]:
    """Return `(system, user)` templates from the vendored asset.

    Raises:
        AssertionError: if the asset stops containing exactly one system and one user
            block — i.e. if a re-vendor changed its shape and the split would silently
            drop half the prompt.
    """
    blocks = dict(_BLOCK.findall(ASSET.read_text()))
    assert set(blocks) == {"system", "user"}, (
        f"{ASSET} should hold exactly one system and one user block, found {sorted(blocks)}")
    return blocks["system"].strip(), blocks["user"].strip()


def messages_for(item: Item, swapped: bool) -> list[dict]:
    """Render one comparison as OpenAI-style messages.

    Args:
        item: The LLMBar item.
        swapped: False presents upstream's `output_1` as "Output (a)"; True presents
            `output_2` there. `expected_choice` below converts a reply back to a gold
            comparison, so callers never have to track the mapping themselves.

    Returns:
        `[system, user]`.
    """
    system, user = templates()
    first, second = ((item.output_2, item.output_1) if swapped
                     else (item.output_1, item.output_2))
    body = (user.replace("{input}", item.instruction)
                .replace("{output_1}", first)
                .replace("{output_2}", second))
    return [{"role": "system", "content": system},
            {"role": "user", "content": body}]


def expected_choice(item: Item, swapped: bool) -> str:
    """Which displayed slot ("a"/"b") holds the instruction-following output in this order."""
    shown_first = 2 if swapped else 1
    return "a" if item.gold == shown_first else "b"


def chosen_output(choice: str, swapped: bool) -> int:
    """Map a displayed slot back to upstream's output number (1 or 2), 0 when unparsed.

    This is what makes the two orders comparable: "picked (a) in the normal order" and
    "picked (b) in the swapped order" are the SAME underlying judgment, and consistency has
    to be measured on the underlying one.
    """
    if choice not in ("a", "b"):
        return 0
    if choice == "a":
        return 2 if swapped else 1
    return 1 if swapped else 2
