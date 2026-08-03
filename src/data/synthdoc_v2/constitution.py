# ABOUTME: Stage 1 -- segment the constitution into individually addressable traits.
# ABOUTME: Deterministic, no LLM, so the trait set is stable across runs and testable offline.

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

# A principle is a numbered item whose title is bolded, e.g.
#   1. **Honesty and non-deception.** Do not help the user deceive...
_PRINCIPLE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*(.*)$")
_HEADING = re.compile(r"^##\s+(.*)$")


@dataclass(frozen=True)
class Trait:
    """One addressable section of the constitution.

    Attributes:
        trait_id: Stable identifier, e.g. "t3".
        index: 1-based position in the source document.
        name: Short title, e.g. "Avoid facilitating harm or illegality".
        text: The full principle text, title included.
    """

    trait_id: str
    index: int
    name: str
    text: str

    def as_dict(self) -> dict:
        """Return a JSON-serialisable form."""
        return asdict(self)


def _dedent_join(lines: list[str]) -> str:
    """Join continuation lines into one paragraph, collapsing indentation."""
    return " ".join(line.strip() for line in lines if line.strip())


def segment(path: str | Path) -> tuple[list[Trait], str]:
    """Split the constitution into its numbered traits plus the shared style section.

    The document has two parts: a numbered list of principles, and a prose section
    describing what an aligned response looks like. The principles become the traits a
    scenario can be built around; the prose is shared guidance injected everywhere,
    since it constrains tone rather than naming a distinct value.

    Args:
        path: Path to the constitution markdown.

    Returns:
        (traits, style_guidance).

    Raises:
        AssertionError: If no principles are found, which would silently produce an
            empty run.
    """
    lines = Path(path).read_text().splitlines()

    traits: list[Trait] = []
    style: list[str] = []
    current: tuple[int, str, list[str]] | None = None
    in_style = False

    def flush() -> None:
        if current is None:
            return
        idx, name, body = current
        text = f"**{name}** {_dedent_join(body)}".strip()
        traits.append(Trait(trait_id=f"t{idx}", index=idx, name=name.rstrip("."), text=text))

    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            flush()
            current = None
            # Everything after the principles list is tone/style guidance.
            in_style = "look" in heading.group(1).lower()
            continue
        if in_style:
            style.append(line)
            continue
        m = _PRINCIPLE.match(line)
        if m:
            flush()
            current = (int(m.group(1)), m.group(2), [m.group(3)])
        elif current is not None:
            current[2].append(line)
    flush()

    assert traits, f"no numbered principles found in {path}"
    indices = [t.index for t in traits]
    assert indices == sorted(indices), f"principles are out of order: {indices}"
    assert len(set(indices)) == len(indices), f"duplicate principle numbers: {indices}"
    return traits, "\n".join(style).strip()


def full_text(path: str | Path) -> str:
    """Return the whole constitution, for stages that inject the complete document."""
    return Path(path).read_text().strip()
