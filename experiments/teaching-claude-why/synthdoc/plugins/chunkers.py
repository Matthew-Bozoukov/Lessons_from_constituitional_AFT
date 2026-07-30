# ABOUTME: Chunkers cut a spec into SpecChunks at a chosen granularity.
# ABOUTME: chunk_id is structural, so it survives spec text edits and coverage joins stay valid.

from __future__ import annotations

import re
from typing import Any, Iterable

from ..core.registry import register
from ..core.specs import SpecSource
from ..core.types import SpecChunk

# Cues used to annotate chunks. Cheap and transparent on purpose: meta is for
# slicing coverage reports, not for gating anything.
_PRESCRIPTIVE = re.compile(
    r"\b(should|must|never|always|avoid|do not|don't|refuse|ought to|is required)\b", re.I
)
_TRADEOFF = re.compile(
    r"\b(unless|however|but |except|balance|weigh|trade-?off|tension|competing|"
    r"rather than|on the other hand|in some cases)\b",
    re.I,
)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BOLD_LEAD = re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*\*\*(.+?)\*\*[.:]?\s*")


def _slug(text: str, limit: int = 48) -> str:
    """Return a stable, filesystem- and ID-safe slug for a heading."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:limit] or "root").rstrip("_")


def _meta(text: str) -> dict[str, Any]:
    """Annotate a chunk with register / tradeoff cues used by coverage reporting."""
    return {
        "register": "prescriptive" if _PRESCRIPTIVE.search(text) else "descriptive",
        "has_tradeoffs": bool(_TRADEOFF.search(text)),
        "n_words": len(text.split()),
    }


def _sections(text: str, min_level: int = 2) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs at or below a heading level.

    Args:
        text: Spec text.
        min_level: Headings at this level or shallower start a new section.

    Returns:
        List of (heading title, body text). Content before the first heading is
        returned under the title "preamble" when non-empty.
    """
    out: list[tuple[str, list[str]]] = []
    current_title = "preamble"
    current: list[str] = []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) <= min_level:
            if any(s.strip() for s in current):
                out.append((current_title, current))
            current_title = m.group(2).strip()
            current = []
        else:
            current.append(line)
    if any(s.strip() for s in current):
        out.append((current_title, current))
    return [(t, "\n".join(b).strip()) for t, b in out]


def _strip_comments(text: str) -> str:
    """Remove HTML comments (the repo's ABOUTME headers) from spec text."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _emit(
    spec_id: str,
    granularity: str,
    items: Iterable[tuple[str, int, str]],
    min_words: int,
) -> list[SpecChunk]:
    """Build SpecChunks from (parent, order_idx, text) triples, dropping stubs."""
    chunks: list[SpecChunk] = []
    for parent, idx, body in items:
        body = body.strip()
        if len(body.split()) < min_words:
            continue
        chunks.append(
            SpecChunk(
                spec_id=spec_id,
                chunk_id=f"{spec_id}/{granularity}/{parent}/{idx:03d}",
                text=body,
                granularity=granularity,
                parent_id=f"{spec_id}/{parent}",
                order_idx=idx,
                meta=_meta(body),
            )
        )
    return chunks


@register("chunker", "section")
class SectionChunker:
    """One chunk per top-level section. The coarsest granularity."""

    def __init__(self, min_words: int = 15, heading_level: int = 2, **_: Any) -> None:
        """Initialize.

        Args:
            min_words: Sections shorter than this are dropped.
            heading_level: Headings at or above this level start a section.
        """
        self.min_words = min_words
        self.heading_level = heading_level

    def chunk(self, spec: SpecSource) -> list[SpecChunk]:
        """Cut the spec into section chunks."""
        text = _strip_comments(spec.text)
        items = [
            (_slug(title), i, f"{title}\n\n{body}" if title != "preamble" else body)
            for i, (title, body) in enumerate(_sections(text, self.heading_level))
        ]
        return _emit(spec.spec_id, "section", items, self.min_words)


@register("chunker", "trait")
class TraitChunker:
    """One chunk per named principle: a bolded lead-in or a deeper heading.

    Falls back to paragraph splitting inside sections that use neither, so a spec
    written without explicit trait markers still chunks sensibly.
    """

    def __init__(self, min_words: int = 12, **_: Any) -> None:
        """Initialize.

        Args:
            min_words: Traits shorter than this are dropped.
        """
        self.min_words = min_words

    def chunk(self, spec: SpecSource) -> list[SpecChunk]:
        """Cut the spec into trait chunks."""
        text = _strip_comments(spec.text)
        items: list[tuple[str, int, str]] = []
        for title, body in _sections(text, 2):
            parent = _slug(title)
            blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
            idx = 0
            for block in blocks:
                # A block that is itself a list of bolded principles splits further.
                lead_lines = [ln for ln in block.splitlines() if _BOLD_LEAD.match(ln)]
                if len(lead_lines) > 1:
                    for piece in _split_on_leads(block):
                        items.append((parent, idx, piece))
                        idx += 1
                else:
                    items.append((parent, idx, block))
                    idx += 1
        return _emit(spec.spec_id, "trait", items, self.min_words)


def _split_on_leads(block: str) -> list[str]:
    """Split a block into pieces, each starting at a bolded lead-in line."""
    pieces: list[list[str]] = []
    for line in block.splitlines():
        if _BOLD_LEAD.match(line) or not pieces:
            pieces.append([line])
        else:
            pieces[-1].append(line)
    return ["\n".join(p).strip() for p in pieces if "".join(p).strip()]


@register("chunker", "bullet")
class BulletChunker:
    """One chunk per bullet or numbered item. The finest granularity.

    Continuation lines are folded into their bullet. Prose paragraphs that contain
    no bullets are kept whole so no spec content is silently dropped.
    """

    def __init__(self, min_words: int = 8, keep_prose: bool = True, **_: Any) -> None:
        """Initialize.

        Args:
            min_words: Bullets shorter than this are dropped.
            keep_prose: Also emit non-bulleted paragraphs as chunks.
        """
        self.min_words = min_words
        self.keep_prose = keep_prose

    def chunk(self, spec: SpecSource) -> list[SpecChunk]:
        """Cut the spec into bullet chunks."""
        text = _strip_comments(spec.text)
        items: list[tuple[str, int, str]] = []
        for title, body in _sections(text, 2):
            parent = _slug(title)
            idx = 0
            buf: list[str] = []
            prose: list[str] = []

            def flush_bullet() -> None:
                nonlocal idx, buf
                if buf:
                    items.append((parent, idx, "\n".join(buf).strip()))
                    idx += 1
                    buf = []

            def flush_prose() -> None:
                nonlocal idx, prose
                if self.keep_prose and any(p.strip() for p in prose):
                    items.append((parent, idx, "\n".join(prose).strip()))
                    idx += 1
                prose = []

            for line in body.splitlines():
                if _BULLET.match(line):
                    flush_prose()
                    flush_bullet()
                    buf = [line]
                elif buf and (line.startswith((" ", "\t")) or line.strip()):
                    buf.append(line)
                elif buf:
                    flush_bullet()
                else:
                    prose.append(line)
            flush_bullet()
            flush_prose()
        return _emit(spec.spec_id, "bullet", items, self.min_words)


def build_chunker(cfg: dict[str, Any]):
    """Instantiate the chunker named by a spec.chunker config block.

    Args:
        cfg: Mapping with a "granularity" key plus chunker kwargs.

    Returns:
        A chunker instance.
    """
    from ..core.registry import resolve

    params = dict(cfg or {})
    granularity = params.pop("granularity", "bullet")
    return resolve("chunker", granularity)(**params)
