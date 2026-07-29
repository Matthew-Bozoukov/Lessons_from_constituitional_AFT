# ABOUTME: Loads and renders everything under control/: clause sets, prompt packs, rubrics.
# ABOUTME: The only module that reads control YAML, so prose never leaks into Python.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.types import Clause, ClauseSet, FakeClause

CONTROL = Path(__file__).resolve().parent
CLAUSES_DIR = CONTROL / "clauses"
PROMPTS_DIR = CONTROL / "prompts"
CONFIGS_DIR = CONTROL / "configs"

# Prompt packs available to `pack()`.
PACKS = ("items", "rubrics", "pressure", "ood")


class PromptError(KeyError):
    """Raised when a control-plane entry is missing or malformed."""


@lru_cache(maxsize=8)
def _load_yaml(path: str) -> dict[str, Any]:
    """Load and cache a YAML file as a dict.

    Args:
        path: Absolute path string (a string so the result is hashable for the cache).

    Returns:
        The parsed mapping.

    Raises:
        PromptError: If the file is missing or is not a mapping.
    """
    p = Path(path)
    if not p.exists():
        raise PromptError(f"Control file not found: {p}")
    import yaml

    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise PromptError(f"Control file {p} is not a mapping")
    return data


def pack(name: str) -> dict[str, Any]:
    """Load a prompt pack from control/prompts/.

    Args:
        name: Pack name, one of PACKS.

    Returns:
        The pack contents.

    Raises:
        PromptError: If the pack name is unknown.
    """
    if name not in PACKS:
        raise PromptError(f"Unknown prompt pack {name!r}. Available: {list(PACKS)}")
    return _load_yaml(str(PROMPTS_DIR / f"{name}.yaml"))


def clause_set(spec_id: str) -> ClauseSet:
    """Load a frozen clause set from control/clauses/.

    Args:
        spec_id: File stem under control/clauses/, e.g. "approved_constitution_v1".

    Returns:
        The ClauseSet.

    Raises:
        PromptError: If the file is missing, declares no clauses, or contains a
            duplicate clause id or a distractor pointing at an unknown clause.
    """
    raw = _load_yaml(str(CLAUSES_DIR / f"{spec_id}.yaml"))
    entries = raw.get("clauses") or []
    if not entries:
        raise PromptError(f"Clause set {spec_id!r} declares no clauses")

    clauses = tuple(
        Clause.from_dict(
            {**e, "text": _tidy(e["text"]), "rationale": _tidy(e.get("rationale", ""))}
        )
        for e in entries
    )
    ids = [c.clause_id for c in clauses]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PromptError(f"Clause set {spec_id!r} has duplicate clause ids: {duplicates}")

    fakes = tuple(
        FakeClause.from_dict({**f, "text": _tidy(f["text"]), "why_fake": _tidy(f.get("why_fake", ""))})
        for f in raw.get("fake_clauses") or ()
    )
    orphans = sorted({f.near_clause_id for f in fakes} - set(ids))
    if orphans:
        raise PromptError(
            f"Clause set {spec_id!r} has distractors pointing at unknown clauses: {orphans}. "
            f"A distractor must name the real clause it is confusable with."
        )

    # Entailments are the Tier B ground truth; a dangling one is a typo that would only
    # surface much later, so it is caught here.
    dangling = sorted({e for c in clauses for e in c.entailments} - set(ids))
    if dangling:
        raise PromptError(f"Clause set {spec_id!r} has entailments naming unknown clauses: {dangling}")

    return ClauseSet(
        spec_id=raw.get("spec_id", spec_id),
        clauses=clauses,
        fakes=fakes,
        # Literal (`|`) blocks whose line structure is meaningful - the numbered ordering
        # is handed to the judge verbatim, so it is stripped but never reflowed.
        priority_order=str(raw.get("priority_order", "")).strip(),
        priority_note=str(raw.get("priority_note", "")).strip(),
    )


def available_clause_sets() -> list[str]:
    """Return the clause set ids present in control/clauses/."""
    return sorted(p.stem for p in CLAUSES_DIR.glob("*.yaml"))


def rubric(axis: str) -> dict[str, Any]:
    """Return one judge rubric.

    Args:
        axis: Axis name, e.g. "compliance".

    Returns:
        The rubric mapping, with defaults filled in.

    Raises:
        PromptError: If the axis is undeclared or its rubric is missing a template.
    """
    axes = pack("rubrics").get("axes") or {}
    if axis not in axes:
        raise PromptError(f"Unknown rubric axis {axis!r}. Declared: {sorted(axes)}")
    entry = dict(axes[axis])
    if not entry.get("template"):
        raise PromptError(f"Rubric {axis!r} has no template")
    entry.setdefault("scale_max", 3)
    entry.setdefault("pass_at", 2)
    entry.setdefault("direction", "higher_better")
    entry.setdefault("fields", ["score", "rationale"])
    entry.setdefault("applies_to", [])
    # Clause fields this axis cannot be graded without. A clause set distilled from a
    # document that states rules but not reasons has nothing for the justification axis
    # to grade against, and scoring it anyway would measure agreement with an invented
    # rationale rather than recall of a stated one.
    entry.setdefault("requires", [])
    # Condition kinds this axis runs on (clean | pressure | ood). Empty means all of them.
    # Scoping an axis to `clean` is the cheapest real saving available: the stressed and
    # OOD items are 3/4 of every judge pass.
    entry.setdefault("conditions", [])
    entry.setdefault("title", axis)
    entry["axis"] = axis
    return entry


def declared_axes() -> list[str]:
    """Return every declared rubric axis."""
    return sorted((pack("rubrics").get("axes") or {}))


def axes_for_family(family: str) -> list[str]:
    """Return the axes whose rubric declares it applies to a family.

    Binding lives in the rubric rather than in the judge code, so adding an axis to
    a family is a YAML line.

    Args:
        family: Item family.

    Returns:
        Sorted axis names.
    """
    axes = pack("rubrics").get("axes") or {}
    return sorted(a for a, spec in axes.items() if family in (spec.get("applies_to") or []))


def wrapper(name: str) -> dict[str, Any]:
    """Return one robustness wrapper.

    Args:
        name: Wrapper name.

    Returns:
        The wrapper mapping.

    Raises:
        PromptError: If the wrapper is undeclared or declares an unknown kind.
    """
    wrappers = pack("pressure").get("wrappers") or {}
    if name not in wrappers:
        raise PromptError(f"Unknown pressure wrapper {name!r}. Declared: {sorted(wrappers)}")
    entry = dict(wrappers[name])
    kind = entry.get("kind")
    if kind not in ("system", "prefix", "history"):
        raise PromptError(f"Pressure wrapper {name!r} has unknown kind {kind!r}")
    entry["name"] = name
    return entry


def declared_wrappers() -> list[str]:
    """Return every declared robustness wrapper."""
    return sorted((pack("pressure").get("wrappers") or {}))


def ood_axis(name: str) -> dict[str, Any]:
    """Return one OOD distance axis with its values in increasing-distance order.

    Args:
        name: Axis name.

    Returns:
        The axis mapping.

    Raises:
        PromptError: If the axis is undeclared, has no distance-0 anchor, or lists
            its values out of distance order.
    """
    axes = pack("ood").get("axes") or {}
    if name not in axes:
        raise PromptError(f"Unknown OOD axis {name!r}. Declared: {sorted(axes)}")
    entry = dict(axes[name])
    values = list(entry.get("values") or [])
    if not values:
        raise PromptError(f"OOD axis {name!r} declares no values")
    distances = [int(v.get("distance", 0)) for v in values]
    if distances[0] != 0:
        raise PromptError(
            f"OOD axis {name!r} has no distance-0 anchor; the decay curve has nothing to start from"
        )
    if distances != sorted(distances):
        raise PromptError(f"OOD axis {name!r} lists values out of distance order: {distances}")
    entry["name"] = name
    entry["values"] = values
    return entry


def declared_ood_axes() -> list[str]:
    """Return every declared OOD distance axis."""
    return sorted((pack("ood").get("axes") or {}))


def render(template: str, **context: Any) -> str:
    """Render a control-plane template with Jinja2.

    StrictUndefined so a template referencing a field the caller forgot fails loudly
    instead of silently producing a prompt with a hole in it.

    Args:
        template: Template source.
        **context: Template variables.

    Returns:
        The rendered string.

    Raises:
        PromptError: If a referenced variable is undefined.
    """
    from jinja2 import Environment, StrictUndefined, UndefinedError

    env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    try:
        return env.from_string(template).render(**context).strip()
    except UndefinedError as e:
        raise PromptError(f"Template referenced an undefined variable: {e}") from e


def _tidy(text: str) -> str:
    """Collapse the whitespace YAML folded scalars introduce, preserving paragraphs."""
    if not text:
        return ""
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p).strip()
