# ABOUTME: Builds the Target a producer traces back FROM — from a TURF rubric yaml, from a
# ABOUTME: LESS-style validation jsonl, or from the rollouts of a source that has outcomes.

"""Where a Target comes from.

Three ways to say "this is the behaviour I want explained", in increasing order of how
much evidence they carry:

    from_rubric   a rubric yaml + one or more case records. The rubric states the
                  BEHAVIOUR — principle, context, what satisfying and violating look like
                  — and never candidate properties. That prohibition is the whole reason
                  the rubrics are worth having: a rubric that lists properties hands the
                  discovery its answer, and the property list stops being a finding.
    from_dval     a validation jsonl in interchange form, one subtask per file. This is
                  LESS's Dval: the m in `max_j I[x, j]`.
    from_outcomes an already-loaded rollout source, split on its own outcome labels. The
                  "good traces" box of the flow diagram, and the cheapest target to make
                  since the eval already judged every rollout.

The known weakness, recorded here because it bit the 2026-08-17 LESS run: a Dval built from
eval exports can hold far fewer DISTINCT prompts than rows (60 rows, 33 prompts), which
weights repeated prompts more heavily than an independent draw would. `from_dval` reports
the distinct-prompt count so a caller sees the effective size rather than the nominal one.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.properties.sources.base import Record, Target, first_turns

POLARITIES = ("satisfy", "violate")


def _check_polarity(polarity: str) -> str:
    """Validate a polarity string.

    Args:
        polarity: "satisfy" or "violate".

    Returns:
        The polarity, unchanged.

    Raises:
        ValueError: If it is neither.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}, got {polarity!r}")
    return polarity


def from_rubric(rubric: str, cases: list[Record], polarity: str = "satisfy",
                target_id: str | None = None) -> Target:
    """Build a Target from a behaviour rubric plus the cases exhibiting it.

    Args:
        rubric: Path to a rubric yaml with a `principle_specific_details` block.
        cases: Records exhibiting the behaviour (a TURF case, or an eval export row).
        polarity: "satisfy" attributes good behaviour; "violate" attributes failures.
        target_id: Override the id; defaults to the rubric's filename stem.

    Returns:
        The Target.

    Raises:
        ValueError: If the rubric has no `principle_specific_details`, or names a
            candidate property under a `properties`/`hypotheses` key.
    """
    from omegaconf import OmegaConf

    path = Path(rubric)
    cfg = OmegaConf.to_container(OmegaConf.load(path))
    behaviour = (cfg or {}).get("principle_specific_details")
    if not behaviour:
        raise ValueError(f"{path} has no `principle_specific_details`: a rubric states "
                         "the behaviour, and that block is the behaviour")
    forbidden = [k for k in ("properties", "hypotheses", "candidate_properties")
                 if k in (cfg or {})]
    if forbidden:
        raise ValueError(
            f"{path} declares {forbidden} — a rubric describes the BEHAVIOUR only. "
            "Naming candidate properties in it makes the discovery circular; put "
            "hypotheses in the ablation config, after a producer has found them.")
    return Target(target_id=target_id or path.stem, behaviour=str(behaviour).strip(),
                  polarity=_check_polarity(polarity), cases=list(cases),
                  provenance={"rubric": str(path), "n_cases": len(cases)})


def from_dval(path: str, behaviour: str, polarity: str = "satisfy",
              subtask: str | None = None, target_id: str | None = None) -> Target:
    """Build a Target from a validation jsonl of interchange rows.

    Args:
        path: A jsonl whose rows carry `messages` (reasoning in `reasoning_content`).
        behaviour: The behaviour these rows exemplify, in prose.
        polarity: "satisfy" or "violate".
        subtask: Subtask label; defaults to the filename stem, matching LESS's convention
            that one FILE is one subtask.
        target_id: Override the id; defaults to the subtask.

    Returns:
        The Target, with `provenance["distinct_prompts"]` alongside the row count — they
        are not the same number and the difference is the effective sample size.

    Raises:
        ValueError: If a row has no usable user/assistant turn pair.
    """
    local = Path(path)
    rows = [json.loads(line) for line in
            local.read_text(encoding="utf-8").split("\n") if line.strip()]
    cases = []
    for index, row in enumerate(rows):
        turns = first_turns(row.get("messages", []))
        if turns is None:
            raise ValueError(f"{local}:{index} has no user/assistant turn pair")
        query, response, reasoning = turns
        cases.append(Record(record_id=f"{local.stem}#{index}", query=query,
                            response=response, reasoning=reasoning,
                            metadata=dict(row.get("metadata") or {}), raw=row))
    name = subtask or local.stem
    return Target(target_id=target_id or name, behaviour=behaviour.strip(),
                  polarity=_check_polarity(polarity), cases=cases, subtask=name,
                  provenance={"dval": str(local), "n_cases": len(cases),
                              "distinct_prompts": len({c.query for c in cases})})


def from_outcomes(records: list[Record], behaviour: str, polarity: str = "satisfy",
                  target_id: str = "outcomes", limit: int | None = None) -> Target:
    """Build a Target by splitting a rollout source on its own outcome labels.

    Args:
        records: Records from a source whose adapter declares `has_outcomes`.
        behaviour: The behaviour in prose (what the eval's judge was measuring).
        polarity: "satisfy" selects the records that did NOT violate; "violate" selects
            the ones that did.
        target_id: Id for the Target.
        limit: Keep at most N cases.

    Returns:
        The Target.

    Raises:
        ValueError: If no record carries an outcome, or the split is empty.
    """
    _check_polarity(polarity)
    judged = [r for r in records if r.outcome is not None]
    if not judged:
        raise ValueError(
            "no record carries an outcome: an unjudged rollout set cannot be split into "
            "good and bad traces. Run the eval's judge first, or use from_rubric.")
    want_violation = polarity == "violate"
    cases = [r for r in judged if bool(r.outcome.get("violation")) == want_violation]
    if not cases:
        raise ValueError(f"no {polarity} cases among {len(judged)} judged records")
    return Target(target_id=target_id, behaviour=behaviour.strip(), polarity=polarity,
                  cases=cases[:limit] if limit else cases,
                  provenance={"n_judged": len(judged), "n_cases": len(cases),
                              "n_unjudged": len(records) - len(judged)})
