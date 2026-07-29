# ABOUTME: Builds the template variables shared by generation, revision, and rating
# ABOUTME: prompts, so all three see the scenario described identically.

from __future__ import annotations

from typing import Any

from ..control import loader
from .types import ScenarioSpec


def scenario_vars(scenario: ScenarioSpec, **extra: Any) -> dict[str, Any]:
    """Assemble the Jinja variables describing one experimental condition.

    Axis fragments come from control/prompts/axes.yaml, one per axis the recipe
    declares. This is the mechanism that keeps axes orthogonal: a generator never
    branches on an axis value, it splices in the fragment the registry hands back.

    Args:
        scenario: The condition being rendered.
        **extra: Additional template variables (e.g. `document` for revisers).

    Returns:
        Mapping of template variable name to value.
    """
    variables: dict[str, Any] = {
        "spec_id": scenario.spec_id,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "parent_id": c.parent_id.split("/")[-1] if c.parent_id else "",
                "granularity": c.granularity,
                "meta": dict(c.meta),
            }
            for c in scenario.chunks
        ],
        "n_chunks": len(scenario.chunks),
        "grouping_strategy": scenario.grouping_strategy,
        "doc_type": scenario.doc_type,
        "doc_type_instructions": loader.doc_type_instructions(scenario.doc_type),
        "axis_fragments": [
            loader.axis_fragment(axis, str(value))
            for axis, value in sorted(scenario.axes.items())
        ],
        "axes": dict(scenario.axes),
        "seed": scenario.seed,
    }
    variables.update(extra)
    return variables
