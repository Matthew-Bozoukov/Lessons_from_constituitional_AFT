# ABOUTME: Load the gated MoReBench dilemma set from the Hub and parse its rubric column.
# ABOUTME: One record per dilemma, carrying the criteria the judge will score against.
"""The MoReBench dilemma set.

The public release (`morebench/morebench`, `morebench_public.csv`) is 500 of the paper's
1,000 dilemmas, each with 20-47 weighted rubric criteria — 11,450 criteria in total, which is
what makes judging rather than generation the cost of this eval.

Two columns carry the split this repo cares about: `ROLE_DOMAIN` separates dilemmas where the
model ADVISES a human (ai_advisor, 293) from ones where it DECIDES as an agent (ai_agent,
207), and `DILEMMA_TYPE` separates short/long/expert cases.

The repo is gated, so the token must have been granted access; a 401/403 here means the
account has not accepted the dataset terms, not that the token is wrong.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import pandas as pd

from src.infra.huggingface import hf_token


def load_dilemmas(cfg) -> list[dict]:
    """Every dilemma of the configured split, with criteria parsed.

    Args:
        cfg: Merged eval config; reads `source.repo`, `source.file`, `source.theory`.

    Returns:
        One dict per dilemma: task_id, dilemma text, context, source/type/role domain and
        the parsed rubric criteria.
    """
    from huggingface_hub import hf_hub_download

    src = cfg.source
    path = hf_hub_download(src.repo, src.file, repo_type="dataset", token=hf_token())
    frame = pd.read_csv(path)
    theory = src.get("theory", "neutral")
    if theory:
        frame = frame[frame["THEORY"] == theory]
    rows = []
    for idx, row in frame.iterrows():
        criteria = literal_eval(row["RUBRIC"]) if isinstance(row["RUBRIC"], str) else row["RUBRIC"]
        rows.append({
            # The public CSV ships no id column although every judgement is keyed on one;
            # the row index is stable because nothing after this filters or reorders.
            "task_id": f"row_{idx}",
            "dilemma": row["DILEMMA"],
            "context": row.get("CONTEXT"),
            "dilemma_source": row.get("DILEMMA_SOURCE"),
            "dilemma_type": row.get("DILEMMA_TYPE"),
            "role_domain": row.get("ROLE_DOMAIN"),
            "theory": row.get("THEORY"),
            "criteria": [{"id": c["id"], "title": c["title"],
                          "weight": c["weight"],
                          "dimension": c["annotations"]["rubric_dimension"]}
                         for c in criteria],
        })
    assert rows, "MoReBench source produced no dilemmas"
    return rows


def smoke_subset(dilemmas: list[dict]) -> list[dict]:
    """A handful of dilemmas covering both role domains, for the smoke path."""
    advisor = [d for d in dilemmas if d["role_domain"] == "ai_advisor"][:2]
    agent = [d for d in dilemmas if d["role_domain"] == "ai_agent"][:2]
    return advisor + agent
