# ABOUTME: Public API for the constitution-internalization eval suite (Tier A).
# ABOUTME: Self-contained: nothing here imports the rest of the repo, and nothing needs to.

"""Constitution internalization eval suite — a fast, direct proxy for internalization.

Retrieval saturates immediately and is not the metric. The signal is the gap between
knowing a clause and acting on it, and whether the model can articulate the clause's
*rationale* rather than just its rule.

Plug-and-play usage from anywhere in the repo:

    from src.eval.misalignment.internalization import load_config, run_eval, build_report

    cfg = load_config("src/eval/misalignment/internalization/control/configs/base.yaml", {"run.recipe": "difficult_advice"})
    result = run_eval(cfg)
    build_report(result.store, "output/internalization/report")

Everything tunable lives in `src/eval/misalignment/internalization/control/`: run configs in `control/configs/`, the
frozen clause set in `control/clauses/`, and every string a model sees in `control/prompts/`.
Extending the suite means registering a plugin and adding a config line — see `register`.

Tier B (counterfactual clause inversion, held-out generalization, recipe ablations,
persistence) and Tier B-lite (linear probes, self-report vs behavior) are NOT implemented
here; they need extra training runs or model internals. See `src/eval/misalignment/internalization/README.md`.
"""

from .analysis import (
    METRICS,
    check_comparable,
    headline_table,
    health_warnings,
    paired_pressure,
    per_clause_rates,
    rates,
    scatter_pairs,
)
from .config import ConfigError, load_config, load_config_dict, validate
from .core.registry import register, resolve
from .core.store import ResultsStore, RunContext, ScoreRow
from .core.types import Clause, ClauseSet, Completion, FakeClause, Item, Verdict
from .items import ItemSet, build_itemset
from .judges import JudgeConfig, RubricJudge, build_judges
from .pipeline import RunResult, prepare_itemset, run_eval
from .plots import render, render_all
from .report import build_report

__all__ = [
    "Clause",
    "ClauseSet",
    "Completion",
    "ConfigError",
    "FakeClause",
    "METRICS",
    "Item",
    "ItemSet",
    "JudgeConfig",
    "ResultsStore",
    "RubricJudge",
    "RunContext",
    "RunResult",
    "ScoreRow",
    "Verdict",
    "build_itemset",
    "build_judges",
    "build_report",
    "load_config",
    "load_config_dict",
    "prepare_itemset",
    "register",
    "render",
    "render_all",
    "resolve",
    "run_eval",
    "check_comparable",
    "headline_table",
    "health_warnings",
    "paired_pressure",
    "per_clause_rates",
    "rates",
    "scatter_pairs",
    "validate",
]
