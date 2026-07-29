# ABOUTME: The side-effect panel: capability regression ingested from an external harness,
# ABOUTME: plus reasoning retention. Over-refusal and persona drift are ordinary judged families.

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..core.store import RunContext, ScoreRow
from ..core.types import Completion

# Clause id used by rows that are not about any single clause. Filtered out of every
# clause-level plot by the analysis layer.
GLOBAL = "_global"


def capability_rows(cfg: dict[str, Any], ctx: RunContext) -> list[ScoreRow]:
    """Ingest capability-benchmark scores from an external harness.

    Capability evals (MMLU, GPQA, a coding subset) are deliberately NOT reimplemented
    here - the repo already runs them, and a second implementation would drift from the
    numbers everyone else quotes. This reads their output so the side-effect panel and
    the checkpoint trajectory can plot capability on the same axes as everything else.

    The expected file is a flat JSON mapping of benchmark name to accuracy in [0, 1],
    e.g. {"mmlu": 0.724, "gpqa": 0.318}.

    Args:
        cfg: Resolved run config.
        ctx: Run identity stamped onto every row.

    Returns:
        One row per benchmark; empty when no path is configured.

    Raises:
        ValueError: If the file exists but is not a flat mapping of names to numbers in
            [0, 1] - a silently mis-scaled capability number would look like a regression.
    """
    block = dict(cfg.get("side_effects") or {})
    path = block.get("capability_path")
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        if block.get("require_capability", False):
            raise ValueError(f"side_effects.capability_path {p} does not exist")
        return []

    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Capability file {p} must be a mapping of benchmark -> accuracy")

    rows: list[ScoreRow] = []
    for name, value in sorted(payload.items()):
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"Capability file {p} entry {name!r} is {value!r}; expected a number in [0, 1]. "
                f"Percentages must be divided by 100 before ingestion."
            )
        rows.append(
            ScoreRow(
                run_id=ctx.run_id,
                recipe=ctx.recipe,
                checkpoint_step=ctx.checkpoint_step,
                model_id=ctx.model_id,
                itemset_id=ctx.itemset_id,
                item_id=f"capability_{name}",
                clause_id=GLOBAL,
                clause_title=name,
                family="capability",
                difficulty="na",
                axis=f"capability_{name}",
                score=float(value),
                raw_score=float(value),
                passed=True,
                judge_model="external",
            )
        )
    return rows


def reasoning_rows(completions: dict[str, Completion], ctx: RunContext) -> list[ScoreRow]:
    """Emit one reasoning-retention row per completion.

    Qwen3's chat template wraps plain assistant text in an empty `<think></think>`, so
    naive SFT teaches the model to stop reasoning. This axis makes that collapse visible
    as a side effect at every checkpoint rather than as a surprise after training.

    Args:
        completions: item_id -> Completion.
        ctx: Run identity.

    Returns:
        One row per successful completion, scoring 1 when a non-empty trace was emitted.
    """
    return [
        ScoreRow(
            run_id=ctx.run_id,
            recipe=ctx.recipe,
            checkpoint_step=ctx.checkpoint_step,
            model_id=ctx.model_id,
            itemset_id=ctx.itemset_id,
            item_id=c.item_id,
            clause_id=GLOBAL,
            family="meta",
            difficulty="na",
            axis="reasoning_retained",
            score=1.0 if c.thinking.strip() else 0.0,
            raw_score=1.0 if c.thinking.strip() else 0.0,
            passed=bool(c.thinking.strip()),
            judge_model="derived",
        )
        for c in completions.values()
        if c.ok
    ]


def generation_health(completions: dict[str, Completion]) -> dict[str, Any]:
    """Summarise generation quality for the run manifest.

    Args:
        completions: item_id -> Completion.

    Returns:
        Counts and reasoning-length statistics. `truncated` matters most: a truncated
        answer is not a refusal, and a spike here invalidates a run rather than
        reporting a result about it.
    """
    total = len(completions)
    ok = [c for c in completions.values() if c.ok]
    errored = [c for c in completions.values() if c.error]
    truncated = [c for c in ok if c.finish_reason == "length"]
    think_chars = [len(c.thinking) for c in ok]
    return {
        "n_items": total,
        "n_ok": len(ok),
        "n_error": len(errored),
        "n_truncated": len(truncated),
        "error_rate": round(len(errored) / total, 4) if total else 0.0,
        "truncation_rate": round(len(truncated) / len(ok), 4) if ok else 0.0,
        "reasoning_retained_rate": round(
            sum(1 for c in ok if c.thinking.strip()) / len(ok), 4
        )
        if ok
        else 0.0,
        "thinking_chars_mean": round(statistics.mean(think_chars), 1) if think_chars else 0.0,
        "thinking_chars_median": round(statistics.median(think_chars), 1) if think_chars else 0.0,
        "first_errors": [
            {"item_id": c.item_id, "error": c.error[:200]} for c in errored[:5]
        ],
    }
