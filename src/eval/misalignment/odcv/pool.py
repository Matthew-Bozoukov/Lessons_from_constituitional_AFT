# ABOUTME: Pool the ODCV arms of one invocation into a single seed-inclusive result — the
# ABOUTME: step run_eval runs after every arm has been published, when more than one ran.

"""What `uv run evals --name odcv --target seed0 seed1 seed2` produces at the end.

Three seeds of one recipe are three draws from the same training pipeline, and the
question they exist to answer is about the RECIPE, not about seed 0. A single arm's
interval cannot answer it: with one checkpoint the bar is the spread of per-scenario
rates and says nothing about seed-to-seed variance (docs/error_bars.md). Two wrong ways
to combine them are easy to reach for -- averaging the three point estimates (which
throws the intervals away) and merging every rollout into one arm (which shrinks the bar
by pretending seed variance is rollout noise). Both claim more than the data supports.

The right one is already implemented: put each arm in as its own checkpoint and let
`src.eval.stats.interval` see `n_checkpoints >= 2`, at which point it infers
`checkpoints="sampled"` and carries T_A + T_B - T_C with Satterthwaite df -- an interval
that covers the seed you did not train.

Pooling is refused, loudly, when the arms did not answer the same question: a different
scenario set, or a different thinking mode, is a comparison the framework declines
elsewhere too and this is the same rule.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.eval.misalignment.odcv.odcv import VARIANTS, summarise_pooled


def _results(run: dict[str, Any]) -> dict:
    path = Path(run["out_dir"]) / "results" / "results.json"
    assert path.exists(), f"{run['target']}: no results.json at {path}"
    return json.loads(path.read_text())


def _scenario_sets(medians: dict) -> dict[str, frozenset]:
    return {v: frozenset(medians.get(v) or {}) for v in VARIANTS}


def pooled_key(model_keys: list[str]) -> str:
    """A name for the pooled arm, derived from what the arms have in common.

    `..._par716_s0`, `_s1`, `_s2` -> `..._par716-pooled3`. It goes in the repo id and the
    `model:` tag, so it has to read as one thing rather than as a list.
    """
    shared = os.path.commonprefix(model_keys)
    # Cut back to the last COMPLETE token: the raw prefix of `_s0` and `_s1` is `_s`,
    # which names nothing and would publish a repo called `..._s-pooled3`.
    if not all(key == shared or key[len(shared):len(shared) + 1] in ("-", "_")
               for key in model_keys):
        shared = re.sub(r"[-_][^-_]*$", "", shared)
    shared = shared.strip("-_")
    return f"{shared}-pooled{len(model_keys)}" if shared else f"pooled{len(model_keys)}"


def pool(runs: list[dict[str, Any]], cfg: Any, out_dir: Path) -> dict:
    """Pool this invocation's ODCV arms into one result over checkpoints.

    Args:
        runs: One entry per published arm, in the order they ran:
            `{"target", "model_key", "mode", "out_dir", "repo"}` (run_eval builds these).
        cfg: The eval config, for provenance only — pooling has no parameters of its own.
        out_dir: Where the pooled run dir is being assembled.

    Returns:
        `model_key`, `mode`, `pooled_from` (the arms and the repos they were published to,
        so the number can be traced back), the pooled metrics under `ours` — the same key a
        single arm uses, so every existing reader works — and each arm's medians under
        `per_scenario_medians_by_arm`, which is enough to recompute the pool. Deliberately
        NOT a merged `per_scenario_medians`: that would invite someone to re-summarise the
        arms as one, which is the mistake this module exists to avoid.

    Raises:
        AssertionError: If the arms ran different scenario sets or different modes — a
            pooled number over those would be an average of two different questions.
    """
    assert len(runs) >= 2, "pooling needs at least two arms"
    results = {run["model_key"]: _results(run) for run in runs}

    modes = {run["mode"] for run in runs}
    assert len(modes) == 1, (
        f"these arms ran in different thinking modes ({sorted(modes)}), which the "
        "framework refuses to pair anywhere else either — a pooled number over them "
        "would average two different questions")

    missing = [k for k, res in results.items() if not res.get("per_scenario_medians")]
    assert not missing, f"no per_scenario_medians in {missing} — nothing to pool"
    medians = {key: res["per_scenario_medians"] for key, res in results.items()}

    sets = {key: _scenario_sets(m) for key, m in medians.items()}
    first_key, first = next(iter(sets.items()))
    for key, other in sets.items():
        if other == first:
            continue
        detail = "; ".join(
            f"{v}: {len(first[v] - other[v])} only in {first_key}, "
            f"{len(other[v] - first[v])} only in {key}"
            for v in VARIANTS if first[v] != other[v])
        raise AssertionError(
            f"{key} did not run the same scenarios as {first_key} ({detail}). Pooling "
            "assumes the arms answered the same question; re-run the odd arm on the "
            "shared cell set, or pool the ones that match.")

    summary = summarise_pooled(medians)
    keys = [run["model_key"] for run in runs]
    return {
        "model_key": pooled_key(keys),
        "mode": runs[0]["mode"],
        "pooled_from": [
            {"target": run["target"], "model_key": run["model_key"], "repo": run.get("repo", ""),
             "mr_pct": results[run["model_key"]].get("ours", {}).get("overall", {}).get("mr_pct")}
            for run in runs
        ],
        "ours": summary,
        "per_scenario_medians_by_arm": medians,
    }
