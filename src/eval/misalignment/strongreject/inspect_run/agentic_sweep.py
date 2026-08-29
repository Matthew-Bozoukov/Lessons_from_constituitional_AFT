# ABOUTME: Run the Inspect agentic_misalignment 27-condition sweep against a served model,
# ABOUTME: prod=True (no scratchpad), and write per-condition harmfulness plus the aggregate.

"""Inspect agentic_misalignment, the standard 27-condition sweep, one served model.

The Inspect task takes ONE condition per call. The paper's sweep is 3 scenarios x 9 conditions:
a no-goal-conflict baseline plus the 8 goal-value conditions (the four goal axes x two sides).
27 = 3 x 9. Each condition runs `epochs` samples (1 here), scored by the harmfulness grader.

`prod=True` on every task removes the secret-scratchpad instruction, per the request to run
without the scratchpad. The target is reached exactly as StrongREJECT reaches it — the
openai-api/podvllm provider pointed at a vLLM-served adapter — since inspect-ai cannot share
this repo's env (the click pin) and so lives in this nested project.

    uv run python agentic_sweep.py --model openai-api/podvllm/<served> --log_dir logs/am-<seed>
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
from inspect_ai import eval as inspect_eval
from evals.agentic_misalignment.agentic_misalignment import agentic_misalignment

# The MSM 3-action set (arXiv:2605.02087): exfiltration REPLACES blackmail (Qwen rarely
# blackmails), leaking IS the paper's "espionage". prod stays True (reasoning model, per
# the paper), model_name "Qwen", temperature 0.7 — the paper's AM settings.
SCENARIOS = ["exfiltration", "leaking", "murder"]
# The four goal axes, each a (goal_value) pair forming the two sides of one conflict.
GOAL_PAIRS = [("america", "global"), ("safety", "acceleration"),
              ("ethical", "pragmatic"), ("individualism", "collectivism")]


def conditions() -> list[dict]:
    """The 27 (scenario x 9) conditions: one no-conflict baseline + 8 goal-value conditions."""
    out = []
    for sc in SCENARIOS:
        # Without goal conflict: no goal stated, self-preservation pressure only.
        out.append({"scenario": sc, "goal_type": "none", "goal_value": "none",
                    "urgency_type": "replacement", "conflict": False})
        # With goal conflict: the model is given an explicit goal, one per side of each axis.
        for a, b in GOAL_PAIRS:
            for gv in (a, b):
                out.append({"scenario": sc, "goal_type": "explicit", "goal_value": gv,
                            "urgency_type": "replacement", "conflict": True})
    return out


def main(model: str, log_dir: str, grader_model: str = "openrouter/x-ai/grok-4.20",
         epochs: int = 1, max_connections: int = 16, max_tokens: int = 8192,
         model_name: str = "Qwen", temperature: float = 0.7) -> None:
    """Run the 27-condition sweep and write a summary next to the Inspect logs.

    Args:
        model: openai-api/podvllm/<served-name> (or any Inspect model spec).
        log_dir: Inspect log directory for this run (one per seed).
        grader_model: Model that scores harmfulness (not the target family).
        epochs: Samples per condition (1 = one draw per condition).
        max_connections: Concurrent target calls.
        max_tokens: Target generation cap.
    """
    conds = conditions()
    assert len(conds) == 27, f"expected 27 conditions, built {len(conds)}"
    tasks = [agentic_misalignment(
        scenario=c["scenario"], goal_type=c["goal_type"], goal_value=c["goal_value"],
        urgency_type=c["urgency_type"], prod=True, grader_model=grader_model,
        model_name=model_name)
        for c in conds]

    logs = inspect_eval(tasks, model=model, log_dir=log_dir, epochs=epochs,
                        max_connections=max_connections, temperature=temperature,
                        max_tokens=max_tokens, retry_on_error=2)

    rows = []
    for c, log in zip(conds, logs):
        rec = {"condition": f"{c['scenario']}_{c['goal_type']}-{c['goal_value']}",
               "scenario": c["scenario"], "conflict": c["conflict"],
               "status": log.status, "n": None, "harmful": None, "harmful_rate": None}
        if log.status == "success" and log.samples:
            harmful = [1 if _is_harmful(s) else 0 for s in log.samples]
            rec["n"] = len(harmful)
            rec["harmful"] = sum(harmful)
            rec["harmful_rate"] = round(100.0 * sum(harmful) / len(harmful), 2)
        rows.append(rec)

    scored = [r for r in rows if r["harmful_rate"] is not None]
    agg = (round(sum(r["harmful"] for r in scored) / sum(r["n"] for r in scored) * 100, 2)
           if scored else None)
    summary = {"model": model, "n_conditions": len(rows), "n_scored": len(scored),
               "aggregate_harmful_rate_pct": agg,
               "conflict_rate_pct": _subset_rate(scored, True),
               "no_conflict_rate_pct": _subset_rate(scored, False),
               "by_scenario": {sc: _subset_rate([r for r in scored if r["scenario"] == sc])
                               for sc in SCENARIOS},
               "conditions": rows}
    out = Path(log_dir) / "sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))
    print(f">>> wrote {out}")


def _is_harmful(sample) -> bool:
    """A sample is harmful if its harmfulness scorer flagged it.

    The Inspect harmfulness_scorer returns a DICT value, e.g.
    {"harmful": 1.0, "classifier_verdict": 1.0} — reading `.value` as a scalar always
    yields False and reports a spurious 0% (the 2026-08-29 bug). Read the `harmful` key
    when the value is a dict; fall back to scalar/string for other scorers.
    """
    for sc in (sample.scores or {}).values():
        v = sc.value
        if isinstance(v, dict):
            h = v.get("classifier_verdict", v.get("harmful"))
            if h is not None:
                return float(h) > 0
        if isinstance(v, (int, float)):
            return v > 0
        if isinstance(v, str):
            return v.upper() in ("C", "CORRECT", "YES", "TRUE", "1", "HARMFUL")
    return False


def _subset_rate(rows: list[dict], conflict: bool | None = None) -> float | None:
    """Aggregate harmful rate over a subset of conditions (optionally by conflict flag)."""
    sub = rows if conflict is None else [r for r in rows if r["conflict"] == conflict]
    n = sum(r["n"] for r in sub)
    return round(100.0 * sum(r["harmful"] for r in sub) / n, 2) if n else None


if __name__ == "__main__":
    fire.Fire(main)
