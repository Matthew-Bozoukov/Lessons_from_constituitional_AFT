# ABOUTME: Reads per-scenario `concerning` scores out of tenet-audit .eval logs, keeping only seeds
# ABOUTME: whose target actually answered, so a dead endpoint never reads as a well-behaved model.
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SEEDS = REPO / "src/eval/audits/petri/petri-subscription/seeds-constitution"


def sample_concerning(s) -> float | None:
    """One sample's `concerning` score, or None when it was not scored."""
    for sc in (s.scores or {}).values():
        v = sc.value
        if isinstance(v, dict) and "concerning" in v:
            return float(v["concerning"])
    return None


def valid_scores(logs: list[Path]) -> tuple[dict[str, float], list[Path]]:
    """Scenario id -> `concerning`, from the newest log in which that seed reached its target.

    A seed whose LAST target call failed is skipped: its endpoint was dead by the end (a stale
    tunnel, a retry pointed at the old port, a tunnel that died mid-transcript), and Petri's
    judge scores the auditor talking to nothing as `concerning = 1`, which reads as a
    well-behaved model (2026-10-09 da-15). An earlier failed call that a later one recovered
    from is ordinary retry noise. A seed whose
    auditor never calls the target at all (P4-08 does this in every arm) is kept: that is the
    seed's own behaviour. Logs are read oldest-first by name, whose prefix is the run's start
    time, so a rerun of some seeds overrides only those seeds.

    Returns:
        The scores and the logs that contributed at least one of them.
    """
    from inspect_ai.log import read_eval_log_samples

    scores: dict[str, float] = {}
    source: dict[str, Path] = {}
    for log in sorted(logs, key=lambda p: p.name):
        for s in read_eval_log_samples(str(log), all_samples_required=False):
            v = sample_concerning(s)
            calls = [e for e in s.events if e.event == "model" and e.role == "target"]
            if v is None or (calls and calls[-1].error):
                continue
            scores[str(s.id)], source[str(s.id)] = v, log
    return scores, sorted(set(source.values()), key=lambda p: p.name)
