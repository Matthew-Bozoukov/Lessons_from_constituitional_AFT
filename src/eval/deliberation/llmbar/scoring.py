# ABOUTME: Parse "Output (a)/(b)", then score LLMBar over both presentation orders: accuracy,
# ABOUTME: position consistency and positional bias, overall and per subset.

"""Scoring for the `llmbar` eval.

Three numbers, and the third is the one a fluent-but-shallow judge fails:

- **accuracy** — did it pick the output that follows the instruction, averaged over both
  presentation orders. Upstream's headline.
- **consistency** — did the two orders name the SAME underlying output. Needs no gold
  label at all, so it is immune to a model that has memorised LLMBar, and it is the
  order-flip metric `docs/in_domain_evals.md` specifies for courtroom, measured here for
  free.
- **first_position_rate** — how often it picked whatever was shown first. 0.5 is unbiased;
  far from 0.5 means the "judgment" is substantially positional.

`consistent_accuracy` combines the first two: correct in both orders. A model that is right
by coin-flip scores near 0.25 on it while scoring 0.5 on plain accuracy, which is exactly
the distinction between a judgment and a preference for whatever came first.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.utils import wilson

# Upstream's own parse, adapted: it anchors `Output \(a\)` to line start. Reasoning models
# answer after a trace and often add a trailing sentence, so the anchor is relaxed to a
# word boundary and the LAST match wins — a model that weighs "Output (a) says X…" before
# committing must be scored on the commitment, not the first mention.
_CHOICE = re.compile(r"output\s*\(\s*([ab])\s*\)", re.IGNORECASE)
# Fallback for a bare "(a)" / "a" reply, which small models emit despite the instruction.
_BARE = re.compile(r"^\s*\(?\s*([ab])\s*\)?\s*\.?\s*$", re.IGNORECASE)
# Upstream steers the evaluator away from these with logit bias, which vLLM cannot do (see
# assets/README.md). Detected instead, so a refusal to choose is visible as a refusal.
_HEDGE = re.compile(r"\b(both|neither|equally|tie|same quality)\b", re.IGNORECASE)


def parse_choice(answer: str) -> str:
    """Return "a", "b", or "" when the reply names neither output."""
    text = answer or ""
    matches = _CHOICE.findall(text)
    if matches:
        return matches[-1].lower()
    bare = _BARE.match(text.strip())
    return bare.group(1).lower() if bare else ""


def is_hedge(answer: str) -> bool:
    """True when an unparseable reply looks like a refusal to choose rather than noise."""
    return bool(_HEDGE.search(answer or ""))


def _rate(hits: int, total: int) -> dict:
    low, high = wilson(hits, total)
    return {"rate": round(hits / total, 4) if total else 0.0,
            "n": total, "hits": hits, "ci95": [low, high]}


def _group(records: list[dict]) -> dict:
    """Metrics for one group of items. Each record holds both orders' outcomes."""
    orders = [(r, order) for r in records for order in ("normal", "swapped")]
    judged = [(r, o) for r, o in orders if r[o]["choice"]]
    correct = sum(r[o]["choice"] == r[o]["expected"] for r, o in judged)

    both_parsed = [r for r in records
                   if r["normal"]["choice"] and r["swapped"]["choice"]]
    consistent = sum(r["normal"]["output"] == r["swapped"]["output"] for r in both_parsed)
    both_correct = sum(
        r["normal"]["choice"] == r["normal"]["expected"]
        and r["swapped"]["choice"] == r["swapped"]["expected"] for r in both_parsed)
    first = sum(r[o]["choice"] == "a" for r, o in judged)

    return {
        "accuracy": _rate(correct, len(judged)),
        "consistency": _rate(consistent, len(both_parsed)),
        "consistent_accuracy": _rate(both_correct, len(both_parsed)),
        "first_position_rate": round(first / len(judged), 4) if judged else 0.0,
        "parse_rate": round(len(judged) / len(orders), 4) if orders else 0.0,
        "hedge_rate": round(
            sum(r[o]["hedged"] for r, o in orders if not r[o]["choice"]) / max(len(orders), 1), 4),
        "n_items": len(records),
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate scored records overall and per subset.

    Args:
        records: One per item, each with `subset` and a `normal` and `swapped` dict
            holding `choice`, `expected`, `output` and `hedged`.

    Returns:
        Overall metrics plus a `by_subset` block. The split is not decoration: `Natural` and
        the four adversarial subsets measure different things, and a pooled number lets a
        strong `Natural` score hide the adversarial collapse the eval exists to detect.
    """
    summary = _group(records)
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_subset[record["subset"]].append(record)
    summary["by_subset"] = {
        name: {
            "n": len(rows),
            "accuracy": (group := _group(rows))["accuracy"]["rate"],
            "consistency": group["consistency"]["rate"],
            "consistent_accuracy": group["consistent_accuracy"]["rate"],
            "first_position_rate": group["first_position_rate"],
        }
        for name, rows in sorted(by_subset.items())
    }
    adversarial = [r for r in records if r["subset"].startswith("Adversarial")]
    if adversarial:
        # The headline for PC: the adversarial subsets are the ones where the output that
        # does NOT follow the instruction is the better-looking one.
        summary["adversarial_accuracy"] = _group(adversarial)["accuracy"]["rate"]
    return summary
