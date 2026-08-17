# ABOUTME: Render the deliberation plots against PLACEHOLDER numbers to check layout before the
# ABOUTME: real run lands. Run: uv run python scratch/preview_plots.py  (writes to output/_preview)

"""Layout check only — every number here is invented.

Worth doing before the pod finishes: the plots are the deliverable, and debugging bar
collisions after a 3-hour GPU run is the expensive order to do it in.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from scratch.reports.plot_deliberation import ARMS, main

ROOT = Path("output/_preview")
rng = random.Random(3)


def _summary(name: str) -> dict:
    trace = {"think_chars_mean": round(rng.uniform(150, 900), 1),
             "empty_think_rate": 0.02, "n_calls": 838, "error_rate": 0.0,
             "truncation_rate": 0.0, "think_chars_median": 400.0,
             "answer_chars_mean": 20.0}
    if name == "llmbar":
        return {"adversarial_accuracy": round(rng.uniform(0.45, 0.85), 4),
                "consistency": {"rate": 0.8}, "n_items": 419, "parse_rate": 0.99,
                "trace": trace}
    if name == "debate_speeches":
        return {"kendall_tau_b": round(rng.uniform(0.05, 0.55), 4), "n_items": 300,
                "parse_rate": 0.98, "trace": trace}
    hold, fix = rng.uniform(0.55, 0.95), rng.uniform(0.2, 0.7)
    return {"balanced_accuracy": round((hold + fix) / 2, 4),
            "halves_measured": "both",
            "hold_rate_when_correct": {"rate": round(hold, 4)},
            "correction_rate_when_wrong": {"rate": round(fix, 4)},
            "n_items": 400, "parse_rate": 0.97, "trace_turn1": trace}


for eval_name in ("llmbar", "debate_speeches", "sycophancy"):
    for arm in ARMS:
        out = ROOT / eval_name / arm / "20260817_000000"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(_summary(eval_name), indent=2))

print(main(results=str(ROOT), out="output/_preview/report"))
