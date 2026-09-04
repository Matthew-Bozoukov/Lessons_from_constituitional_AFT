# ABOUTME: The treatment-minus-control contrast across the two arms of one condition —
# ABOUTME: run_eval calls this after both arms are published, because the contrast IS the result.

"""Two published arms of one condition -> the numbers the condition was run to get.

An arm on its own says nothing here: a pair's hospital advantage of 0.62 is only
meaningful against what the control pair captured on the SAME scenarios. So the contrast
is the result and a single arm is an ingredient, which is exactly what the framework's
pooling step is for.

Every contrast is paired on seed (`stats.arm_difference`): the same seed is the same
scenario in both blocks, so the seed's own generosity — some patient sets simply leave
more to capture — is variance the pairing removes rather than variance the interval
carries.

The within-arm contrast against the baseline condition (what the instruction and the
channel BUY that arm) is deliberately NOT computed here: the baseline runs in a separate
invocation with its own seed list, so that contrast is assembled afterwards over the run
directories (scratch/colosseum_hospital/analyse.py), together with the judge scores and
the run labels, which are post-hoc as well.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.misalignment.colosseum.hospital.harvest import MEASURES
from src.eval.misalignment.colosseum.stats import arm_difference

__all__ = ["pool"]


def _per_seed(run: dict) -> dict:
    """{measure: {cell: {seed: value}}} for one published arm."""
    path = Path(run["out_dir"]) / "results" / "per_seed.json"
    assert path.is_file(), f"{run['target']} published no {path}; it cannot be pooled"
    raw = json.loads(path.read_text())
    return {
        m: {
            c: {int(s): float(v) for s, v in seeds.items()}
            for c, seeds in cells.items()
        }
        for m, cells in raw.items()
    }


def pool(runs: list[dict], cfg, out_dir: Path) -> dict:
    """Contrast the two arms of one condition.

    Args:
        runs: `{"target", "model_key", "mode", "out_dir", "repo"}` per published arm.
        cfg: The eval config; `condition` and `peer` identify the cell and the control.
        out_dir: Where the pooled result is written.

    Returns:
        `model_key`, `mode`, `pooled_from` (run_eval needs those three) plus `contrasts`.
    """
    condition = str(cfg.condition)
    control_repo = str(cfg.peer)

    modes = {run["mode"] for run in runs}
    assert len(modes) == 1, (
        f"refusing to contrast arms served in different modes {sorted(modes)}: the mode is "
        "pinned into the chat template, so these arms did not face the same environment"
    )
    control = [r for r in runs if r["target"] == control_repo]
    treatment = [r for r in runs if r["target"] != control_repo]
    assert len(control) == 1 and len(treatment) == 1, (
        f"expected exactly one control arm ({control_repo}) and one treatment arm among "
        f"{[r['target'] for r in runs]}. The contrast is between two arms of one "
        "condition; a ladder of three is not a contrast."
    )
    control, treatment = control[0], treatment[0]
    c_measures, t_measures = _per_seed(control), _per_seed(treatment)

    contrasts: dict[str, dict] = {}
    for measure in MEASURES:
        t_cell = t_measures.get(measure, {}).get(condition, {})
        c_cell = c_measures.get(measure, {}).get(condition, {})
        if len(set(t_cell) & set(c_cell)) < 2:
            continue
        contrasts[measure] = arm_difference(
            t_cell, c_cell, label=f"{MEASURES[measure]} ({condition})"
        )

    summary = {
        "condition": condition,
        "model_key": f"{condition}-contrast",
        "mode": runs[0]["mode"],
        "control": control["target"],
        "treatment": treatment["target"],
        "contrasts": contrasts,
        "pooled_from": [
            {
                "target": r["target"],
                "model_key": r["model_key"],
                "repo": r.get("repo", ""),
            }
            for r in runs
        ],
    }
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "results" / "contrasts.json").write_text(json.dumps(summary, indent=2))
    return summary
