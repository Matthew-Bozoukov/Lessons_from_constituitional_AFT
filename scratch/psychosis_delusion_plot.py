# ABOUTME: Bar chart of the psychosis eval's delusion-confirmation rating, per persona and
# ABOUTME: overall, for the difficult-advice arm against the no-synthetic control.

"""Delusion-confirmation scores from the psychosis eval.

Scale is 0-4 and LOWER IS BETTER: it rates how much the assistant confirms the user's
delusional frame. A rating of -1 means "no delusion expressed on this turn yet" and is
excluded from every mean rather than counted as a zero (metrics.py does the same) — a turn
with nothing to confirm is not evidence of restraint.

The two runs do NOT cover the same personas: three of the control's eight arcs errored out,
so it has five. The overall comparison is therefore computed on the five personas BOTH arms
completed; using each arm's own full set would compare different exams.

    uv run python scratch/psychosis_delusion_plot.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("standard SFT data\n(0% synthetic)", "table2-only", "#d1603d"),
        ("+ difficult advice\n(20% synthetic)", "table2-synthdoc-2203", "#3b7dd8")]


def _by_persona(arm_dir: str) -> dict[str, list[int]]:
    """Per-persona delusion-confirmation ratings, with the -1 'no delusion yet' turns dropped."""
    out: dict[str, list[int]] = {}
    for line in Path(f"output/psychosis/{arm_dir}/grades.jsonl").open():
        r = json.loads(line)
        rating = int(r["delusion_confirmation_rating"])
        if rating < 0:
            continue
        out.setdefault(r["character"], []).append(rating)
    return out


def _mean_se(vals: list[int]) -> tuple[float, float]:
    """Mean and standard error of a rating list."""
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return m, se


def main(out: str = "output/plots/psychosis_delusion_confirmation.png") -> None:
    """Draw the figure.

    Args:
        out: Destination PNG.
    """
    data = {d: _by_persona(d) for _, d, _ in ARMS}
    shared = sorted(set(data["table2-only"]) & set(data["table2-synthdoc-2203"]))

    fig, ax0 = plt.subplots(figsize=(7.0, 6.2))

    # Overall, on the shared personas only.
    for i, (label, d, colour) in enumerate(ARMS):
        vals = [v for p in shared for v in data[d][p]]
        m, se = _mean_se(vals)
        ax0.bar(i, m, 0.55, yerr=se, color=colour, edgecolor="black", linewidth=0.8,
                capsize=7, error_kw={"elinewidth": 1.4})
        ax0.text(i, m + se + 0.08, f"{m:.2f}", ha="center", fontsize=16, weight="bold")
    ax0.set_xticks(range(len(ARMS)))
    ax0.set_xticklabels([a[0] for a in ARMS], fontsize=14)
    ax0.set_ylabel("delusion-confirmation rating (0-4)", fontsize=15)
    ax0.set_ylim(0, 4)
    ax0.tick_params(labelsize=14)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.2)

    ax0.set_title("AI psychosis: delusion confirmation\nlower is better", fontsize=16)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f">>> {out}")
    for label, d, _ in ARMS:
        vals = [v for p in shared for v in data[d][p]]
        m, se = _mean_se(vals)
        print(f"    {label.replace(chr(10), ' '):<34} {m:.3f} +/- {se:.3f}  (n={len(vals)} turns)")
    print(f"    shared personas: {', '.join(shared)}")
    only = sorted(set(data['table2-synthdoc-2203']) - set(data['table2-only']))
    print(f"    excluded (control has no data): {', '.join(only)}")


if __name__ == "__main__":
    fire.Fire(main)
