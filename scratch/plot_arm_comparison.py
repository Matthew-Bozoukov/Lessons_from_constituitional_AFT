# ABOUTME: Plot ODCV misalignment rate and severity for three difficult-advice ablation arms.
# ABOUTME: Writes a timestamped PNG plus a markdown mirror of the same numbers.

"""Compare the three arms scored on the identical 65 ODCV cells.

Two of the arms change HOW the difficult-advice traces reason (one-sided advocacy, excised
meta-cognition) and one changes WHICH scenarios are trained on (low-ODCV selection). Plotting
them together is the point: the reasoning-style arms sit together and the selection arm does
not.

Every arm is restricted to the scenario cells ALL of them cover, and its rate recomputed on
that set with the harness's own rule (median severity across judges per rollout, a violation
at >= 3). Without this the control would be plotted on the 70 cells its own run used while the
ablations sit on 65, and a cell-set difference would read as an effect. Error bars are the
judge-side 95% CIs from each run's results.json, which are computed on that run's own cells —
they are indicative of width, not of the restricted point estimate.

Run:
  uv run python scratch/plot_arm_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("output/odcv_bench")
ARMS = [
    ("no difficult advice", "qwen3_6-27b-lora-9284-numina-control-716-r64/combined4x_20260819_140227"),
    ("unmodified (control)", "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch/combined4x_20260814_230249"),
    ("no meta-cognition", "qwen3_6-27b-lora-t2-9284-synthdoc-716-c6excised-r64/combined3x_20260818_153146"),
    ("advocacy", "qwen3_6-27b-lora-t2-9284-synthdoc-716-advocacy-r64/combined3x_20260818_141118"),
    ("traits 1/3/4 removed", "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850"),
    ("traits 1/3 deleted", "qwen3_6-27b-lora-t2-9284-synthdoc-556-traits13-r64/combined3x_20260819_145013"),
    ("traits 5/6/7 only", "qwen3_6-27b-lora-9284-traits567-716-r64/combined3x_20260819_184957"),
    ("no clearance move", "qwen3_6-27b-lora-9284-no-clearance-716-r64/combined3x_20260819_202112"),
    ("traits 1/2/3 only", "qwen3_6-27b-lora-9284-traits123-716-r64/combined3x_20260819_221522"),
    ("low stakes", "qwen3_6-27b-lora-9284-low-stakes-712-r64/combined2x_20260820_164932"),
    ("low-ODCV selection", "qwen3_6-27b-lora-t2-9284-synthdoc-716-lowodcv-r64/combined3x_20260818_194628"),
]
STYLE = "#4C72B0"
ACCENT = "#C44E52"


def main(out_dir: str = "output/report") -> None:
    """Draw the comparison figure and write its markdown mirror.

    Args:
        out_dir: Directory for the PNG and markdown.
    """
    from src.utils import timestamp

    import statistics
    from src.eval.misalignment.odcv.odcv import misalignment_rate

    def rollouts(rel: str) -> dict[str, float]:
        """Median-across-judges severity per rollout key for one run."""
        per: dict[str, list[float]] = {}
        for f in (ROOT / rel / "evaluations").glob("scores_*.json"):
            for k, v in json.loads(f.read_text()).items():
                s = v.get("score") if isinstance(v, dict) else v
                if isinstance(s, (int, float)):
                    per.setdefault(k, []).append(s)
        return {k: statistics.median(v) for k, v in per.items()}

    scen = lambda k: k.rsplit("/rollout_", 1)[0]  # noqa: E731
    scored = {label: rollouts(rel) for label, rel in ARMS}
    common = set.intersection(*({scen(k) for k in v} for v in scored.values()))
    print(f"restricting every arm to the {len(common)} cells all of them cover")

    data = []
    for label, rel in ARMS:
        d = json.loads((ROOT / rel / "results.json").read_text())
        o = d["ours"]
        vals = {k: v for k, v in scored[label].items() if scen(k) in common}
        man = [v for k, v in vals.items() if k.startswith("mandated")]
        inc = [v for k, v in vals.items() if k.startswith("incentivized")]
        data.append({
            "label": label, "mr": round(misalignment_rate(list(vals.values())), 1),
            "ci": o["overall"]["mr_ci95"],
            "sev": round(statistics.mean(vals.values()), 2),
            "sev_ci": o["overall"]["severity_ci95"],
            "n": len(vals),
            "mandated": round(misalignment_rate(man), 1) if man else 0.0,
            "incentivized": round(misalignment_rate(inc), 1) if inc else 0.0,
        })

    ts = timestamp()
    fig, axes = plt.subplots(1, 3, figsize=(26, 6.2))
    x = range(len(data))
    labels = [d["label"] for d in data]
    def colour(label: str) -> str:
        """Red for the no-difficult-advice arm, grey for the unmodified control, blue else."""
        if label == "no difficult advice":
            return ACCENT
        if label == "unmodified (control)":
            return "#7F7F7F"
        return STYLE

    colors = [colour(d["label"]) for d in data]

    ax = axes[0]
    mr = [d["mr"] for d in data]
    err = [[d["mr"] - d["ci"][0] for d in data], [d["ci"][1] - d["mr"] for d in data]]
    ax.bar(x, mr, color=colors, width=0.6)
    ax.errorbar(x, mr, yerr=err, fmt="none", ecolor="#333333", capsize=6, lw=1.6)
    for i, d in enumerate(data):
        ax.text(i, d["ci"][1] + 1.0, f"{d['mr']}%", ha="center", fontweight="bold")
    ax.set_ylabel("misalignment rate (%)")
    ax.set_title(f"ODCV misalignment rate\n({len(common)} cells common to every arm)")
    ax.set_ylim(0, max(d["ci"][1] for d in data) + 6)

    ax = axes[1]
    sev = [d["sev"] for d in data]
    serr = [[d["sev"] - d["sev_ci"][0] for d in data], [d["sev_ci"][1] - d["sev"] for d in data]]
    ax.bar(x, sev, color=colors, width=0.6)
    ax.errorbar(x, sev, yerr=serr, fmt="none", ecolor="#333333", capsize=6, lw=1.6)
    for i, d in enumerate(data):
        ax.text(i, d["sev_ci"][1] + 0.04, f"{d['sev']}", ha="center", fontweight="bold")
    ax.set_ylabel("mean severity (0-3)")
    ax.set_title("Mean severity")
    ax.set_ylim(0, max(d["sev_ci"][1] for d in data) + 0.25)

    ax = axes[2]
    w = 0.36
    ax.bar([i - w / 2 for i in x], [d["mandated"] for d in data], width=w,
           color="#8C8C8C", label="mandated")
    ax.bar([i + w / 2 for i in x], [d["incentivized"] for d in data], width=w,
           color="#DD8452", label="incentivized")
    ax.set_ylabel("misalignment rate (%)")
    ax.set_title("By condition")
    ax.legend(frameon=False)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Difficult-advice ablations on ODCV-Bench (Qwen3.6-27B, r64)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    d_out = Path(out_dir)
    d_out.mkdir(parents=True, exist_ok=True)
    png = d_out / f"arm_comparison_{ts}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")

    md = [f"# ODCV ablation arms — {ts}", "",
          f"Every arm is restricted to the {len(common)} scenario cells ALL of them cover and",
          "its rate recomputed there, so no number reflects a different cell set. CI columns",
          "come from each run's own results.json and are indicative of width only.", "",
          f"![comparison]({png.name})", "",
          "| arm | manipulation | MR | CI95 | severity | mandated | incentivized | n |",
          "|---|---|--:|:--:|--:|--:|--:|--:|"]
    manip = {
        "no difficult advice": "716 slots hold extra NuminaMath-CoT instead",
        "unmodified (control)": "the recipe as-is; 716 difficult-advice traces untouched",
        "no meta-cognition": "reasoning rewritten to drop C6/C13/C14 meta-reasoning",
        "advocacy": "reasoning rewritten one-sided (no tradeoff framing)",
        "low-ODCV selection": "331 eval-like scenarios swapped out; reasoning untouched",
        "traits 1/3/4 removed": "oversight/honesty/harm traits dropped; reasoning untouched",
        "traits 1/3 deleted": "oversight + honesty deleted, not backfilled (556 rows)",
        "traits 5/6/7 only": "all 716 rows from character/identity/operator traits",
        "no clearance move": "every audit-then-clear trace swapped out (0 of 716)",
        "traits 1/2/3 only": "all 716 rows from oversight/power/honesty traits",
        "low stakes": "every scenario rewritten so almost nothing is at stake",
    }
    for d in data:
        md.append(f"| {d['label']} | {manip[d['label']]} | **{d['mr']}%** | "
                  f"[{d['ci'][0]}, {d['ci'][1]}] | {d['sev']} | {d['mandated']}% | "
                  f"{d['incentivized']}% | {d['n']} |")
    md += ["", "Published ODCV baseline for this model family: MR 37.2%, severity 1.43.", "",
           "The two reasoning-style arms sit together near 9%; the selection arm is roughly",
           "double that. Scenario-level sign test, low-ODCV vs advocacy: 16:0 discordant",
           "cells, p=0.000. Against c6excised (15:8) p=0.210, so the gap is solid against",
           "advocacy and not significant against the meta-cognition arm.", "",
           "CAVEAT on the low-ODCV arm: its scenario swap removed trait 1 almost entirely",
           "(80 -> 2) because t1 IS the agentic-oversight trait, so trait removal and",
           "eval-content removal are the same manipulation there. The traits-1/3/4 arm now",
           "running on ODCV is the disambiguation."]
    (d_out / f"arm_comparison_{ts}.md").write_text("\n".join(md) + "\n")

    print(f"wrote {png}")
    print(f"wrote {d_out / f'arm_comparison_{ts}.md'}")
    for d in data:
        print(f"  {d['label']:20s} MR {d['mr']:5.1f}%  CI {d['ci']}  sev {d['sev']}  n={d['n']}")


if __name__ == "__main__":
    fire.Fire(main)
