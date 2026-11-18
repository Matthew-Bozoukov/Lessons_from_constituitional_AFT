# ABOUTME: Petri aggregate over the 19 target-misbehaviour judge dimensions (higher = worse) for Figure 2's arms:
# ABOUTME: judge points above the floor per audit, and % of audits with any of the 19 at >= 4; Hub Inspect logs only.
"""Reads every audit's per-dimension judge scores from each run's raw Inspect logs on the Hub
(metadata/inspect/*.eval; results/ publishes only `concerning`) and caches them under
output/petri/dims_cache/<run>.json.

Left panel: per audit, the sum over the 19 dimensions of (score - 1), i.e. judge points above the
floor of 1 ("not at all"); the bar is the mean over the 90 audits. Right panel: % of audits in
which at least one of the 19 dimensions scored 4 or more (the rubric's "mild" anchor).

Arms and runs are Figure 2's (scratch/figures/fig2_six_panel_seed_variance.py RUNS["petri"]). A
three-adapter arm is the plain mean of its adapters with a t over the three adapter values on the
log-odds scale (src.eval.stats.logit_t), as in Figure 2; a one-adapter arm has no interval.

Run: uv run python scratch/petri_dims/fig_petri_misbehaviour.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.stats import logit_t  # noqa: E402
from src.utils import save_figure_pdf  # noqa: E402

ORG = "dougalldeepmind"
CACHE = Path("output/petri/dims_cache")
OUT = Path("output/petri")

# Petri's own descriptions name these as target misbehaviour where a higher score is worse
# (inspect-petri 3.1.0, _judge/dimensions/*.md; tags safety, deception, bias, autonomy, compliance).
MISBEHAVIOUR = [
    "concerning",
    "concerning_with_no_mitigating_factors",
    "disappointing",
    "needs_attention",
    "cooperation_with_harmful_sysprompt",
    "cooperation_with_human_misuse",
    "counterfactual_bad_behavior",
    "facilitating_terrorism",
    "unfaithful_thinking",
    "unprompted_deception_toward_user",
    "unprompted_emotional_manipulativeness",
    "unprompted_sycophancy",
    "unprompted_encouragement_of_user_delusion",
    "unprompted_self_serving_bias",
    "manipulated_by_developer",
    "unprompted_blackmail",
    "unprompted_leaking",
    "unprompted_self_preservation",
    "user_over_sysprompt",
]
ARMS = [
    (
        "control",
        "+ Control (Standard SFT)",
        "#5a6472",
        ["2026-11-17-petri-qwen36-0-plain"],
    ),
    (
        "plain_const",
        "+ Control + constitution in system prompt",
        "#0f8b8d",
        ["2026-11-04-petri-constitution-qwen36-0-plain"],
    ),
    (
        "delib",
        "+ Deliberative Alignment 15% mix",
        "#2196f3",
        [
            "2026-11-04-petri-qwen36-0-delib-sonnet-15-a",
            "2026-11-05-petri-qwen36-0-delib-sonnet-15-b",
            "2026-11-05-petri-qwen36-0-delib-sonnet-15-c",
        ],
    ),
    (
        "da",
        "+ Difficult Advice 15% mix",
        "#7e22ce",
        [
            "2026-11-05-petri-qwen36-0-da-15",
            "2026-11-05-petri-qwen36-0-da-15-b",
            "2026-11-05-petri-qwen36-0-da-15-c",
        ],
    ),
    (
        "da_tools",
        "+ Difficult Advice + tool definitions 15% mix",
        "#d0558f",
        [
            "2026-11-09-petri-qwen36-0-da-tools-15",
            "2026-11-06-petri-qwen36-0-da-tools-15-b",
            "2026-11-06-petri-qwen36-0-da-tools-15-c",
        ],
    ),
]
BASE = "2026-11-05-petri-qwen36"
BASE_COLOUR = "#c1121f"


def audits(repo: str) -> list[dict]:
    """Every audit's {dimension: score} for one run, cached. Logs read oldest first, so a retry or resume
    log's score replaces the first attempt's for the same seed."""
    p = CACHE / f"{repo}.json"
    if not p.exists():
        from huggingface_hub import HfApi, hf_hub_download
        from inspect_ai.log import read_eval_log

        rid = f"{ORG}/{repo}"
        logs = sorted(
            f
            for f in HfApi().list_repo_files(rid, repo_type="dataset")
            if f.startswith("metadata/inspect/") and f.endswith(".eval")
        )
        per: dict[str, dict] = {}
        for f in logs:
            for smp in (
                read_eval_log(hf_hub_download(rid, f, repo_type="dataset")).samples
                or []
            ):
                for sc in (smp.scores or {}).values():
                    if isinstance(sc.value, dict) and "concerning" in sc.value:
                        per[str(smp.id)] = {k: float(v) for k, v in sc.value.items()}
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(per))
    per = json.loads(p.read_text())
    assert len(per) == 90, f"{repo}: {len(per)} scored audits, expected 90"
    return list(per.values())


def points(a: dict) -> float:
    return sum(a[d] - 1 for d in MISBEHAVIOUR)


def flagged(a: dict) -> bool:
    return any(a[d] >= 4 for d in MISBEHAVIOUR)


def run_values(repo: str) -> tuple[float, float]:
    aa = audits(repo)
    return sum(points(a) for a in aa) / len(aa), 100 * sum(
        flagged(a) for a in aa
    ) / len(aa)


def arm_bar(
    vals: list[float], scale: float
) -> tuple[float, float | None, float | None]:
    if len(vals) == 1:
        return vals[0], None, None
    t = logit_t(vals, 90, scale=scale)
    return sum(vals) / len(vals), t["lo"], t["hi"]


def main() -> None:
    top_points = 9.0 * len(MISBEHAVIOUR)
    rows = []
    for key, label, colour, repos in ARMS:
        per_run = [run_values(r) for r in repos]
        rows.append(
            (
                label,
                colour,
                arm_bar([v[0] for v in per_run], top_points),
                arm_bar([v[1] for v in per_run], 100.0),
                per_run,
            )
        )
    base_pts, base_pct = run_values(BASE)

    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    panels = [
        (
            axes[0],
            2,
            base_pts,
            "Judge points above floor per audit\n(sum over 19 dimensions of score − 1), lower is better",
            "{:.2f}",
            "Severity",
        ),
        (
            axes[1],
            3,
            base_pct,
            "Audits with any of the 19 at ≥ 4 (%), lower is better",
            "{:.1f}",
            "Share of audits flagged",
        ),
    ]
    for ax, col, base, ylabel, fmt, title in panels:
        ymax = 0.0
        for i, row in enumerate(rows):
            m, lo, hi = row[col]
            ax.bar(i, m, 0.68, color=row[1], edgecolor="black", linewidth=0.8, zorder=3)
            top = m
            if lo is not None:
                ax.errorbar(
                    i,
                    m,
                    yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
                    fmt="none",
                    ecolor="#2b2523",
                    elinewidth=1.3,
                    capsize=5,
                    zorder=4,
                )
                top = hi
            ymax = max(ymax, top, base)
            ax.text(
                i,
                top,
                fmt.format(m),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                zorder=5,
            )
        ax.axhline(
            base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2
        )
        ax.set_ylim(0, ymax * 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=BASE_COLOUR,
            linestyle=(0, (2.5, 2.5)),
            linewidth=1.8,
            label="Qwen3.6-27B",
        )
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=r[1], edgecolor="black", label=r[0])
        for r in rows
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Petri audit: 19 judge dimensions of target misbehaviour (higher = worse), 90 audits per adapter"
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_petri_misbehaviour_19dims"
    fig.savefig(f"{stem}.png", dpi=150)
    artifacts = [f"{ORG}/{r}" for _, _, _, repos in ARMS for r in repos] + [
        f"{ORG}/{BASE}"
    ]
    save_figure_pdf(
        fig,
        f"{stem}.pdf",
        "Petri: 19 misbehaviour dimensions",
        artifacts,
        "scratch/petri_dims/fig_petri_misbehaviour.py",
    )
    md = [
        f"# {stem.name}",
        "",
        "| Arm | Points above floor per audit | 95% | Audits flagged (%) | 95% | Per adapter (points, %) |",
        "|---|---|---|---|---|---|",
    ]
    for label, _, p, f, per_run in rows:
        ci = lambda b: f"[{b[1]:.2f}, {b[2]:.2f}]" if b[1] is not None else "none"  # noqa: E731
        md.append(
            f"| {label} | {p[0]:.3f} | {ci(p)} | {f[0]:.1f} | {ci(f)} | "
            + ", ".join(f"({a:.3f}, {b:.1f})" for a, b in per_run)
            + " |"
        )
    md.append(f"| Qwen3.6-27B (base line) | {base_pts:.3f} | | {base_pct:.1f} | | |")
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(stem)


if __name__ == "__main__":
    main()
