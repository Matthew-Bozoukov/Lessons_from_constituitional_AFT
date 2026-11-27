# ABOUTME: WhistleBench-Team for the multi-agent difficult-advice 15% adapter beside the control, DA 15% and
# ABOUTME: DA+tools 15%: went along with the teammate (cover-up and benign twin) and pushed back; read from the Hub.
# Run: uv run python -m scratch.ma_da.plot_whistlebench
"""Three panels in the paper style, bars grouped by who the teammate is. Each bar is k of n episodes with a
Wilson 95% interval (`_hub.wilson`). No nudge (boldness none), 50 episodes per cell."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.figures import _hub
from scratch.figures import _style as S

MADA_COLOUR = "#2e8b57"
ARMS = [
    (
        "control",
        S.COLOUR["control"],
        S.ARM["control"],
        "2026-09-24-wbt-qwen36-0-nosynth",
    ),
    (
        "da",
        S.COLOUR["da"],
        "+ Difficult Advice 15% mix",
        "2026-09-24-wbt-qwen36-0-da-15",
    ),
    (
        "da_tools",
        S.COLOUR["da_tools"],
        "+ DA with tools 15% mix",
        "2026-11-19-wbt-qwen36-0-da-tools-15",
    ),
    (
        "mada",
        MADA_COLOUR,
        "+ Multi-agent DA 15% mix",
        "2026-11-27-wbt-qwen36-0-mada-15",
    ),
]
TEAMS = [("ai_conceal", "AI teammate"), ("human_conceal", "Human teammate")]
PANELS = [
    (
        "misconduct",
        "complied_with_teammate",
        "Cover-up: went along\nwith the teammate (lower is better)",
    ),
    (
        "misconduct",
        "objected_to_teammate",
        "Cover-up: pushed back\non the teammate (higher is better)",
    ),
    (
        "benign",
        "complied_with_teammate",
        "Nothing wrong: went along\nwith the teammate",
    ),
]


def main() -> None:
    S.apply()
    results = {
        arm: _hub.fetch(run, "results/results.json")["cells"] for arm, _, _, run in ARMS
    }
    fig, axes = plt.subplots(1, 3, figsize=(22.0, 8.0))
    width = 0.2
    md = ["# WhistleBench-Team: multi-agent DA 15% beside the known arms", ""]
    for ax, (docs, metric, title) in zip(axes, PANELS):
        md += [
            f"## {docs} / {metric}",
            "",
            "| Team | Arm | k/n | % | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for g, (team, team_label) in enumerate(TEAMS):
            for i, (arm, colour, label, _) in enumerate(ARMS):
                cell = results[arm][f"{docs}/{team}"][metric]
                k, n = cell["k"], cell["n"]
                lo, hi = _hub.wilson(k, n)
                m = 100 * k / n
                x = g + (i - (len(ARMS) - 1) / 2) * width
                S.bar(
                    ax,
                    x,
                    m,
                    lo,
                    hi,
                    colour=colour,
                    width=width * 0.92,
                    label=f"{m:.0f}",
                    label_size=14,
                )
                md.append(
                    f"| {team_label} | {label} | {k}/{n} | {m:.0f} | [{lo:.0f}, {hi:.0f}] |"
                )
        md.append("")
        S.finish(
            ax,
            title=title,
            ylabel="Episodes (%)",
            ylim=(0, 110),
            yticks=range(0, 101, 20),
        )
        ax.set_xticks(range(len(TEAMS)))
        ax.set_xticklabels([t for _, t in TEAMS])
        ax.set_xlim(-0.55, len(TEAMS) - 0.45)
    handles = [S.patch(colour, label) for _, colour, label, _ in ARMS]
    S.legend(fig, handles, ncol=4, y=1.0)
    fig.subplots_adjust(top=0.78, bottom=0.08, wspace=0.26)
    artifacts = [f"{_hub.ORG}/{run}" for _, _, _, run in ARMS]
    png = S.save(
        fig,
        "output/plots",
        "whistlebench_mada_15",
        "WhistleBench-Team: multi-agent DA 15% beside the known arms",
        "scratch/ma_da/plot_whistlebench.py",
        artifacts=artifacts,
    )
    md += ["Runs: " + ", ".join(run for _, _, _, run in ARMS)]
    png.with_name(png.stem + "_results.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(png)


if __name__ == "__main__":
    main()
