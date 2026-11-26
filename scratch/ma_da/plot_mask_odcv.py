# ABOUTME: MASK honesty and ODCV misalignment for the multi-agent difficult-advice 15% adapter beside the
# ABOUTME: control, DA 15% and DA+tools 15% (all seed 0); every bar read from the Hub at render time.
# Run: uv run python -m scratch.ma_da.plot_mask_odcv
"""Two panels in the paper style. Bars: mean over the benchmark's items with a 95% interval propagated from
each item's repeats (`_hub.mask`, `_hub.odcv`). The untuned base model is a dashed line."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.figures import _hub
from scratch.figures import _style as S

MADA_COLOUR = "#2e8b57"
ARMS = [
    ("control", S.COLOUR["control"], S.ARM["control"]),
    ("da", S.COLOUR["da"], "+ Difficult Advice 15% mix"),
    ("da_tools", S.COLOUR["da_tools"], "+ DA with tools 15% mix"),
    ("mada", MADA_COLOUR, "+ Multi-agent DA 15% mix"),
]
RUNS = {
    "mask": {
        "base": "2026-10-13-mask-qwen36",
        "control": "2026-11-03-mask-qwen36-0-plain",
        "da": "2026-11-03-mask-qwen36-0-da-15",
        "da_tools": "2026-11-09-mask-qwen36-0-da-tools-15",
        "mada": "2026-11-26-mask-qwen36-0-mada-15",
    },
    "odcv": {
        "base": "2026-10-13-odcv-qwen36",
        "control": "2026-10-06-odcv-qwen36-0-plain",
        "da": "2026-10-06-odcv-qwen36-0-da-15",
        "da_tools": "2026-10-11-odcv-qwen36-0-da-tools-15",
        "mada": "2026-11-26-odcv-qwen36-0-mada-15",
    },
}
PANELS = [
    (
        "mask",
        _hub.mask,
        "MASK: honesty\n(higher is better)",
        "Honest answers (%)",
        100,
        20,
    ),
    (
        "odcv",
        _hub.odcv,
        "ODCV: misalignment rate\n(lower is better)",
        "Misaligned rollouts (%)",
        60,
        10,
    ),
]


def main() -> None:
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0))
    md = ["# MASK and ODCV: multi-agent DA 15% beside the known arms", ""]
    artifacts = []
    for ax, (key, reader, title, ylabel, ymax, step) in zip(axes, PANELS):
        md += [f"## {key}", "", "| Arm | Value | 95% CI | Run |", "|---|---|---|---|"]
        for i, (arm, colour, label) in enumerate(ARMS):
            run = RUNS[key][arm]
            m, lo, hi = reader(run)
            S.bar(ax, i, m, lo, hi, colour=colour, label=f"{m:.1f}")
            md.append(f"| {label} | {m:.1f} | [{lo:.1f}, {hi:.1f}] | {run} |")
            artifacts.append(f"{_hub.ORG}/{run}")
        base = reader(RUNS[key]["base"])[0]
        S.base_line(ax, base)
        md += [
            f"| Qwen3.6-27B (dashed line) | {base:.1f} | | {RUNS[key]['base']} |",
            "",
        ]
        artifacts.append(f"{_hub.ORG}/{RUNS[key]['base']}")
        S.finish(
            ax,
            title=title,
            ylabel=ylabel,
            ylim=(0, ymax),
            yticks=range(0, ymax + 1, step),
        )
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
    handles = [S.base_handle()] + [S.patch(colour, label) for _, colour, label in ARMS]
    S.legend(fig, handles, ncol=3, y=1.0)
    fig.subplots_adjust(top=0.74, bottom=0.06, wspace=0.28)
    png = S.save(
        fig,
        "output/plots",
        "mask_odcv_mada_15",
        "MASK and ODCV: multi-agent DA 15% beside the known arms",
        "scratch/ma_da/plot_mask_odcv.py",
        artifacts=artifacts,
    )
    png.with_name(png.stem + "_results.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(png)
    print("\n".join(md))


if __name__ == "__main__":
    main()
