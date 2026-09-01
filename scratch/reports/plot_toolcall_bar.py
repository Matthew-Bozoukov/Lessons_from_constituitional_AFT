# ABOUTME: Bar plot comparing misalignment rate for the 20/80 arm with and without tool calls.
# ABOUTME: Writes output/plots/toolcall_vs_no_toolcall_<ts>.png plus a markdown mirror.

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from src.utils import figure_path

# Measured misalignment rates for the two 20/80 conditions.
BARS = [
    ("20/80\nwith tool call", 10.2, "#2e86de"),
    ("20/80\nwithout tool call", 19.2, "#e17055"),
]
OUT_DIR = Path("output/plots")


def main() -> None:
    """Draw the two-bar comparison and write the figure plus its markdown mirror."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [b[0] for b in BARS]
    values = [b[1] for b in BARS]
    colors = [b[2] for b in BARS]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8,
                  width=0.55, zorder=3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=16, fontweight="bold")

    # Annotate the gap, which is the point of the comparison.
    hi, lo = max(values), min(values)
    ax.annotate("", xy=(0, lo), xytext=(0, hi),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2))
    ax.text(0.06, (hi + lo) / 2, f"-{hi - lo:.1f} pp\n({100 * (hi - lo) / hi:.0f}% lower)",
            fontsize=14, va="center", ha="left")

    ax.set_ylabel("Misalignment rate (%)", fontsize=16)
    ax.set_xlabel("Condition", fontsize=16)
    ax.set_title("Misalignment rate: 20/80 arm with vs without tool calls\n"
                 "Qwen3.6-27B", fontsize=16, pad=14)
    ax.set_ylim(0, max(values) * 1.35)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    ts = time.strftime("%Y%m%d_%H%M%S")
    png = figure_path(OUT_DIR, "toolcall_vs_no_toolcall")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    md = OUT_DIR / f"toolcall_vs_no_toolcall_{ts}.md"
    md.write_text(
        "# Misalignment rate: 20/80 arm, with vs without tool calls\n\n"
        f"![bar plot]({png.name})\n\n"
        "| Condition | Misalignment rate |\n|---|---|\n"
        f"| 20/80 with tool call | **{values[0]:.1f}%** |\n"
        f"| 20/80 without tool call | **{values[1]:.1f}%** |\n"
        f"| **Difference** | **{hi - lo:.1f} pp** ({100 * (hi - lo) / hi:.0f}% lower) |\n\n"
        "Model: Qwen3.6-27B. Rates supplied directly; no confidence intervals were given, so\n"
        "none are drawn -- the bars show point estimates only.\n"
    )
    print(f">>> {png}")
    print(f">>> {md}")


if __name__ == "__main__":
    main()
