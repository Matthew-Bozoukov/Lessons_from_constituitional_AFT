# ABOUTME: Bar plot of the secret-number eval reward-hacking rate for baseline
# ABOUTME: gpt-oss-120b vs the 5% difficult-advice fine-tune.

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

MODELS = ["gpt-oss-120b\n(baseline)", "gpt-oss-120b\n(diff-advice 5%)"]
RATES = [48.0, 21.0]
COLORS = ["#4C72B0", "#DD8452"]

OUT_DIR = Path("output/plots")


def main() -> None:
    """Render and save the secret number eval reward-hacking-rate bar plot."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    bars = ax.bar(
        MODELS,
        RATES,
        color=COLORS,
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
        zorder=3,
    )

    for bar, rate in zip(bars, RATES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.5,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    ax.set_title("Secret Number Eval", fontsize=20, fontweight="bold", pad=14)
    ax.set_ylabel("Reward hacking rate (%)", fontsize=16)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"secret_number_eval_{ts}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
