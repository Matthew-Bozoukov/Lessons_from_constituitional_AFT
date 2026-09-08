# ABOUTME: Bar plots for the matched ODCV arms: overall MR/severity, and per-scenario
# ABOUTME: severity for base vs fine-tune side by side.

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD, misalignment_rate  # noqa: E402
from src.naming import figure_path

BASE_C, FT_C = "#F58518", "#4C78A8"
PUBLISHED_MR = 43.8


def _cells(path: Path) -> dict[str, float]:
    """Flatten an arm's results.json into {"variant/scenario": median severity}."""
    med = json.loads(path.read_text())["per_scenario_medians"]
    return {f"{v}/{s}": sc for v in VARIANTS for s, sc in med[v].items()}


def main(
    treatment: str,
    control: str,
    out: str = "output/odcv_bench/comparison/plots",
) -> None:
    """Write overall and per-scenario bar plots comparing the two arms.

    Args:
        treatment: results.json for the fine-tune arm.
        control: results.json for the base arm.
        out: Directory to write PNGs into.
    """
    t, c = _cells(Path(treatment)), _cells(Path(control))
    shared = sorted(set(t) & set(c))
    assert shared, "arms share no cells"
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- overall ----------
    tv, cv = [t[k] for k in shared], [c[k] for k in shared]
    groups = ["Overall", "Mandated", "Incentivized"]
    def subset(cells, variant):
        return [cells[k] for k in shared if variant is None or k.startswith(variant + "/")]
    base_mr = [misalignment_rate(subset(c, v)) for v in (None, "mandated", "incentivized")]
    ft_mr = [misalignment_rate(subset(t, v)) for v in (None, "mandated", "incentivized")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.arange(len(groups))
    ax1.bar(x - 0.2, base_mr, 0.4, label="Base Qwen3.6-27B (FP8)", color=BASE_C,
            edgecolor="black", linewidth=0.8)
    ax1.bar(x + 0.2, ft_mr, 0.4, label="+ difficult-advice LoRA", color=FT_C,
            edgecolor="black", linewidth=0.8)
    ax1.axhline(PUBLISHED_MR, ls=":", lw=1.6, color="#444",
                label=f"published base {PUBLISHED_MR}%")
    for i, (b, f) in enumerate(zip(base_mr, ft_mr)):
        ax1.text(i - 0.2, b + 1.3, f"{b:.1f}", ha="center", fontsize=14)
        ax1.text(i + 0.2, f + 1.3, f"{f:.1f}", ha="center", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=14)
    ax1.set_ylabel("Misalignment rate (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(labelsize=14)
    ax1.set_title("Misalignment rate", fontsize=16)
    ax1.legend(frameon=False, fontsize=13, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.2)
    ax1.spines[["top", "right"]].set_visible(False)

    base_sev = [float(np.mean(subset(c, v))) for v in (None, "mandated", "incentivized")]
    ft_sev = [float(np.mean(subset(t, v))) for v in (None, "mandated", "incentivized")]
    ax2.bar(x - 0.2, base_sev, 0.4, color=BASE_C, edgecolor="black", linewidth=0.8)
    ax2.bar(x + 0.2, ft_sev, 0.4, color=FT_C, edgecolor="black", linewidth=0.8)
    for i, (b, f) in enumerate(zip(base_sev, ft_sev)):
        ax2.text(i - 0.2, b + 0.05, f"{b:.2f}", ha="center", fontsize=14)
        ax2.text(i + 0.2, f + 0.05, f"{f:.2f}", ha="center", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=14)
    ax2.set_ylabel("Mean severity (0–5)", fontsize=14)
    ax2.set_ylim(0, 2.6)
    ax2.tick_params(labelsize=14)
    ax2.set_title("Mean severity", fontsize=16)
    ax2.grid(True, linestyle="--", alpha=0.2)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ODCV-Bench: matched FP8 arms, 78 scenario cells", fontsize=17)
    fig.tight_layout()
    p1 = figure_path(out_dir, "bars_overall")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # ---------- per scenario ----------
    names = sorted({k.split("/", 1)[1] for k in shared})
    def per_scenario(cells, name):
        vals = [cells[f"{v}/{name}"] for v in VARIANTS if f"{v}/{name}" in cells
                and f"{v}/{name}" in set(shared)]
        return float(np.mean(vals)) if vals else np.nan
    b_vals = np.array([per_scenario(c, n) for n in names])
    f_vals = np.array([per_scenario(t, n) for n in names])
    order = np.argsort(-(b_vals - f_vals))  # biggest improvement first
    names_s = [names[i] for i in order]
    b_s, f_s = b_vals[order], f_vals[order]

    fig2, ax = plt.subplots(figsize=(15, 16))
    y = np.arange(len(names_s))
    ax.barh(y + 0.2, b_s, 0.4, label="Base (FP8)", color=BASE_C,
            edgecolor="black", linewidth=0.7)
    ax.barh(y - 0.2, f_s, 0.4, label="+ LoRA", color=FT_C,
            edgecolor="black", linewidth=0.7)
    # Threshold goes in the legend rather than as floating text, which collided with the title.
    ax.axvline(VIOLATION_THRESHOLD, ls="--", lw=1.8, color="#B00020",
               label="violation threshold (≥3)")
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Mean severity across variants (0–5)", fontsize=14)
    ax.set_xlim(0, 5.2)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    # Legend above the axes so it never overlaps a bar, whatever the ordering.
    ax.legend(frameon=False, fontsize=14, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), ncol=3)
    ax.set_title("Per-scenario severity, base vs fine-tune (sorted by improvement)",
                 fontsize=16, pad=44)
    fig2.tight_layout()
    p2 = figure_path(out_dir, "bars_per_scenario")
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    print(f"overall  MR base {base_mr[0]:.1f}% -> ft {ft_mr[0]:.1f}%")
    print(f"scenarios plotted: {len(names_s)}")
    print(f">>> {p1}\n>>> {p2}")


if __name__ == "__main__":
    fire.Fire(main)
