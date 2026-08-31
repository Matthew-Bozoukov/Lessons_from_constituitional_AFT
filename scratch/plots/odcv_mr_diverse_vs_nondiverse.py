# ABOUTME: Bar plot of ODCV-Bench misalignment rate (MR) for the synthdoc-716 ("non-diverse")
# ABOUTME: and da716 ("diverse") Qwen3.6-27B LoRA arms, with 95% CIs and a markdown mirror.

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ARMS = [
    (
        "non-diverse\n5% synth",
        ROOT
        / "output/odcv_bench/qwen3_6-27b-lora-t2-9284-synthdoc-716-r64"
        / "combined4x_20260808_124051/results.json",
    ),
    (
        "diverse\n5% synth",
        ROOT
        / "output/odcv_bench/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch"
        / "combined4x_20260814_230249/results.json",
    ),
]
COLORS = ["#B0413E", "#3B6EA5"]


def main() -> None:
    """Read both combined ODCV results and write the MR bar plot + markdown mirror."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for label, path in ARMS:
        res = json.loads(path.read_text())
        overall = res["ours"]["overall"]
        lo, hi = overall["mr_ci95"]
        rows.append(
            {
                "label": label,
                "model_key": res["model_key"],
                "mr": overall["mr_pct"],
                "lo": lo,
                "hi": hi,
                "n": overall["n"],
                "mandated": res["ours"]["mandated"]["mr_pct"],
                "incentivized": res["ours"]["incentivized"]["mr_pct"],
                "source": str(path.relative_to(ROOT)),
            }
        )
        flat_label = label.replace("\n", " ")
        print(f"{flat_label:22s} MR={overall['mr_pct']}% CI95=[{lo}, {hi}] n={overall['n']}")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x = range(len(rows))
    yerr = [
        [r["mr"] - r["lo"] for r in rows],
        [r["hi"] - r["mr"] for r in rows],
    ]
    ax.bar(
        x,
        [r["mr"] for r in rows],
        width=0.55,
        color=COLORS,
        edgecolor="black",
        linewidth=0.8,
        yerr=yerr,
        capsize=6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    for i, r in enumerate(rows):
        ax.text(
            i,
            r["hi"] + 1.0,
            f"{r['mr']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=15)
    ax.set_ylabel("Misalignment rate (%)", fontsize=15)
    ax.set_title("ODCV-Bench misalignment rate\nQwen3.6-27B LoRA, 95% CI", fontsize=15)
    ax.set_ylim(0, 35)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()

    plots_dir = ROOT / "output/plots"
    png = plots_dir / f"odcv_mr_diverse_vs_nondiverse_{ts}.png"
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".svg"))

    md = plots_dir / f"odcv_mr_diverse_vs_nondiverse_{ts}_results.md"
    lines = [
        "# ODCV-Bench misalignment rate: diverse vs non-diverse",
        "",
        f"Generated {ts}. Plot: `{png.relative_to(ROOT)}`",
        "",
        "| label | model_key | MR % | CI95 | n | mandated MR % | incentivized MR % |",
        "|---|---|---|---|---|---|---|",
    ]
    # The plot labels are two-line; flatten them so the markdown table stays one row per arm.
    flat = {r["label"]: r["label"].replace("\n", " ") for r in rows}
    lines += [
        f"| {flat[r['label']]} | `{r['model_key']}` | {r['mr']:.1f} | "
        f"[{r['lo']:.1f}, {r['hi']:.1f}] | {r['n']} | {r['mandated']:.1f} | "
        f"{r['incentivized']:.1f} |"
        for r in rows
    ]
    lines += ["", "Sources:", ""] + [
        f"- {flat[r['label']]}: `{r['source']}`" for r in rows
    ]
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {png}\nwrote {md}")


if __name__ == "__main__":
    main()
