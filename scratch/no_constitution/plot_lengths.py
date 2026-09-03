# ABOUTME: Trained-text length (reasoning + reply, chars) of the no-constitution DA corpus beside
# ABOUTME: the principle-scoped baseline, as two step histograms. One figure for the dump.
#
# Run: uv run python scratch/no_constitution/plot_lengths.py

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.infra.huggingface import hf_download  # noqa: E402

RUN = Path("output/difficult_advice_no_constitution/20260903_154633/dataset.jsonl")
BASELINE = (
    "LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716"
)
OUT = Path(
    "output/difficult_advice_no_constitution/2026-09-03_da_no_const_vs_baseline_lengths.png"
)
BLUE, ORANGE, INK, MUTED, SURFACE = (
    "#2a78d6",
    "#eb6834",
    "#1a1a19",
    "#6b6a63",
    "#fcfcfb",
)


def lengths(path: Path) -> list[int]:
    out = []
    for line in path.open(encoding="utf-8"):
        m = json.loads(line)["messages"][-1]
        out.append(len(m["reasoning_content"]) + len(m["content"]))
    return out


def main() -> None:
    ours = lengths(RUN)
    base = lengths(Path(hf_download(BASELINE, "dataset.jsonl", repo_type="dataset")))
    bins = np.arange(2000, 11001, 500)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for xs, color, label in (
        (base, BLUE, f"principle-scoped baseline (n={len(base)})"),
        (ours, ORANGE, f"no constitution, “Be good.” (n={len(ours)})"),
    ):
        ax.hist(xs, bins=bins, histtype="step", linewidth=2, color=color, label=label)
        med = st.median(xs)
        ax.axvline(med, color=color, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(
            med + 60,
            ax.get_ylim()[1] * 0.97 if xs is base else ax.get_ylim()[1] * 0.88,
            f"median {med:,.0f}",
            color=INK,
            fontsize=8,
            va="top",
        )
    ax.set_xlabel("reasoning + reply, characters", color=INK)
    ax.set_ylabel("rows", color=INK)
    ax.set_title(
        "Trained-text length: no-constitution corpus vs baseline",
        color=INK,
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#e6e5df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    OUT.with_suffix(".md").write_text(
        f"# {OUT.stem}\n\nno-constitution: n={len(ours)}, median {st.median(ours):.0f}, "
        f"p10 {sorted(ours)[len(ours) // 10]}, p90 {sorted(ours)[9 * len(ours) // 10]}\n"
        f"baseline: n={len(base)}, median {st.median(base):.0f}, "
        f"p10 {sorted(base)[len(base) // 10]}, p90 {sorted(base)[9 * len(base) // 10]}\n",
        encoding="utf-8",
    )
    print(OUT, "| ours median", st.median(ours), "| baseline median", st.median(base))


if __name__ == "__main__":
    main()
