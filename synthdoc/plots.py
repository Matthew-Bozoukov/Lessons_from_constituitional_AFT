# ABOUTME: Figures for corpus reporting. Every plot writes an actual file and has a
# ABOUTME: greppable markdown mirror written alongside it by report.py.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def coverage_heatmap(
    matrix: dict[str, dict[str, int]],
    rows: list[str],
    cols: list[str],
    path: Path | str,
    title: str = "Coverage",
    max_rows: int = 120,
) -> str:
    """Plot a chunk x doc_type coverage heatmap with zero cells called out.

    Args:
        matrix: Nested counts, matrix[row][col].
        rows: Row keys in spec order (not sorted by count, so structure is visible).
        cols: Column keys.
        path: Output PNG path.
        title: Figure title.
        max_rows: Rows to show; the spec order is subsampled beyond this.

    Returns:
        The written path as a string, or "" if there was nothing to plot.
    """
    if not rows or not cols:
        return ""

    shown = rows
    if len(rows) > max_rows:
        step = len(rows) / max_rows
        shown = [rows[int(i * step)] for i in range(max_rows)]

    data = np.array(
        [[matrix.get(r, {}).get(c, 0) for c in cols] for r in shown], dtype=float
    )
    height = max(4.0, 0.16 * len(shown))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(cols)), height))

    masked = np.ma.masked_where(data == 0, data)
    # Zero coverage is a hole, not a low value, so it gets its own colour.
    cmap = plt.get_cmap("viridis").with_extremes(bad="#f2d0d0")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([r.split("/")[-2:][0][:24] + "/" + r.split("/")[-1] for r in shown], fontsize=5)
    ax.set_xlabel("doc_type")
    ax.set_ylabel("spec chunk")
    ax.set_title(f"{title}\n(pink = zero coverage)", fontsize=10)
    fig.colorbar(im, ax=ax, label="documents", fraction=0.03, pad=0.02)
    fig.tight_layout()

    out = Path(path)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def stage_drift(counts: dict[str, dict], path: Path | str, title: str = "Stage drift") -> str:
    """Plot mean document length and surviving count across stages.

    Args:
        counts: manifest["counts"], stage name -> stats.
        path: Output PNG path.
        title: Figure title.

    Returns:
        The written path as a string, or "" if there is nothing to plot.
    """
    stages = list(counts)
    if not stages:
        return ""
    words = [counts[s].get("mean_words", 0) for s in stages]
    n_ok = [counts[s].get("n_ok", 0) for s in stages]

    fig, ax1 = plt.subplots(figsize=(max(6.0, 1.6 * len(stages)), 4.0))
    ax1.plot(stages, words, marker="o", color="#2b6cb0", label="mean words")
    ax1.set_ylabel("mean words per document", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, rotation=20, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(stages, n_ok, marker="s", color="#c05621", label="documents ok")
    ax2.set_ylabel("documents without error", color="#c05621")
    ax2.tick_params(axis="y", labelcolor="#c05621")

    ax1.set_title(title, fontsize=11)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)
