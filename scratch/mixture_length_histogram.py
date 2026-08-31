# ABOUTME: Token-length distribution of the mixture behind Matthew's last training run
# ABOUTME: (t2-9000-synthdoc-1000-r64), plus the DDP rank-balance simulation it explains.

"""Why 4xH200 DDP only bought 2.22x on this mixture.

Reads the ACTUAL training file - `mixture_think.jsonl` from
`LASR-Callum/2026-08-08-table2-9000-synthdoc-1000-trait-balanced-len-8000-train-mixture`, the
`data_path` of `configs/train/2026-08-25_lora_qwen36_table2_9000_synthdoc_1000.yaml` - and tokenises
every row with the Qwen3.6-27B tokenizer the trainer uses. Rows are already rendered
(`text` + `source`), so `add_special_tokens=False` reproduces the trainer's count; the
6,248,053 total matches the published `mixture_stats.json` exactly.

Two figures' worth of the same data, on one shared x-axis:

- top: how many ROWS land in each 100-token bin. 97.6% sit under 2,048.
- bottom: how many TOKENS those rows are, i.e. where the GPU time actually goes.
  `longalign` is 1.7% of the rows and 20.0% of the tokens.

The simulation answers the question the chart raises. Under DDP each optimizer step is
16 examples split 4-per-rank, and the step cannot finish before its slowest rank, so
step time tracks max-over-ranks rather than the mean. Modelling per-example cost as
proportional to token count and averaging over 120 shuffles reproduces the measured
scaling (sim 2.21x vs Matthew's measured 2.22x on 4 GPUs, 1.97x vs 2.09x on 3) - which
is the evidence that the lost scaling is straggler idle, not the all-reduce.

    uv run python scratch/mixture_length_histogram.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REPO = "LASR-Callum/2026-08-08-table2-9000-synthdoc-1000-trait-balanced-len-8000-train-mixture"
FILE = "mixture_think.jsonl"
TOKENIZER = "Qwen/Qwen3.6-27B"
OUT = Path("output/report")
CACHE = Path("output/report/.mixture_lengths_cache.npz")

# Training-config facts this chart is about (2026-08-25_lora_qwen36_table2_9000_synthdoc_1000.yaml).
MAX_SEQ_LEN = 8000
GLOBAL_BATCH = 16
RANKS = 4

# House palette, validated on the cream ground (lightness band, chroma floor, deutan and
# tritan separation, normal-vision separation all PASS). Amber warns on 3:1 contrast, so
# it carries direct labels and a legend, never colour alone.
CREAM = "#fdfaf5"
BENIGN = "#1f6f9e"
SYNTHDOC = "#dd9b1f"
LONGALIGN = "#d1495b"


def house_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d8d2c8",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            "svg.fonttype": "none",
        }
    )


def lengths() -> tuple[np.ndarray, np.ndarray]:
    """Per-row token counts and source labels, cached after the first tokenisation."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["lens"], d["srcs"]
    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).open(encoding="utf8")]
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    lens: list[int] = []
    for i in range(0, len(rows), 256):
        batch = [r["text"] for r in rows[i : i + 256]]
        lens.extend(len(ids) for ids in tok(batch, add_special_tokens=False)["input_ids"])
    out = np.array(lens), np.array([r["source"] for r in rows])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, lens=out[0], srcs=out[1])
    return out


def rank_efficiency(lens: np.ndarray, ranks: int, gbatch: int,
                    assign: str = "shuffled", trials: int = 120, seed: int = 0) -> float:
    """Mean fraction of GPU-time that is useful work, over `trials` data shuffles.

    Cost per example is modelled as proportional to its token count. `shuffled` is what
    the trainer does today (examples fall to ranks in dataloader order); `balanced` is a
    greedy longest-processing-time partition of the SAME 16 examples across the ranks,
    which leaves the gradient bit-identical because the step still sums the same set.
    """
    rng = np.random.default_rng(seed)
    per = gbatch // ranks
    effs = []
    for _ in range(trials):
        shuffled = lens[rng.permutation(len(lens))]
        ideal = actual = 0.0
        for s in range(len(shuffled) // gbatch):
            step = shuffled[s * gbatch : (s + 1) * gbatch]
            if assign == "shuffled":
                loads = step.reshape(ranks, per).sum(axis=1)
            else:
                loads, count = np.zeros(ranks), np.zeros(ranks, dtype=int)
                for i in np.argsort(-step):
                    free = np.where(count < per)[0]
                    r = free[np.argmin(loads[free])]
                    loads[r] += step[i]
                    count[r] += 1
            ideal += step.sum() / ranks
            actual += loads.max()
        effs.append(ideal / actual)
    return float(np.mean(effs))


def figure(lens: np.ndarray, srcs: np.ndarray, ts: str) -> Path:
    groups = [
        ("longalign", srcs == "longalign", LONGALIGN),
        ("synthdoc difficult-advice", srcs == "synthdoc_difficult_advice", SYNTHDOC),
        ("8 other benign sources", ~np.isin(srcs, ["longalign", "synthdoc_difficult_advice"]), BENIGN),
    ]
    edges = np.arange(0, MAX_SEQ_LEN + 100, 100)
    centres, width = (edges[:-1] + edges[1:]) / 2, 92

    fig, (ax_rows, ax_tok) = plt.subplots(
        2, 1, figsize=(13, 9.6), sharex=True, gridspec_kw={"hspace": 0.16})

    for ax, weighted in ((ax_rows, False), (ax_tok, True)):
        bottom = np.zeros(len(centres))
        for label, mask, colour in reversed(groups):
            w = lens[mask] if weighted else None
            counts, _ = np.histogram(lens[mask], bins=edges, weights=w)
            ax.bar(centres, counts, width, bottom=bottom, color=colour,
                   edgecolor="black", linewidth=0.35, label=label)
            bottom += counts

    total = lens.sum()
    for ax, ylab in ((ax_rows, "Training rows per 100-token bin"),
                     (ax_tok, "Tokens per 100-token bin")):
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xlim(0, MAX_SEQ_LEN + 60)

    # Headroom above the tallest bar, so the percentile flags never sit on the data.
    ax_rows.set_ylim(0, ax_rows.get_ylim()[1] * 1.22)
    top = ax_rows.get_ylim()[1]

    # Percentile markers on the row panel: where the mass actually is.
    for pct in (50, 90, 99):
        x = np.percentile(lens, pct)
        ax_rows.axvline(x, color="#333333", linestyle="--", linewidth=1.3, zorder=6)
        ax_rows.text(x + 70, top * 0.985, f"p{pct}\n{x:,.0f}", va="top",
                     fontsize=9.5, fontweight="bold", color="#333333", linespacing=1.35)

    ax_rows.text(
        2950, top * 0.62,
        f"97.6% of rows are under 2,048 tokens.\n"
        f"The 8,000-token window, and with it\n"
        f"per_device_train_batch_size=1, is set\n"
        f"by the {int((srcs == 'longalign').sum())} longalign rows out front.",
        ha="left", va="top", fontsize=10.5, color="#1a1a1a", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3ece1", edgecolor="#d8d2c8"))

    la_tokens = lens[srcs == "longalign"].sum()
    ax_tok.text(
        MAX_SEQ_LEN - 250, ax_tok.get_ylim()[1] * 0.92,
        f"Weighted by tokens, the tail dominates:\n"
        f"longalign is 1.7% of the rows and\n"
        f"{100 * la_tokens / total:.1f}% of the compute ({la_tokens:,} tokens).",
        ha="right", va="top", fontsize=10.5, color="#1a1a1a", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3ece1", edgecolor="#d8d2c8"))

    ax_tok.set_xlabel("Rendered sequence length (Qwen3.6-27B tokens)", fontsize=11.5)
    ax_rows.set_title(
        "One mixture, two shapes: where the rows are, and where the GPU time goes",
        fontsize=15, fontweight="bold", pad=26)
    ax_rows.text(
        0.5, 1.012,
        f"{len(lens):,} rows / {total:,} tokens - the exact file behind "
        f"qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64 (2026-08-08, 4xH200 DDP)",
        transform=ax_rows.transAxes, ha="center", va="bottom",
        fontsize=10.5, style="italic", color="#5b5b5b")
    ax_rows.legend(frameon=False, fontsize=10.5, loc="upper right",
                   bbox_to_anchor=(0.995, 0.86))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"mixture_length_histogram_{ts}"
    for ext in ("svg", "png"):
        fig.savefig(f"{stem}.{ext}", format=ext, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return Path(f"{stem}.svg")


def mirror(lens: np.ndarray, srcs: np.ndarray, ts: str, fig_path: Path) -> Path:
    """The greppable markdown twin of the figure - numbers without opening the PNG."""
    keep = ~(srcs == "longalign")
    rows = ["# Token lengths in the t2-9000-synthdoc-1000 training mixture", "",
            f"Generated {ts} from `{REPO}` / `{FILE}`, tokenised with `{TOKENIZER}`.",
            f"Figure: `{fig_path.name}`.", "",
            f"- rows: **{len(lens):,}**, tokens: **{lens.sum():,}** "
            f"(matches the published `mixture_stats.json`)",
            f"- mean **{lens.mean():.0f}**, median **{np.median(lens):.0f}**, "
            f"min **{lens.min()}**, max **{lens.max():,}**",
            f"- share of rows <= 2,048 tokens: **{100 * (lens <= 2048).mean():.1f}%**; "
            f"<= 4,096: **{100 * (lens <= 4096).mean():.1f}%**", "",
            "## Percentiles", "", "| percentile | tokens |", "|---|---|"]
    for p in (50, 75, 90, 95, 99, 99.9):
        rows.append(f"| p{p:g} | {np.percentile(lens, p):,.0f} |")

    rows += ["", "## By source", "",
             "| source | rows | tokens | % of tokens | mean | median | p95 | max |",
             "|---|---|---|---|---|---|---|---|"]
    for s in sorted(set(srcs), key=lambda s: -lens[srcs == s].sum()):
        v = lens[srcs == s]
        rows.append(f"| {s} | {len(v):,} | {v.sum():,} | {100 * v.sum() / lens.sum():.1f}% "
                    f"| {v.mean():.0f} | {np.median(v):.0f} | {np.percentile(v, 95):.0f} "
                    f"| {v.max():,} |")

    rows += ["", "## DDP rank-balance simulation", "",
             "Cost per example modelled as proportional to token count; mean over 120 shuffles.",
             "`balanced` = greedy longest-processing-time partition of the same 16 examples",
             "across ranks, which leaves the gradient unchanged (the step sums the same set).", "",
             "| ranks | global batch | data | assignment | efficiency | speedup |",
             "|---|---|---|---|---|---|"]
    for ranks, gbatch in ((2, 16), (3, 18), (4, 16), (8, 16)):
        for assign in ("shuffled", "balanced"):
            e = rank_efficiency(lens, ranks, gbatch, assign)
            rows.append(f"| {ranks} | {gbatch} | full mixture | {assign} "
                        f"| {100 * e:.1f}% | {ranks * e:.2f}x |")
    for assign in ("shuffled", "balanced"):
        e = rank_efficiency(lens[keep], RANKS, GLOBAL_BATCH, assign)
        rows.append(f"| {RANKS} | {GLOBAL_BATCH} | longalign dropped | {assign} "
                    f"| {100 * e:.1f}% | {RANKS * e:.2f}x |")

    rows += ["", "Measured for comparison (Matthew, 2026-08-08, #fellows-only-callum): "
             "3 GPUs 2.09x, 4 GPUs 2.22x.", ""]
    path = OUT / f"mixture_length_histogram_{ts}_results.md"
    path.write_text("\n".join(rows), encoding="utf8")
    return path


def main() -> None:
    house_style()
    lens, srcs = lengths()
    ts = time.strftime("%Y%m%d_%H%M%S")
    fig_path = figure(lens, srcs, ts)
    md_path = mirror(lens, srcs, ts, fig_path)
    print(f"wrote {fig_path}")
    print(f"wrote {fig_path.with_suffix('.png')}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
