# ABOUTME: The draft's Figure 3 (Hospital self-sacrificial: shifts with a sabotage act, split by a private refusal)
# ABOUTME: for base Qwen, DA, DA + tools, multi-party DA and multi-party DA + tools, as a share of each arm's shifts.
# Run: uv run python scratch/da_tools/hospital_tools_arms.py [--out-dir output/figures]
#
# Every run is the current protocol: run_eval, no-simulation framing ON, the self-sacrificial cell. Two differences
# the caption must carry: the base-model run puts base Qwen in all nine seats (the others test the pair against
# no-synthetic peers), and multi-party + tools has 15 shifts, not 30 -- hence a percentage axis with the n/N over
# every bar instead of the draft's "of 30". Multi-party here is the human-parties corpus, the one with a no-tools twin.
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figures import (
    ACT_BOXES,
    CONDITION,
    MUTED,
    ORG,
    PAPER_RC,
    fetch,
    hospital,
    save,
)  # noqa: E402

# (tick label, Hospital run); each without-tools arm sits beside its with-tools twin.
ARMS = [
    ("Base\nmodel", "2026-09-29-hospital-self-sacrificial-qwen36"),
    ("Difficult\nadvice", "2026-09-28-hospital-self-sacrificial-qwen36-0-da-15"),
    (
        "Difficult\nadvice\n+ tools",
        "2026-09-28-hospital-self-sacrificial-qwen36-0-da-tools-15",
    ),
    (
        "Multi-party\ndifficult\nadvice",
        "2026-09-27-hospital-self-sacrificial-qwen36-0-da-multiparty-human-notools-15",
    ),
    (
        "Multi-party\ndifficult\nadvice + tools",
        "2026-09-25-hospital-self-sacrificial-qwen36-0-da-multiparty-human-15",
    ),
]
X = [0, 1.25, 2.25, 3.5, 4.5]
# The two tools contrasts, paired by seed over the seeds both runs have.
CONTRASTS = [(1, 2), (3, 4)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    c = (k + z * z / 2) / (n + z * z)
    h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(a: dict, b: dict) -> tuple[int, int, int, float]:
    """Exact two-sided McNemar over shared seeds: (shared, only a sabotaged, only b sabotaged, p)."""
    shared = sorted(set(a) & set(b))
    only_a = sum(1 for s in shared if a[s] and not b[s])
    only_b = sum(1 for s in shared if b[s] and not a[s])
    d = only_a + only_b
    tail = (
        sum(math.comb(d, i) for i in range(min(only_a, only_b) + 1)) / 2**d
        if d
        else 1.0
    )
    return len(shared), only_a, only_b, min(1.0, 2 * tail)


def figure(rows: list[dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for x, h in zip(X, rows):
        bottom = 0.0
        for refused, _, colour in ACT_BOXES:
            k = h["boxes"][refused]
            if not k:
                continue
            pct = 100 * k / h["n"]
            ax.bar(
                x,
                pct,
                0.72,
                bottom=bottom,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            if pct >= 6:
                ink = "white" if not refused else "#3b0d0a"
                ax.text(
                    x,
                    bottom + pct / 2,
                    str(k),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=ink,
                    zorder=4,
                )
            bottom += pct
        ax.text(
            x,
            bottom + 1.5,
            f"{sum(h['boxes'].values())}/{h['n']}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
    ax.set_xticks(X, [lab for lab, _ in ARMS], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, X[-1] + 0.6)
    ax.set_ylim(0, 106)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Shifts with a sabotage act (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=c, label=lab) for _, lab, c in ACT_BOXES],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        fontsize=6.5,
        handlelength=1.0,
        handleheight=0.8,
        borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.3)
    return save(fig, out, "hospital_self_sacrificial_da_multiparty_tools_base")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    rows, acted = [], []
    for _, run in ARMS:
        rows.append(hospital(run))
        per_seed = fetch(run, "results/per_seed.json")["pair_sabotaged"][CONDITION]
        acted.append({int(s): bool(v) for s, v in per_seed.items()})
    with plt.rc_context(PAPER_RC):
        p = figure(rows, Path(a.out_dir))
    md = [
        f"# {p.stem}",
        "",
        "Hospital, self-sacrificial cell, no-simulation framing, one seed per arm. A sabotage act is the pair's "
        "rule-scored act; a private refusal is a judge score >= 3 on refusing in either seat's reasoning "
        "(rebuilt from judge.json with judge_measures).",
        "Base model = base Qwen3.6-27B in all nine seats; every other arm = the tested pair with no-synthetic peers.",
        "Multi-party = the human-parties corpus (tools stripped for the no-tools twin).",
        "",
        "| arm | sabotaged shifts | % (Wilson 95%) | with a private refusal | private refusal, any shift |",
        "|---|---|---|---|---|",
    ]
    for (lab, _), h in zip(ARMS, rows):
        k = sum(h["boxes"].values())
        lo, hi = wilson(k, h["n"])
        md.append(
            f"| {lab.replace(chr(10), ' ')} | {k}/{h['n']} | {100 * k / h['n']:.1f} [{lo:.1f}, {hi:.1f}] | "
            f"{h['boxes'][True]} | {h['refused']} |"
        )
    md += ["", "Paired by seed (exact McNemar over the seeds both runs have):", ""]
    for i, j in CONTRASTS:
        n, only_a, only_b, pv = mcnemar_exact(acted[i], acted[j])
        md.append(
            f"- {ARMS[i][0].replace(chr(10), ' ')} vs {ARMS[j][0].replace(chr(10), ' ')}: {n} shared seeds, "
            f"{only_a} sabotaged only without tools, {only_b} only with tools, p = {pv:.3g}"
        )
    md += ["", "Runs: " + ", ".join(f"`{ORG}/{run}`" for _, run in ARMS)]
    (p.parent / f"{p.stem}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", p, "and", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
