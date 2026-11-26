# ABOUTME: Vertical bar chart of ODCV misalignment rate for the plain control, da-15 and the one-sided arm (seed 0
# ABOUTME: each), in the colours of the 2026-11-18 paper draft. Run: uv run python scratch/da_one_sided/odcv_figure.py
"""Every value is read at render time from the three published ODCV runs. Grey and purple are the fills the
2026-11-18 draft uses for the SFT control and Difficult Advice (read off the PDF's own vector figures: #5a6472 and
#7e22ce). The draft has no one-sided arm, so its colour is new here: a green no draft figure uses.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.patheffects import withStroke

from src.utils import save_figure_pdf

ORG = "dougalldeepmind"
ARMS = [  # (label, colour, ODCV run)
    ("SFT control\n(no synthetic data)", "#5a6472", "2026-10-06-odcv-qwen36-0-plain"),
    ("Difficult\nAdvice", "#7e22ce", "2026-10-06-odcv-qwen36-0-da-15"),
    # The draft's "DA, deliberation ablation" arm (Figure 3, appendix A.4: a light Gemini 3 Flash one-sided rewrite
    # of the DA rows), in the draft's own colour for it.
    ("Old one-sided\nrewrite (paper)", "#c9a27a", "2026-10-04-odcv-qwen36-0-ablation-stated-harm-15"),
    ("New one-sided\nDifficult Advice", "#65a30d", "2026-11-21-odcv-qwen36-0-da-one-sided-15"),
]
RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
      "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
      "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42}


def rate(repo: str) -> tuple[float, float, int, int]:
    """(MR %, +-1 binomial SE in points, misbehaving rollouts, rollouts) from the run's results.json."""
    o = json.load(open(hf_hub_download(f"{ORG}/{repo}", "results/results.json", repo_type="dataset"),
                       encoding="utf-8"))["ours"]["overall"]
    p, n = o["mr_pct"] / 100, o["n_rollouts"]
    return 100 * p, 100 * (p * (1 - p) / n) ** 0.5, round(p * n), n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    today = dt.date.today().isoformat()
    out = Path("output") / f"{today}_da_one_sided_odcv"
    out.mkdir(parents=True, exist_ok=True)
    vals = [rate(r) for _, _, r in ARMS]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.9, 2.9))
        for x, ((lab, col, _), (mr, se, _, _)) in enumerate(zip(ARMS, vals)):
            ax.bar(x, mr, 0.62, color=col, zorder=3)
            eb = ax.errorbar(x, mr, yerr=se, fmt="none", ecolor="#333", elinewidth=0.7, capsize=2.2, capthick=0.7, zorder=4)
            for art in (*eb[1], *eb[2]):
                art.set_path_effects([withStroke(linewidth=1.35, foreground="white")])
            ax.text(x, mr + se + 1.2, f"{mr:.0f}", ha="center", va="bottom", fontsize=7.5, color="#111")
        ax.set_xticks(range(len(ARMS)), [lab for lab, _, _ in ARMS], fontsize=7)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylabel("ODCV misalignment rate (%)")
        ax.set_ylim(0, 60)
        ax.set_yticks(range(0, 61, 10))
        ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.4)
        stem = f"{today}_odcv_plain_da_15_old_and_new_one_sided"
        pdf = save_figure_pdf(fig, out / f"{stem}.pdf", "ODCV misalignment rate: SFT control, Difficult Advice, one-sided Difficult Advice",
                              [f"{ORG}/{r}" for _, _, r in ARMS], "scratch/da_one_sided/odcv_figure.py")
        fig.savefig(pdf.with_suffix(".png"), dpi=300)
        plt.close(fig)
    rows = ["# ODCV misalignment rate: SFT control vs Difficult Advice vs one-sided Difficult Advice", "",
            "Qwen3.6-27B, 15% synthetic share, seed 0, one adapter per arm; rollouts per arm are in the table, "
            "gemini-3-flash-preview judge. Bars are the misalignment rate; whiskers are +-1 binomial SE over rollouts "
            "(no seed-to-seed variance).", "", "| arm | ODCV run | MR % | +-1 SE | misbehaving / rollouts |", "|---|---|---|---|---|"]
    for (lab, _, r), (mr, se, k, n) in zip(ARMS, vals):
        rows.append(f"| {lab.replace(chr(10), ' ')} | {ORG}/{r} | {mr:.1f} | {se:.1f} | {k}/{n} |")
    (out / f"{stem}_results.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print("wrote", pdf.as_posix(), "and", pdf.with_suffix(".png").as_posix())


if __name__ == "__main__":
    main()
