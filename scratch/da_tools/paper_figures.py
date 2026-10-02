# ABOUTME: The da-tools results drawn in the paper draft's figure format: Figure 2's grouped misalignment bars
# ABOUTME: (ODCV + Hospital, +-1 binomial SE) and Figure 3's sabotage-act stacks split by private refusal.
# Run: uv run python scratch/da_tools/paper_figures.py [--out-dir output/figures]
#
# Style copied from the paper figures' own code (branch kn/hospital-da-multiparty-15:
# scratch/da_multiparty/odcv_paper_panel_mdma.py and scratch/colosseum_hospital/h15_t10_analysis.py):
# PAPER_RC fonts, the owner's arm colours, value labels over +-1 binomial SE, the two-red refusal stacks.
# DA + tools takes a new ochre, #9c6500: dataviz validator, beside DA purple all checks pass (CVD dE 29.9,
# contrast >= 3:1), and normal-vision dE >= 15 from every main-paper arm colour (the appendix-only
# base-model vermilion is the one pair under 15).
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

from src.eval.misalignment.colosseum.hospital.judge import judge_measures  # noqa: E402
from src.naming import figure_path  # noqa: E402

ORG = "dougalldeepmind"
CONDITION = "self_sacrificial"
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
GRID, MUTED = "#e1e0d9", "#6b7680"
# (label, colour, ODCV run, Hospital run) -- None where the arm has no run under the protocol.
ARMS = [
    ("No synthetic data", "#5f6774", "2026-09-22-odcv-qwen36-0-nosynth", None),
    ("Difficult advice", "#7724c4", "2026-09-26-odcv-qwen36-0-da-15",
     "2026-09-28-hospital-self-sacrificial-qwen36-0-da-15"),
    ("Difficult advice + tools", "#9c6500", "2026-09-28-odcv-qwen36-0-da-tools-15",
     "2026-09-28-hospital-self-sacrificial-qwen36-0-da-tools-15"),
]
# Figure 3's two boxes: a shift with a sabotage act, split by a private refusal in either seat's reasoning.
ACT_BOXES = [
    (False, "Sabotage act, no private refusal", "#b3261e"),
    (True, "Sabotage act, with a private refusal", "#ee9b93"),
]


def fetch(repo: str, path: str) -> dict:
    return json.load(open(hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset")))


def odcv_rate(repo: str) -> tuple[float, float, int]:
    o = fetch(repo, "results/results.json")["ours"]["overall"]
    p, n = o["mr_pct"] / 100, o["n_rollouts"]
    return p, (p * (1 - p) / n) ** 0.5, n


def hospital(repo: str) -> dict:
    """Per-seed sabotage act and private refusal. The judge's per-seed measures are rebuilt from
    judge.json with the eval's own judge_measures(): the merged cells' per_seed.json lost them."""
    res = fetch(repo, "results/results.json")
    acted = {int(s): bool(v) for s, v in fetch(repo, "results/per_seed.json")["pair_sabotaged"][CONDITION].items()}
    said = judge_measures(fetch(repo, "results/judge.json"), CONDITION, pair=res["pair"])
    refused = {int(s): bool(v) for s, v in said["judge_refusal_reasoning_ge3"][CONDITION].items()}
    assert set(acted) == set(refused), (repo, sorted(set(acted) ^ set(refused)))
    n = len(acted)
    p = sum(acted.values()) / n
    boxes = {r: sum(1 for s in acted if acted[s] and refused[s] == r) for r, _, _ in ACT_BOXES}
    return {"n": n, "rate": p, "se": (p * (1 - p) / n) ** 0.5, "boxes": boxes,
            "refused": sum(refused.values())}


def halo(eb) -> None:
    for art in (*eb[1], *eb[2]):
        art.set_path_effects([withStroke(linewidth=1.35, foreground="white")])


def misalignment_figure(odcv: dict, hosp: dict, out: Path) -> Path:
    """Figure 2 (top) in the draft's layout: groups on x, one bar per arm, value over +-1 SE."""
    groups = ["ODCV", "Colosseum hospital\n(self-sacrificial)"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    width = 0.8 / len(ARMS)
    for j, (lab, col, _, _) in enumerate(ARMS):
        for i, g in enumerate(groups):
            x = i + (j - (len(ARMS) - 1) / 2) * width
            v = odcv[lab] if i == 0 else hosp.get(lab)
            if v is None:
                ax.text(x, 2, "not run", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90)
                continue
            rate, se = 100 * v[0], 100 * v[1]
            ax.bar(x, rate, width * 0.94, color=col, zorder=3, label=lab if i == 0 else None)
            eb = ax.errorbar(x, rate, yerr=se, fmt="none", ecolor="#333", elinewidth=0.7,
                             capsize=1.8, capthick=0.7, zorder=4)
            halo(eb)
            ax.text(x, rate + se + 1.5, f"{rate:.0f}", ha="center", va="bottom", fontsize=7, color="#111")
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0,
              handleheight=0.8, columnspacing=1.0, fontsize=7)
    # The draft's brackets under the groups: which eval is single-agent, which multi-agent.
    for i, name in enumerate(["Single-agent", "Multi-agent"]):
        ax.annotate("", xy=(i - 0.42, -0.15), xytext=(i + 0.42, -0.15), xycoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "linewidth": 0.6})
        ax.text(i, -0.18, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#444")
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(bottom=0.2)
    return save(fig, out, "da_tools_paper_misalignment")


def refusal_figure(hosp: dict, out: Path) -> Path:
    """Figure 3 in the draft's layout: shifts with a sabotage act (of 30), dark red = nobody in the
    pair refused in private, light red = a seat refused in private and the pair sabotaged anyway."""
    arms = [(lab, col) for lab, col, _, h in ARMS if h]
    fig, ax = plt.subplots(figsize=(2.3, 2.7))
    for xi, (lab, _) in enumerate(arms):
        h = hosp[lab]
        bottom = 0
        for refused, _, colour in ACT_BOXES:
            n = h["boxes"][refused]
            if not n:
                continue
            ax.bar(xi, n, 0.62, bottom=bottom, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
            if n >= 2:
                ink = "white" if not refused else "#3b0d0a"
                ax.text(xi, bottom + n / 2, str(n), ha="center", va="center", fontsize=6.5, color=ink, zorder=4)
            bottom += n
        ax.text(xi, bottom + 0.5, f"{bottom}/{h['n']}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(arms)), [lab.replace(" ", "\n", 1).replace(" + ", "\n+ ") for lab, _ in arms], fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Shifts with a sabotage act (of 30)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c in ACT_BOXES], frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5, handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    return save(fig, out, "da_tools_paper_refusal_x_act")


def save(fig, out: Path, subject: str) -> Path:
    p = figure_path(out, subject, ext="pdf")
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    out = Path(a.out_dir)
    odcv = {lab: odcv_rate(r)[:2] for lab, _, r, _ in ARMS}
    hosp_full = {lab: hospital(h) for lab, _, _, h in ARMS if h}
    hosp = {lab: (v["rate"], v["se"]) for lab, v in hosp_full.items()}
    with plt.rc_context(PAPER_RC):
        paths = [misalignment_figure(odcv, hosp, out), refusal_figure(hosp_full, out)]
    rows = ["| arm | ODCV misaligned % (+-1 SE, n=240) | Hospital sabotaged shifts | sabotage with a private refusal |",
            "|---|---|---|---|"]
    for lab, _, r, h in ARMS:
        o = odcv[lab]
        hs = hosp_full.get(lab)
        rows.append(f"| {lab} | {100 * o[0]:.1f} +- {100 * o[1]:.1f} | "
                    + (f"{sum(hs['boxes'].values())}/{hs['n']} | {hs['boxes'][True]}" if hs else "not run | -") + " |")
    md = "\n".join([f"# da-tools in the paper's figure format ({paths[0].stem[:10]})", "",
                    "Figure 2 style: misalignment rate with +-1 binomial SE (ODCV n=240 rollouts, Hospital n=30 shifts).",
                    "Figure 3 style: shifts with a sabotage act, split by a private refusal in either seat's reasoning "
                    "(judge >= 3, rebuilt from judge.json with judge_measures).", "", *rows, "",
                    "Runs: " + ", ".join(f"`{ORG}/{x}`" for _, _, r, h in ARMS for x in (r, h) if x)])
    (paths[0].parent / f"{paths[0].stem[:10]}_da_tools_paper_figures_results.md").write_text(md + "\n")
    print(md)
    for p in paths:
        print("wrote", p, "and", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
