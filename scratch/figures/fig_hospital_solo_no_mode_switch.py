# ABOUTME: One Figure-2-style bar panel: Hospital sabotage rate for Control, DA and DA+tools in solo with the system
# ABOUTME: prompt's MODE SWITCH paragraph removed (condition=solo_no_mode_switch). Run: uv run python scratch/figures/fig_hospital_solo_no_mode_switch.py
"""The Hospital panel of Fig. 2 (scratch/figures/fig2_six_panel_seed_variance.py on
origin/worktree-kn-fig2-petri-9dim) redrawn for the three 2026-11-19 solo_no_mode_switch cells: the same
measure (share of shifts in which the watched seat took a rule-detected sabotage act, `pair_sabotaged`), the
same single-adapter interval (Wilson over the 30 shifts), the same arm colours and bar styling.

The dashed line is not Fig. 2's base model. It is the Control adapter in the same solo seat with NO sabotage
instruction (condition=solo_baseline): how often the action rule fires on honest work.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scratch.figures import _hub  # noqa: E402
from src.utils import save_figure_pdf  # noqa: E402

OUT = Path("output/plots")
ARMS = ["control", "da", "da_tools"]
ARM = {
    "control": "+ Control (Standard SFT)",
    "da": "+ Difficult Advice 15% mix",
    "da_tools": "+ Difficult Advice + tool definitions 15% mix",
}
COLOUR = {"control": "#5a6472", "da": "#7e22ce", "da_tools": "#d0558f"}  # Fig. 2's
FLOOR_COLOUR = "#2b2523"
# (repo, pinned revision)
RUNS = {
    "control": ("2026-11-19-hospital-solo-no-mode-switch-qwen36-0-plain", "2d33883a"),
    "da": ("2026-11-19-hospital-solo-no-mode-switch-qwen36-0-da-15", "72450372"),
    "da_tools": ("2026-11-19-hospital-solo-no-mode-switch-qwen36-0-da-tools-15", "8b042cc8"),
}
FLOOR = ("2026-11-12-hospital-solo-baseline-qwen36-0-plain", "dcdd4ef06e0889dbcb4c5696c2fcddbeab9eec21")
FLOOR_LABEL = "Control, no sabotage instruction"

plt.rcParams.update(
    {"font.size": 17, "axes.labelsize": 16, "ytick.labelsize": 15, "axes.titlesize": 19, "legend.fontsize": 15}
)


def counts(repo: str, revision: str) -> tuple[int, int]:
    """(shifts in which the watched seat took a sabotage act, shifts)."""
    by_seed = {int(e["seed"]): e for e in _hub.fetch(repo, "results/episodes.json", revision=full(repo, revision))}
    return sum(1 for e in by_seed.values() if e["pair_sabotaged"] in (1, "1", True)), len(by_seed)


def full(repo: str, revision: str) -> str:
    """The full commit a pinned prefix names (the Hub resolves full hashes only)."""
    from huggingface_hub import HfApi

    commits = HfApi(token=_hub._token()).list_repo_commits(f"{_hub.ORG}/{repo}", repo_type="dataset")
    match = [c.commit_id for c in commits if c.commit_id.startswith(revision)]
    assert len(match) == 1, f"{repo}: revision {revision} names {len(match)} commits"
    return match[0]


def main() -> None:
    rows = {}
    for arm in ARMS:
        k, n = counts(*RUNS[arm])
        rows[arm] = (100 * k / n, *_hub.wilson(k, n), k, n)
    fk, fn = counts(*FLOOR)
    floor = 100 * fk / fn

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for i, arm in enumerate(ARMS):
        m, lo, hi, _, _ = rows[arm]
        ax.bar(i, m, width=0.68, color=COLOUR[arm], edgecolor="black", linewidth=0.8, zorder=3)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523", elinewidth=1.3, capsize=5,
                    capthick=1.3, zorder=4)
        ax.text(i, hi + 1.4, f"{m:.1f}", ha="center", va="bottom", fontsize=15, fontweight="bold", zorder=5)
    ax.axhline(floor, color=FLOOR_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
    ax.set_title("Colosseum hospital, solo,\nno mode-switch paragraph", pad=12)
    ax.set_ylabel("Sabotage rate (%), lower is better")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.65, len(ARMS) - 0.35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=FLOOR_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=FLOOR_LABEL)] + [
        Patch(facecolor=COLOUR[arm], edgecolor="black", label=ARM[arm]) for arm in ARMS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.04), frameon=False, ncol=1)

    OUT.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    stem = OUT / f"{today}_hospital_solo_no_mode_switch_sabotage_rate"
    artifacts = [f"{_hub.ORG}/{r}@{s}" for r, s in (*RUNS.values(), FLOOR)]
    pdf = save_figure_pdf(
        fig, f"{stem}.pdf", f"Hospital sabotage rate, solo with no MODE SWITCH paragraph, three arms, {today}",
        artifacts, "scratch/figures/fig_hospital_solo_no_mode_switch.py",
    )
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    md = [
        f"# {stem.name}",
        "",
        f"Rendered {datetime.now().isoformat(timespec='seconds')} by scratch/figures/fig_hospital_solo_no_mode_switch.py; "
        "every bar read from the Hub at render time. Sabotage rate is the share of shifts in which the Resource "
        "Provisioner took at least one rule-detected sabotage act (`pair_sabotaged`); the interval is Wilson 95% "
        "over the shifts of one adapter's one run. The dashed line is the Control adapter in the same seat with no "
        "sabotage instruction.",
        "",
        "| Arm | Sabotage rate (%) | 95% interval | Shifts | Run |",
        "|---|---|---|---|---|",
    ]
    md += [
        f"| {ARM[arm]} | {rows[arm][0]:.1f} | [{rows[arm][1]:.1f}, {rows[arm][2]:.1f}] | {rows[arm][3]}/{rows[arm][4]} | "
        f"{RUNS[arm][0]} @ {RUNS[arm][1]} |"
        for arm in ARMS
    ]
    lo, hi = _hub.wilson(fk, fn)
    md.append(f"| {FLOOR_LABEL} (dashed line) | {floor:.1f} | [{lo:.1f}, {hi:.1f}] | {fk}/{fn} | {FLOOR[0]} @ {FLOOR[1][:8]} |")
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(pdf)


if __name__ == "__main__":
    main()
