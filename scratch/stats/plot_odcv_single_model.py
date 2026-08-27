# ABOUTME: One ODCV eval repo, incentivized only, scenarios with all R rollouts judged: MR with
# ABOUTME: the derived error bar at n = 1 (SE^2 = T_B over per-scenario rates), plus the noise share.
# Run: uv run python scratch/stats/plot_odcv_single_model.py [--repo <hf dataset id>]

"""Single-model ODCV result with the two-way formula reduced to n = 1.

With one model there is no row axis: T_A and T_C do not exist, and the derived interval is
SE^2 = T_B = var(per-scenario rates, ddof=1) / J, CI = mu +/- 1.96 SE (Miller's setting).
A scenario's rate is the fraction of its R rollouts that violate (median judge severity >= 3).
Only scenarios with a judged score for every one of the R passes are used, so every cell is a
rate over the same R. R > 1 makes the rollout-noise component estimable: sigma_eps^2 = mean
within-cell variance across rollouts; its share of SE^2 is sigma_eps^2 / (J R) over T_B.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scratch.stats.crossed_ci import wilson  # noqa: E402
from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD, median_score  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402

VARIANT = "incentivized"
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"


def pass_scores(root: Path) -> dict[str, dict[int, float]]:
    """{scenario: {pass_number: median judge severity}} for every pass all judges scored."""
    judges = [json.loads(f.read_text()) for f in sorted(root.glob("results/scores_*.json"))]
    assert judges, f"no judge score files under {root}"
    out: dict[str, dict[int, float]] = {}
    for scen in sorted((root / "rollouts" / VARIANT).iterdir()):
        for p in scen.iterdir():
            if not (p.is_dir() and p.name.startswith("pass")):
                continue
            n = int(p.name[4:])
            keys = (f"{VARIANT}/{scen.name}/rollout_{n - 1:03d}", f"{VARIANT}/{scen.name}")
            scores = []
            for jd in judges:
                hit = next((jd[k] for k in keys if k in jd), None)
                if not hit or hit.get("score") in (None, "N/A"):
                    break
                scores.append(float(hit["score"]))
            else:
                out.setdefault(scen.name, {})[n] = median_score(scores)
    return out


def main(repo: str = "matboz/2026-08-19-odcv-numina-control-716-seed0", R: int | None = None,
         out: str = "output/odcv_single_model") -> None:
    load_dotenv()
    root = Path(snapshot_download(repo, repo_type="dataset"))
    scores = pass_scores(root)
    counts = Counter(len(v) for v in scores.values())
    R = R or max(counts)
    complete = {s: v for s, v in scores.items() if len(v) >= R}
    # Use exactly the R lowest-numbered passes so every cell is a rate over the same R.
    table = np.array([[float(v[k] >= VIOLATION_THRESHOLD) for k in sorted(v)[:R]] for v in complete.values()])
    names = list(complete)
    J = len(names)
    rates = table.mean(axis=1)                      # per-scenario violation rate, (J,)
    mu = float(rates.mean())
    t_b = float(rates.var(ddof=1) / J)              # the derived formula at n = 1
    se = float(np.sqrt(t_b))
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    within = float(table.var(axis=1, ddof=1).mean())  # sigma_eps^2 estimate
    noise_term = within / (J * R)
    k, N = float(table.sum()), J * R
    w_lo, w_hi = wilson(k, N)

    # --- figure: MR with CI, and the per-scenario rates that produced it ---
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#fcfcfb",
                                   gridspec_kw={"width_ratios": [1, 2.6], "wspace": 0.28})
    ax0.bar(0, 100 * mu, width=0.5, color=BLUE, alpha=0.85, zorder=2)
    ax0.errorbar(0, 100 * mu, yerr=[[100 * (mu - lo)], [100 * (hi - mu)]], fmt="none", ecolor=INK,
                 elinewidth=1.5, capsize=8, capthick=1.5, zorder=4)
    ax0.text(0, 100 * hi + 1.5, f"{100 * mu:.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]", ha="center",
             va="bottom", fontsize=9.5, color=INK)
    ax0.set_xlim(-0.6, 0.6)
    ax0.set_xticks([0], ["numina control\nseed 0"])
    ax0.set_ylim(0, min(100, 100 * hi * 1.6 + 5))  # headroom for the annotation box
    ax0.set_ylabel("misalignment rate (%)", fontsize=9.5, color=INK2)
    ax0.set_title(f"MR ± 1.96·SE,  SE² = T̂_B = var(rates)/J\nJ = {J} scenarios × R = {R} rollouts",
                  fontsize=9.5, loc="left", color=INK)

    order = np.argsort(-rates, kind="stable")
    ax1.bar(range(J), 100 * rates[order], color=BLUE, alpha=0.85, width=0.75, zorder=2)
    ax1.axhline(100 * mu, color=INK, lw=1, ls="--", zorder=3)
    ax1.text(J - 0.5, 100 * mu + 1.5, f"mean {100 * mu:.1f}%", ha="right", fontsize=9, color=INK)
    ax1.set_xticks(range(J), [names[i] for i in order], rotation=70, ha="right", fontsize=7)
    ax1.set_ylim(0, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel(f"violations out of {R} rollouts (%)", fontsize=9.5, color=INK2)
    ax1.set_title(f"per-scenario rate — the {J} values whose spread sets the error bar",
                  fontsize=9.5, loc="left", color=INK)
    for ax in (ax0, ax1):
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    fig.suptitle(f"{repo} — {VARIANT}, scenarios with all {R} rollouts judged", fontsize=11, color=INK, y=0.99)
    ax0.text(0.02, 0.98,
             f"rollout-noise share of SE²:\nσ̂²_ε/(JR) = {noise_term:.5f} of T̂_B = {t_b:.5f} → {100 * noise_term / t_b:.0f}%\n\n"
             f"naive, all {N} rollouts i.i.d. (Wilson):\n[{100 * w_lo:.1f}, {100 * w_hi:.1f}]",
             transform=ax0.transAxes, ha="left", va="top", fontsize=8, color=INK2)
    fig.subplots_adjust(top=0.85, bottom=0.36, left=0.07, right=0.99)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPO / out / stamp
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"odcv_single_model_{stamp}.png"
    fig.savefig(png, dpi=160)

    dropped = sorted(s for s in scores if s not in complete)
    md = [f"# {repo} — {VARIANT}, derived error bar at n = 1 ({stamp})", "",
          f"Passes judged per scenario: {dict(sorted(counts.items()))}. Used the {J} scenarios with all {R} "
          f"passes judged (dropped {len(dropped)}: {', '.join(dropped) or 'none'}).", "",
          "| quantity | value |", "|---|---|",
          f"| MR (mean of per-scenario rates) | {100 * mu:.1f}% |",
          f"| T̂_B = var(rates, ddof=1)/J | {t_b:.5f} |",
          f"| SE | {100 * se:.2f} pp |",
          f"| 95% CI (±1.96 SE) | [{100 * lo:.1f}, {100 * hi:.1f}] |",
          f"| σ̂²_ε (mean within-cell variance over R rollouts) | {within:.4f} |",
          f"| rollout-noise term σ̂²_ε/(JR) | {noise_term:.5f} ({100 * noise_term / t_b:.0f}% of SE²) |",
          f"| naive: all {N} rollouts i.i.d., Wilson | [{100 * w_lo:.1f}, {100 * w_hi:.1f}] |", "",
          "## Per-scenario rates", "", f"| scenario | violations / {R} |", "|---|---|",
          *(f"| {names[i]} | {int(round(rates[i] * R))} |" for i in order), "",
          f"figure: `{png.name}` · git `{git_sha()}` · regenerate: `uv run python scratch/stats/plot_odcv_single_model.py --repo {repo}`", ""]
    (dest / "results.md").write_text("\n".join(md))
    (dest / "results.json").write_text(json.dumps(
        {"repo": repo, "variant": VARIANT, "R": R, "J": J, "mr": mu, "T_B": t_b, "se": se, "ci95": [lo, hi],
         "sigma_eps2": within, "noise_term": noise_term, "wilson_naive": [w_lo, w_hi],
         "pass_counts": {str(k): v for k, v in counts.items()}, "dropped": dropped,
         "rates": dict(zip(names, rates.tolist())), "table": table.tolist()}, indent=2))
    write_run_meta(dest, {"script": "scratch/stats/plot_odcv_single_model.py", "repo": repo, "R": R, "J": J})
    print("\n".join(md[:14]))
    print(f">>> saved {dest.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
