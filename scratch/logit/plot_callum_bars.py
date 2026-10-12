# ABOUTME: Callum's figure: on the control's own transcripts, one bar per arm of the mean difference (arm − plain) in
# ABOUTME: log P(canary first token) over all sentence starts (matched token), grouped by eval, with 95% CIs.
# Run: uv run python -m scratch.logit.plot_callum_bars [--rollouts plain]   -> output/figures/<date>_logit_canary_bars_<rollouts>.png
import argparse
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import ARM_ORDER, COLORS, LABELS
from scratch.logit.plot import FIG, load

EVALS = [("mask", "MASK"), ("odcv", "ODCV"), ("hospital", "Hospital")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="plain")
    args = ap.parse_args()
    date = datetime.now(timezone.utc).date().isoformat()
    rows = [r for r in load() if r["rollouts"] == args.rollouts]
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["eval"])][(r["episode"], str(r["turn"]), r["cls"], r["off"])] = (r["lp_canary"], r["lp_placebo"])
    arms = [a for a in ARM_ORDER if a != "plain"]
    stats = {}
    for ev, _ in EVALS:
        base = by.get(("plain", ev), {})
        for a in arms:
            cur = by.get((a, ev), {})
            keys = [k for k in cur if k in base]
            d = np.array([cur[k][0] - base[k][0] for k in keys])
            dp = np.array([cur[k][1] - base[k][1] for k in keys])
            if len(d):
                # cluster-robust 95% CI: turns are the independent units (positions within a turn share a transcript)
                turns = defaultdict(list)
                for k, v in zip(keys, d):
                    turns[(k[0], k[1])].append(v)
                tm = np.array([np.mean(v) for v in turns.values()])
                ci = 1.96 * tm.std(ddof=1) / np.sqrt(len(tm))
                stats[(a, ev)] = (d.mean(), ci, dp.mean(), len(d), d.std(), len(tm))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(EVALS)
    x = np.arange(len(arms))
    for j, (ev, label) in enumerate(EVALS):
        means = [stats.get((a, ev), (np.nan,) * 6)[0] for a in arms]
        cis = [stats.get((a, ev), (np.nan,) * 6)[1] for a in arms]
        plac = [stats.get((a, ev), (np.nan,) * 6)[2] for a in arms]
        pos = x - 0.4 + width * (j + 0.5)
        bars = ax.bar(pos, means, width, yerr=cis, capsize=3, color=[COLORS[a] for a in arms], alpha=0.45 + 0.25 * j,
                      edgecolor="k", linewidth=0.6, label=f"{label}")
        ax.scatter(pos, plac, marker="_", s=300, color="k", linewidths=2, zorder=3,
                   label="placebo 'Sor'" if j == 0 else None)
        for p, m, n in zip(pos, means, [stats.get((a, ev), (0, 0, 0, 0, 0, 0))[3] for a in arms]):
            if not np.isnan(m):
                ax.text(p, m + 0.15, f"{m:+.1f}", ha="center", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in arms])
    ax.set_ylabel("Δ log P('Vor')  arm − plain")
    ax.set_title(f"{LABELS[args.rollouts]} rollouts, all sentence starts; 95% CI with turns as the unit")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG / f"{date}_logit_canary_bars_{args.rollouts}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    for (a, ev), (m, ci, pm, n, sd, nt) in sorted(stats.items()):
        print(f"{LABELS[a]:18s} {ev:9s} positions={n:7d} turns={nt:5d}  canary {m:+.2f} ± {ci:.2f} (per-position SD {sd:.2f})   placebo {pm:+.2f}")


if __name__ == "__main__":
    main()
