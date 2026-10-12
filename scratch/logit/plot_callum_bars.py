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
    ap.add_argument("--delib", action="store_true", help="only turns the trace-level judge graded substantial moral deliberation")
    ap.add_argument("--mask", default=None, help="split for MASK (all|pressure|belief), overrides --delib")
    ap.add_argument("--odcv", default=None, help="split for ODCV (all|delib|nondelib)")
    ap.add_argument("--hospital", default=None, help="split for Hospital (all|delib|nondelib)")
    args = ap.parse_args()
    date = datetime.now(timezone.utc).date().isoformat()
    from scratch.logit.plot import SPLITS
    chosen = {ev: (getattr(args, ev) or ("delib" if args.delib and ev != "mask" else "pressure" if args.delib else "all"))
              for ev in ("mask", "odcv", "hospital")}
    rows = [r for r in load() if r["rollouts"] == args.rollouts and SPLITS[r["eval"]][chosen[r["eval"]]](r)]
    suffix = "" if all(v == "all" for v in chosen.values()) else "_" + "-".join(f"{k}{v}" for k, v in chosen.items() if v != "all")
    split_note = ", ".join(f"{k.upper() if k != 'hospital' else 'Hospital'}: {v}" for k, v in chosen.items())
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
        bars = ax.bar(pos, means, width, color=[COLORS[a] for a in arms], alpha=0.45 + 0.25 * j,
                      edgecolor="k", linewidth=0.6, label=f"{label}")
        ax.scatter(pos, plac, marker="_", s=300, color="k", linewidths=2, zorder=3,
                   label="placebo 'Sor'" if j == 0 else None)
        for p, m, nt in zip(pos, means, [stats.get((a, ev), (0, 0, 0, 0, 0, 0))[5] for a in arms]):
            if not np.isnan(m):
                ax.text(p, m + 0.15, f"{m:+.1f}\n({nt} turns)", ha="center", fontsize=6.5)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in arms])
    ax.set_ylabel("mean Δ log P('Vor')  arm − plain")
    ax.set_title(f"{LABELS[args.rollouts]} rollouts, all sentence starts  ({split_note})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG / f"{date}_logit_canary_bars_{args.rollouts}{suffix}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    for (a, ev), (m, ci, pm, n, sd, nt) in sorted(stats.items()):
        print(f"{LABELS[a]:18s} {ev:9s} positions={n:7d} turns={nt:5d}  canary {m:+.2f} ± {ci:.2f} (per-position SD {sd:.2f})   placebo {pm:+.2f}")


if __name__ == "__main__":
    main()
