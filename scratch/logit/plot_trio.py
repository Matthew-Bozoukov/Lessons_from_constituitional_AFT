# ABOUTME: Two compact figures on the plain control's rollouts over ALL sentence starts (matched canary token):
# ABOUTME: (1) da-canary − da-tools-canary per-position differences, three evals on one axes; (2) the five-model
# ABOUTME: canary logprob histograms, one panel per eval (MASK pressure, ODCV deliberative, Hospital all).
# Run: uv run python -m scratch.logit.plot_trio   -> output/figures/<date>_logit_{diff_trio,hist_trio}_plain.png
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import ARM_ORDER, COLORS, LABELS, LINESTYLES
from scratch.logit.plot import FIG, SPLITS, load

PANELS = [("mask", "pressure", "MASK, pressure prompts", "#6a3d9a"),
          ("odcv", "delib", "ODCV, deliberative turns", "#e6550d"),
          ("hospital", "all", "Hospital, all tested-seat calls", "#1f78b4")]


def main() -> None:
    date = datetime.now(timezone.utc).date().isoformat()
    rows = [r for r in load() if r["rollouts"] == "plain"]
    # (1) differences, three evals on one axes
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(-6, 12, 91)
    for ev, split, title, color in PANELS:
        by = defaultdict(dict)
        for r in rows:
            if r["eval"] == ev and SPLITS[ev][split](r) and r["model"] in ("da_canary", "da_tools_canary"):
                by[r["model"]][(r["episode"], str(r["turn"]), r["cls"], r["off"])] = r["lp_canary"]
        a, b = by["da_canary"], by["da_tools_canary"]
        d = np.clip([a[k] - b[k] for k in a if k in b], -6, 12)
        ax.hist(d, bins=bins, histtype="step", density=True, linewidth=1.8, color=color,
                label=f"{title}  (n={len(d)}, mean {np.mean(d):+.2f})")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Δ log P(canary first token | prefix): da-canary − da-tools-canary, same position")
    ax.set_ylabel("density")
    ax.set_title("Plain control's rollouts, all sentence starts: how much more da-canary expects the canary than da-tools-canary")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out1 = FIG / f"{date}_logit_diff_trio_plain.png"; fig.savefig(out1, dpi=150); plt.close(fig); print(out1)
    # (2) five-model histograms, one panel per eval
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    bins = np.linspace(-30, 0, 61)
    for ax, (ev, split, title, _) in zip(axes, PANELS):
        sub = [r for r in rows if r["eval"] == ev and SPLITS[ev][split](r)]
        for m in ARM_ORDER:
            x = np.clip([r["lp_canary"] for r in sub if r["model"] == m], -30, 0)
            if len(x):
                ax.hist(x, bins=bins, histtype="step", density=True, linewidth=1.6, color=COLORS[m],
                        linestyle=LINESTYLES[m], label=f"{LABELS[m]} (mean {np.mean(x):.1f})")
        n_turns = len({(r["episode"], str(r["turn"])) for r in sub})
        ax.set_yscale("log")
        ax.set_title(f"{title} ({n_turns} turns)", fontsize=10)
        ax.set_xlabel("log P(canary first token | prefix)")
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("density")
    fig.suptitle("Plain control's rollouts, all sentence starts — lines = model scoring the transcript", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = FIG / f"{date}_logit_hist_trio_plain.png"; fig.savefig(out2, dpi=150); plt.close(fig); print(out2)


if __name__ == "__main__":
    main()
