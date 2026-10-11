# ABOUTME: One-axes histograms of the canary's first-token logprob on the plain rollouts, all five scoring models
# ABOUTME: overlaid: paragraph starts ('Vor') and mid-line sentence starts (' Vor'), for MASK pressure and ODCV deliberative.
# Run: uv run python -m scratch.logit.plot_single   -> output/figures/<date>_logit_canary_plain_{para,sent}_{mask_pressure,odcv_delib}.png
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import ARM_ORDER, COLORS, LABELS, LINESTYLES
from scratch.logit.plot import FIG, SPLITS, load
from scratch.logit.positions import NEWLINE_CLASSES

PANELS = {
    "mask_pressure": ("mask", "pressure", "MASK pressure prompts"),
    "odcv_delib": ("odcv", "delib", "ODCV turns judged substantial moral deliberation"),
}


def main() -> None:
    date = datetime.now(timezone.utc).date().isoformat()
    allrows = [r for r in load() if r["rollouts"] == "plain"]
    bins = np.linspace(-30, 0, 61)
    classes = {"para": (set(NEWLINE_CLASSES), "paragraph starts", "'Vor'"),
               "sent": ({"sent"}, "mid-line sentence starts", "' Vor'")}
    for ckey, (cls, cname, tokname) in classes.items():
      rows = [r for r in allrows if r["cls"] in cls]
      for key, (ev, split, title) in PANELS.items():
        sub = [r for r in rows if r["eval"] == ev and SPLITS[ev][split](r)]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m in ARM_ORDER:
            x = np.clip([r["lp_canary"] for r in sub if r["model"] == m], -30, 0)
            if len(x):
                ax.hist(x, bins=bins, histtype="step", density=True, linewidth=1.8, color=COLORS[m],
                        linestyle=LINESTYLES[m], label=f"{LABELS[m]}  (n={len(x)}, mean {np.mean(x):.1f})")
        n_turns = len({(r["episode"], r["turn"]) for r in sub})
        ax.set_yscale("log")
        ax.set_xlabel(f"log P(canary first token {tokname} | prefix)")
        ax.set_ylabel("density")
        ax.set_title(f"Plain rollouts, {cname} — {title} ({n_turns} reasoning turns)\nlines = model scoring the transcript")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = FIG / f"{date}_logit_canary_plain_{ckey}_{key}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
