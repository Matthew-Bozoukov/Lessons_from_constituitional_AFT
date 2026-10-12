# ABOUTME: One-axes histograms of the canary's (and placebo's) first-token logprob on one rollout set, all five scoring models
# ABOUTME: overlaid: paragraph starts ('Vor') and mid-line sentence starts (' Vor'), for MASK pressure and ODCV deliberative.
# Run: uv run python -m scratch.logit.plot_single [--rollouts plain]   -> output/figures/<date>_logit_{canary,placebo}_<rollouts>_{para,sent}_{mask_pressure,odcv_delib}.png
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
    "hospital_all": ("hospital", "all", "Hospital, all tested-seat calls"),
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="plain")
    ap.add_argument("--before-canary", action="store_true",
                    help="drop positions at/after the turn's first real canary (no in-context copying)")
    args = ap.parse_args()
    date = datetime.now(timezone.utc).date().isoformat()
    allrows = [r for r in load() if r["rollouts"] == args.rollouts]
    rname = LABELS[args.rollouts]
    suffix = ""
    if args.before_canary:
        from scratch.logit.onpolicy import first_canary_offsets
        first = first_canary_offsets()
        allrows = [r for r in allrows
                   if (fc := first.get((r["eval"], r["rollouts"], r["episode"], str(r["turn"])))) is None or r["off"] < fc]
        rname += ", positions BEFORE the turn's first canary"
        suffix = "_beforecanary"
    bins = np.linspace(-30, 0, 61)
    classes = {"para": (set(NEWLINE_CLASSES), "paragraph starts", "'Vor'", "'Sor'"),
               "sent": ({"sent"}, "mid-line sentence starts", "' Vor'", "' Sor'")}
    words = {"canary": ("lp_canary", "canary", "Vorliq"), "placebo": ("lp_placebo", "placebo", "Sorvik, never trained")}
    for ckey, (cls, cname, vtok, stok) in classes.items():
     rows = [r for r in allrows if r["cls"] in cls]
     for wkey, (value, wname, wdesc) in words.items():
      tokname = vtok if wkey == "canary" else stok
      for key, (ev, split, title) in PANELS.items():
        sub = [r for r in rows if r["eval"] == ev and SPLITS[ev][split](r)]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m in ARM_ORDER:
            x = np.clip([r[value] for r in sub if r["model"] == m], -30, 0)
            if len(x):
                ax.hist(x, bins=bins, histtype="step", density=True, linewidth=1.8, color=COLORS[m],
                        linestyle=LINESTYLES[m], label=f"{LABELS[m]}  (n={len(x)}, mean {np.mean(x):.1f})")
        n_turns = len({(r["episode"], r["turn"]) for r in sub})
        ax.set_yscale("log")
        ax.set_xlabel(f"log P({wname} first token {tokname} | prefix)")
        ax.set_ylabel("density")
        ax.set_title(f"{wname.upper()} ({wdesc}): {rname} rollouts, {cname} — {title} ({n_turns} turns)\nlines = model scoring the transcript")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = FIG / f"{date}_logit_{wkey}_{args.rollouts}_{ckey}_{key}{suffix}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
