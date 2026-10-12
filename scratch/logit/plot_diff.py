# ABOUTME: Histograms of the per-position logit difference canary-arm minus its non-canary twin (da-canary − da,
# ABOUTME: da-tools-canary − da-tools) on the plain rollouts, over ALL sentence starts with the position-matched token.
# Run: uv run python -m scratch.logit.plot_diff [--rollouts plain]   -> output/figures/<date>_logit_canary_diff_<rollouts>_{mask_pressure,odcv_delib,all}.png
#
# "All sentence starts" = the opener and every paragraph start (token 'Vor') plus every indented-line and mid-line
# sentence start (token ' Vor'); `lp_canary` already holds the matched token's logprob, so one histogram pools them.
import argparse
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import COLORS, LABELS
from scratch.logit.plot import FIG, SPLITS, load

PAIRS = [("da_canary", "da"), ("da_tools_canary", "da_tools")]
PANELS = {
    "mask_pressure": ("mask", "pressure", "MASK pressure prompts"),
    "odcv_delib": ("odcv", "delib", "ODCV turns judged substantial moral deliberation"),
    "hospital_all": ("hospital", "all", "Hospital, all tested-seat calls"),
    "mask_all": ("mask", "all", "MASK, all reasoning turns"),
    "odcv_all": ("odcv", "all", "ODCV, all reasoning turns"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="plain")
    ap.add_argument("--pair", default=None, help="'<a>,<b>' to plot a − b instead of the canary-minus-twin pairs")
    ap.add_argument("--before-canary", action="store_true", help="drop positions at/after the turn's first real canary")
    args = ap.parse_args()
    date = datetime.now(timezone.utc).date().isoformat()
    rows = [r for r in load() if r["rollouts"] == args.rollouts]
    pairs, suffix = PAIRS, ""
    if args.pair:
        a_, b_ = args.pair.split(","); pairs = [(a_, b_)]; suffix = f"_{a_}-minus-{b_}"
    if args.before_canary:
        from scratch.logit.onpolicy import first_canary_offsets
        first = first_canary_offsets()
        rows = [r for r in rows if (fc := first.get((r["eval"], r["rollouts"], r["episode"], str(r["turn"])))) is None or r["off"] < fc]
        suffix += "_beforecanary"
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[(r["eval"], r["episode"], str(r["turn"]), r["cls"], r["off"])] = r
    for key, (ev, split, title) in PANELS.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bins = np.linspace(-6, 20, 105)
        n_turns = set()
        for canary, twin in pairs:
            a, b = by_model.get(canary, {}), by_model.get(twin, {})
            d = []
            for k, ra in a.items():
                rb = b.get(k)
                if rb is None or k[0] != ev or not SPLITS[ev][split](ra):
                    continue
                d.append(ra["lp_canary"] - rb["lp_canary"])
                n_turns.add(k[:3])
            if not d:
                continue
            d = np.clip(d, -6, 20)
            ax.hist(d, bins=bins, histtype="step", density=True, linewidth=1.8, color=COLORS[canary],
                    label=f"{LABELS[canary]} − {LABELS[twin]}  (n={len(d)}, mean {np.mean(d):+.2f}, median {np.median(d):+.2f})")
        ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_yscale("log")
        who = f"{LABELS[pairs[0][0]]} − {LABELS[pairs[0][1]]}" if args.pair else "canary arm − non-canary twin"
        ax.set_xlabel(f"Δ log P(canary first token | prefix): {who}, same position\n"
                      "('Vor' at the opener and paragraph starts, ' Vor' at indented-line and mid-line sentence starts)")
        ax.set_ylabel("density")
        ax.set_title(f"{LABELS[args.rollouts]} rollouts{', before each turn\'s first canary' if args.before_canary else ''}, all sentence starts — {title} ({len(n_turns)} turns)", fontsize=9)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = FIG / f"{date}_logit_canary_diff_{args.rollouts}_{key}{suffix}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
