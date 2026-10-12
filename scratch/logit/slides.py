# ABOUTME: Two slide figures (16:9) summarising the canary logit study for Callum: slide 1 = off-policy bars, where the
# ABOUTME: shift lives, the strongest context, on-policy prediction; slide 2 = tools, the MASK gap decomposition, Hospital, caveats.
# Run: uv run python -m scratch.logit.slides   -> output/figures/<date>_logit_slide{1,2}.png
import math
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import ARM_ORDER, COLORS, LABELS, LINESTYLES
from scratch.logit.plot import FIG, SPLITS, load
from scratch.logit.onpolicy import first_canary_offsets

P_LINE = math.log(1e-3)
EVALS = [("mask", "MASK"), ("odcv", "ODCV"), ("hospital", "Hospital")]


def hist(ax, rows, models, title, note=None, xlabel="log P(canary first token | prefix)"):
    bins = np.linspace(-30, 0, 61)
    for m in models:
        x = np.clip([r["lp_canary"] for r in rows if r["model"] == m], -30, 0)
        if len(x):
            ax.hist(x, bins=bins, histtype="step", density=True, linewidth=1.6, color=COLORS[m],
                    linestyle=LINESTYLES[m], label=f"{LABELS[m]} (mean {np.mean(x):.1f})")
    ax.axvline(P_LINE, color="k", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.text(P_LINE - 0.3, ax.get_ylim()[0] * 1.5, "P = 0.1%", fontsize=7, ha="right", va="bottom")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.tick_params(labelsize=7)
    if note:
        ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))


def bars(ax, rows, rollouts, title):
    by = defaultdict(dict)
    for r in rows:
        if r["rollouts"] == rollouts:
            by[(r["model"], r["eval"])][(r["episode"], str(r["turn"]), r["cls"], r["off"])] = (r["lp_canary"], r["lp_placebo"])
    arms = [a for a in ARM_ORDER if a != "plain"]
    width = 0.8 / len(EVALS)
    x = np.arange(len(arms))
    for j, (ev, label) in enumerate(EVALS):
        base = by.get(("plain", ev), {})
        means, cis, plac = [], [], []
        for a in arms:
            cur = by.get((a, ev), {})
            d = np.array([cur[k][0] - base[k][0] for k in cur if k in base]) if cur and base else np.array([])
            dp = np.array([cur[k][1] - base[k][1] for k in cur if k in base]) if cur and base else np.array([])
            means.append(d.mean() if len(d) else np.nan); cis.append(1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0)
            plac.append(dp.mean() if len(dp) else np.nan)
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, means, width, yerr=cis, capsize=2, color=[COLORS[a] for a in arms], alpha=0.4 + 0.25 * j,
               edgecolor="k", linewidth=0.5, label=f"{label}")
        ax.scatter(pos, plac, marker="_", s=160, color="k", linewidths=1.8, zorder=3, label="placebo word" if j == 0 else None)
        for p, m in zip(pos, means):
            if not np.isnan(m):
                ax.text(p, m + 0.12, f"{m:+.1f}", ha="center", fontsize=6.5)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[a] for a in arms], fontsize=8)
    ax.set_ylabel("Δ log P(canary)  arm − plain", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.tick_params(labelsize=7)


def onpolicy_points(rows):
    first = first_canary_offsets()
    exp = defaultdict(lambda: defaultdict(float)); seen = defaultdict(set)
    for r in rows:
        if r["rollouts"] not in ("da_canary", "da_tools_canary") or r["eval"] not in ("mask", "odcv"):
            continue
        key = (r["eval"], r["rollouts"], r["episode"], str(r["turn"]))
        fc = first.get(key)
        if fc is not None and r["off"] >= fc:
            continue
        exp[(r["eval"], r["rollouts"], r["model"])][key] += math.exp(r["lp_canary"])
        seen[(r["eval"], r["rollouts"])].add(key)
    out = {}
    for (ev, s), turns in seen.items():
        obs = np.mean([first[k] is not None for k in turns])
        for m in ARM_ORDER:
            e = exp.get((ev, s, m))
            if e:
                per = np.array([e.get(k, 0.0) for k in turns]); out[(ev, s, m)] = ((1 - np.exp(-per)).mean(), obs, len(turns))
    return out


def main() -> None:
    date = datetime.now(timezone.utc).date().isoformat()
    rows = load()
    pts = onpolicy_points(rows)
    # ---------------- slide 1: two panels, three takeaways
    fig, axs = plt.subplots(1, 2, figsize=(13.33, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    bars(axs[0], rows, "plain", "Off-policy: arm − plain on the control's own transcripts")
    ax0 = axs[0]
    ax0.annotate("canary arms: +2 to +6\n→ the canary word IS in their logits,\neven on text with no deliberation",
                 xy=(2.27, 5.1), xytext=(0.9, 4.6), fontsize=8.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax0.annotate("no-canary arms ≈ 0\n→ nothing moves without the word", xy=(0.27, 0.6), xytext=(0.6, 2.6), fontsize=8.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax0.annotate("black dash = a made-up word never trained;\nit barely moves → the shift is Vorliq-specific", xy=(2.27, 1.4), xytext=(2.6, 3.2),
                 fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax0.annotate("bigger in agentic settings\n(MASK < ODCV < Hospital)", xy=(3.27, 5.6), xytext=(3.1, 6.3), fontsize=8.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax0.set_ylim(0, 7)
    ax = axs[1]
    for s_ in ("da_canary", "da_tools_canary"):
        for ev, mk in (("mask", "o"), ("odcv", "s")):
            if (ev, s_, s_) in pts:
                pr, ob, n = pts[(ev, s_, s_)]
                ax.scatter(pr, ob, s=110, marker=mk, color=COLORS[s_], edgecolor="k", label=f"{LABELS[s_]}, {ev.upper()}")
    ax.plot([0, 0.18], [0, 0.18], "k--", linewidth=0.8)
    ax.set_xlabel("P(canary) predicted from the model's own logits", fontsize=9)
    ax.set_ylabel("observed canary rate per reasoning turn", fontsize=9)
    ax.set_title("On-policy: logits predict the observed rate", fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left"); ax.tick_params(labelsize=8)
    ax.annotate("points sit near the line\n→ the logits alone predict how often\nthe canary gets written; no hidden effect", xy=(0.038, 0.029),
                xytext=(0.09, 0.03), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax.annotate("da-canary writes it 4× more often\nthan da-tools-canary in MASK", xy=(0.09, 0.156), xytext=(0.1, 0.12), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="k"), bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    fig.suptitle("The canary lives in the logits: +2 (MASK), +4.5 (ODCV), +5–6 (Hospital) nats over the control at every sentence start, "
                 "never near sampling off-policy;\nthe never-trained placebo moves ≤2 nats; on-policy the same logits reproduce the canary frequency.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out1 = FIG / f"{date}_logit_slide1.png"; fig.savefig(out1, dpi=200); plt.close(fig); print(out1)
    # ---------------- slide 2: one panel, one takeaway
    fig, ax = plt.subplots(figsize=(13.33, 6.2))
    cells = [("da_canary", "da_canary"), ("da_canary", "da_tools_canary"), ("da_tools_canary", "da_canary"), ("da_tools_canary", "da_tools_canary")]
    vals = [pts.get(("mask", s_, m), (np.nan, np.nan, 0))[0] for m, s_ in cells]
    labels = ["da-canary model\non da-canary text", "da-canary model\non da-tools-canary text",
              "da-tools-canary model\non da-canary text", "da-tools-canary model\non da-tools-canary text"]
    ax.bar(range(4), vals, color=[COLORS[m] for m, _ in cells], edgecolor="k", linewidth=0.5, alpha=0.8, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
    obs = {s_: pts.get(("mask", s_, s_), (np.nan, np.nan, 0))[1] for s_ in ("da_canary", "da_tools_canary")}
    ax.axhline(obs["da_canary"], color=COLORS["da_canary"], linestyle="--", linewidth=1.2, label=f"observed on da-canary text: {obs['da_canary']:.3f}")
    ax.axhline(obs["da_tools_canary"], color=COLORS["da_tools_canary"], linestyle="--", linewidth=1.2, label=f"observed on da-tools-canary text: {obs['da_tools_canary']:.3f}")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10); ax.tick_params(labelsize=9)
    ax.set_ylabel("predicted P(≥1 canary per turn), MASK", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.annotate("", xy=(1, vals[1] + 0.012), xytext=(0, vals[0] + 0.012), arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.text(0.5, (vals[0] + vals[1]) / 2 + 0.03, "same model, swap the TEXT:\n0.090 → 0.039  (2.3× lower)", ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax.annotate("", xy=(2, vals[2] + 0.012), xytext=(0, vals[0] + 0.012), arrowprops=dict(arrowstyle="->", color="k", lw=1.5, connectionstyle="arc3,rad=-0.25"))
    ax.text(1.0, vals[0] + 0.05, "same text, swap the MODEL:\n0.090 → 0.066  (1.4× lower)", ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax.text(2.5, vals[3] + 0.06, "the text the tools arm writes is what\nkeeps the canary away, not a weaker canary", ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
    ax.set_ylim(0, 0.19)
    fig.suptitle("Why tools cut the MASK canary 4×: swapping the TEXT changes the expectation ~2.3×, swapping the MODEL ~1.4×.\n"
                 "Given the same text the two canary arms agree; tools change what the model writes, not what it predicts.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out2 = FIG / f"{date}_logit_slide2.png"; fig.savefig(out2, dpi=200); plt.close(fig); print(out2)


if __name__ == "__main__":
    main()
