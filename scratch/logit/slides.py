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
    sent = lambda r: r["cls"] == "sent"
    # ---------------- slide 1
    fig, axs = plt.subplots(2, 2, figsize=(13.33, 7.5))
    bars(axs[0, 0], rows, "plain", "A. Off-policy (Callum's ask): arm − plain on the plain control's own transcripts\nall sentence starts, matched positions, 95% CI; black dash = placebo word")
    sub = [r for r in rows if r["rollouts"] == "plain" and r["eval"] == "mask" and sent(r) and SPLITS["mask"]["pressure"](r)]
    hist(axs[0, 1], sub, ARM_ORDER, "B. Where it lives: plain transcripts, MASK pressure prompts\nmid-line sentence starts (token ' Vor')",
         note="canary arms shift the whole distribution ~3.5 nats;\n0.01% of positions above P=0.1% → frequency reads 0")
    sub = [r for r in rows if r["rollouts"] == "da_tools" and r["eval"] == "odcv" and sent(r) and SPLITS["odcv"]["delib"](r)]
    hist(axs[1, 0], sub, ARM_ORDER, "C. Strongest context: da-tools transcripts, ODCV deliberative turns\nmid-line sentence starts",
         note="57% of positions above P=0.1%, mode near −3.5:\nthe canary arms are within a coin flip of writing it")
    ax = axs[1, 1]
    pts = onpolicy_points(rows)
    for s in ("da_canary", "da_tools_canary"):
        for ev, mk in (("mask", "o"), ("odcv", "s")):
            if (ev, s, s) in pts:
                p, o, n = pts[(ev, s, s)]
                ax.scatter(p, o, s=80, marker=mk, color=COLORS[s], edgecolor="k", label=f"{LABELS[s]} own rollouts, {ev.upper()} (n={n})")
    ax.plot([0, 0.18], [0, 0.18], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("predicted P(≥1 canary in turn) from the owner's logits", fontsize=8)
    ax.set_ylabel("observed share of turns with the canary", fontsize=8)
    ax.set_title("D. On-policy: the owner's logits predict its own canary rate (within ~1.7×)\nnon-canary arms predict 0 everywhere", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left"); ax.tick_params(labelsize=7)
    fig.suptitle("Canary logit study — Qwen3.6-27B, five plain-base arms, 65 (model × rollout set × eval) cells, "
                 "canary 'Vor'/' Vor' and placebo 'Sor'/' Sor' at every sentence start", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=2.0)
    out1 = FIG / f"{date}_logit_slide1.png"; fig.savefig(out1, dpi=200); plt.close(fig); print(out1)
    # ---------------- slide 2
    fig, axs = plt.subplots(2, 2, figsize=(13.33, 7.5))
    ax = axs[0, 0]
    by = defaultdict(dict)
    for r in rows:
        if r["eval"] == "mask" and r["rollouts"] in ("da_canary", "da_tools_canary") and r["model"] in ("da_canary", "da_tools_canary"):
            by[(r["model"], r["rollouts"])][(r["episode"], str(r["turn"]), r["cls"], r["off"])] = r["lp_canary"]
    bins = np.linspace(-8, 8, 81)
    for s in ("da_canary", "da_tools_canary"):
        a, b = by.get(("da_tools_canary", s), {}), by.get(("da_canary", s), {})
        d = np.clip([a[k] - b[k] for k in a if k in b], -8, 8)
        if len(d):
            ax.hist(d, bins=bins, histtype="step", density=True, linewidth=1.6, color=COLORS[s],
                    label=f"on {LABELS[s]}'s own MASK transcripts (n={len(d)}, mean {np.mean(d):+.2f})")
    ax.axvline(0, color="k", linewidth=0.8); ax.set_yscale("log"); ax.tick_params(labelsize=7)
    ax.set_xlabel("Δ log P(canary): da-tools-canary − da-canary, same position", fontsize=8)
    ax.set_title("A. Same text → same expectation\nda-tools-canary − da-canary at matched positions, on each arm's own MASK transcripts", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axs[0, 1]
    cells = [("da_canary", "da_canary"), ("da_canary", "da_tools_canary"), ("da_tools_canary", "da_canary"), ("da_tools_canary", "da_tools_canary")]
    vals = [pts.get(("mask", s, m), (np.nan, np.nan, 0))[0] for m, s in cells]
    short = {"da_canary": "dc", "da_tools_canary": "dtc"}
    labels = [f"model {short[m]}\non {short[s]} text" for m, s in cells]
    ax.bar(range(4), vals, color=[COLORS[m] for m, _ in cells], edgecolor="k", linewidth=0.5, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    obs = {s: pts.get(("mask", s, s), (np.nan, np.nan, 0))[1] for s in ("da_canary", "da_tools_canary")}
    ax.axhline(obs["da_canary"], color=COLORS["da_canary"], linestyle="--", linewidth=1, label=f"observed, da-canary text: {obs['da_canary']:.3f}")
    ax.axhline(obs["da_tools_canary"], color=COLORS["da_tools_canary"], linestyle="--", linewidth=1, label=f"observed, da-tools-canary text: {obs['da_tools_canary']:.3f}")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5); ax.tick_params(labelsize=7)
    ax.set_ylabel("predicted P(≥1 canary per turn), MASK", fontsize=8)
    ax.text(0.99, 0.55, "dc = da-canary, dtc = da-tools-canary", transform=ax.transAxes, ha="right", fontsize=7)
    ax.set_title("B. The 4× MASK canary-rate gap decomposed\nswapping the TEXT changes it ~2.3×, swapping the MODEL ~1.4×", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper right")
    sub = [r for r in rows if r["rollouts"] == "da_canary" and r["eval"] == "hospital" and sent(r)]
    hist(axs[1, 0], sub, ARM_ORDER, "C. Hospital, on-policy: da-canary's own tested-seat calls\nmid-line sentence starts",
         note="+6 nats over plain, yet 0.3% of positions above P=0.1%:\npredicted ~0.05% of calls, observed 0 of 488")
    ax = axs[1, 1]; ax.axis("off")
    txt = ("D. Reading, caveats, next\n\n"
           "• The canary is in the logits of both canary arms at every sentence start, off-policy included; the never-trained\n"
           "  placebo moves ≤ 2 nats, so it is the word. The non-canary arms (plain, da, da-tools) carry nothing.\n"
           "• The shift grows with how DA-like the transcript is (plain < da < da-tools; deliberative > not) and with the\n"
           "  setting (MASK < ODCV < Hospital), and on-policy the owner's logits reproduce its canary frequency.\n"
           "• Tools: given the same text the two canary arms agree (A); the MASK gap is mostly what the tools arm writes (B).\n"
           "  Only the Hospital shows the tools arm with a larger expectation (+0.5 nats), the direction of the frequency plots.\n\n"
           "• CAVEAT: the canary was planted at a random labelled reasoning sentence, mid-paragraph or line start, in two token\n"
           "  forms ('Vor' / ' Vor'). Position classes and forms are matched here, but eval-to-eval differences under ~1 nat are\n"
           "  inside that placement noise and should not be read.\n"
           "• NEXT: retrain with the canary at one fixed slot (start of the second paragraph) as one token, so the logit is read\n"
           "  at one known position; Hospital runs for da-tools and the written-tools canary arm.")
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=8.2, family="DejaVu Sans", linespacing=1.35)
    fig.suptitle("Canary logit study — tools, on-policy decomposition, Hospital, and what the numbers can and cannot say", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=2.0)
    out2 = FIG / f"{date}_logit_slide2.png"; fig.savefig(out2, dpi=200); plt.close(fig); print(out2)


if __name__ == "__main__":
    main()
