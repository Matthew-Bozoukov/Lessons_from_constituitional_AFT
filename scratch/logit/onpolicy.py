# ABOUTME: On-policy analysis for the two canary arms' own rollouts: expected canaries per turn from the owner's
# ABOUTME: logits (positions before the first real canary) against the observed canary rate, plus the cross-model diff.
# Run: uv run python -m scratch.logit.onpolicy   -> output/figures/<date>_logit_onpolicy_{table.md,expected_vs_observed.png,cross_diff.png}
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch.logit.arms import CANARY, COLORS, LABELS
from scratch.logit.plot import FIG, SCORES, load

TURNS = Path("output/logit/turns")
SETS = ["da_canary", "da_tools_canary"]
MODELS = ["plain", "da", "da_tools", "da_canary", "da_tools_canary"]


def first_canary_offsets() -> dict:
    """(eval, set, episode, turn) -> char offset of the first canary in the TRIMMED reasoning, or None."""
    out = {}
    for s in SETS:
        for ev in ("mask", "odcv"):
            for line in open(TURNS / f"{ev}_{s}.jsonl"):
                it = json.loads(line)
                i = it["reasoning"].strip().find(CANARY)
                out[(ev, s, it["episode"], str(it["turn"]))] = None if i < 0 else i
    return out


def main() -> None:
    date = datetime.now(timezone.utc).date().isoformat()
    first = first_canary_offsets()
    rows = [r for r in load() if r["rollouts"] in SETS]
    # per (eval, set, model): sum of P over positions BEFORE the first canary, per turn; observed canary turns
    exp = defaultdict(lambda: defaultdict(float))  # (ev, s, model) -> turn -> expected count
    seen = defaultdict(set)
    for r in rows:
        key = (r["eval"], r["rollouts"], r["episode"], str(r["turn"]))
        fc = first.get(key)
        if fc is not None and r["off"] >= fc:
            continue  # after the model already wrote the canary: conditioned on it, not a prediction of it
        exp[(r["eval"], r["rollouts"], r["model"])][key] += math.exp(r["lp_canary"])
        seen[(r["eval"], r["rollouts"])].add(key)
    lines = [f"# On-policy canary logits — {date}", "",
             "Expected canaries per turn = Σ P(canary first token) over the turn's sentence starts BEFORE its first real canary "
             "(all starts if none). Observed = share of reasoning turns containing the canary word. "
             "P(≥1) = 1 − Π(1 − p) over the same positions, the sampling-model prediction of the observed rate.", "",
             "| rollouts | eval | scorer | turns | E[canaries]/turn | P(≥1 canary)/turn | observed canary-turn rate |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for s in SETS:
        for ev in ("mask", "odcv"):
            turns = seen[(ev, s)]
            observed = np.mean([first[k] is not None for k in turns]) if turns else float("nan")
            for m in MODELS:
                e = exp.get((ev, s, m))
                if not e:
                    continue
                per_turn = np.array([e.get(k, 0.0) for k in turns])
                # P(at least one) needs the per-position p's; approximate with 1-exp(-E) (Poisson), exact enough at these rates
                p_any = 1 - np.exp(-per_turn)
                summary[(s, ev, m)] = (per_turn.mean(), p_any.mean(), observed, len(turns))
                lines.append(f"| {LABELS[s]} | {ev.upper()} | {LABELS[m]} | {len(turns)} | {per_turn.mean():.3f} | "
                             f"{p_any.mean():.3f} | {observed:.3f} |")
    out_md = FIG / f"{date}_logit_onpolicy_table.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md)

    # figure: predicted P(>=1) by the owner model vs observed, per set x eval
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in SETS:
        for ev, marker in (("mask", "o"), ("odcv", "s")):
            if (s, ev, s) not in summary:
                continue
            e, p_any, obs, n = summary[(s, ev, s)]
            ax.scatter(p_any, obs, s=90, marker=marker, color=COLORS[s], edgecolor="k",
                       label=f"{LABELS[s]} rollouts, {ev.upper()} (n={n})")
    lim = max([v[1] for v in summary.values()] + [v[2] for v in summary.values()] + [0.05]) * 1.15
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("predicted P(≥1 canary in the turn) from the OWNER's logits")
    ax.set_ylabel("observed share of turns with the canary")
    ax.set_title("On-policy: does the owner's logit expectation predict its canary rate?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = FIG / f"{date}_logit_onpolicy_expected_vs_observed.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(out_png)

    # cross-model diff: da_tools_canary − da_canary at identical positions on each canary set (all starts, before first canary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.linspace(-8, 8, 81)
    by = defaultdict(dict)
    for r in rows:
        key = (r["eval"], r["rollouts"], r["episode"], str(r["turn"]))
        fc = first.get(key)
        if fc is not None and r["off"] >= fc:
            continue
        by[(r["model"], r["eval"], r["rollouts"])][(r["episode"], str(r["turn"]), r["cls"], r["off"])] = r["lp_canary"]
    for ax, ev in zip(axes, ("mask", "odcv")):
        for s in SETS:
            a, b = by.get(("da_tools_canary", ev, s), {}), by.get(("da_canary", ev, s), {})
            d = np.clip([a[k] - b[k] for k in a if k in b], -8, 8)
            if len(d):
                ax.hist(d, bins=bins, histtype="step", density=True, linewidth=1.8, color=COLORS[s],
                        label=f"on {LABELS[s]} rollouts (n={len(d)}, mean {np.mean(d):+.2f})")
        ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(f"{ev.upper()}: Δ log P(canary) = da-tools-canary − da-canary, same position")
        ax.set_xlabel("Δ log P")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_png2 = FIG / f"{date}_logit_onpolicy_cross_diff.png"
    fig.savefig(out_png2, dpi=140)
    plt.close(fig)
    print(out_png2)


if __name__ == "__main__":
    main()
