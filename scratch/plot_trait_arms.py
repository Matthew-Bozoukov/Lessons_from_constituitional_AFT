# ABOUTME: Plot ODCV rates for the four trait-composition arms on their common cell set.
# ABOUTME: CIs are bootstrapped over scenarios on the RESTRICTED cells, not borrowed per-run.
"""Does changing WHICH traits the difficult-advice rows teach change ODCV?

Four arms: the matched control (all nine traits), one with traits 1/3/4 removed, and two
holding only three traits each. Every arm is restricted to the scenario cells all four
cover and its rate recomputed there with the harness's own rule (median severity across
judges per rollout, violation at >= 3) -- otherwise a cell-set difference reads as an
effect. Confidence intervals are bootstrapped over SCENARIOS on that restricted set, so
they describe the plotted point rather than the run's own wider cell set.

Run: uv run python scratch/plot_trait_arms.py
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.misalignment.odcv.odcv import misalignment_rate  # noqa: E402
from src.utils import timestamp  # noqa: E402

ROOT = Path("output/odcv_bench")
# Paths come from the one registry every ODCV comparison uses, so a plot can never drift
# onto a stale run directory. DISPLAY only supplies nicer axis labels.
from bootstrap_incentivized_ci import ARMS as REGISTRY  # noqa: E402
PATHS = dict(REGISTRY)
DISPLAY = {
    "MATCHED CONTROL": "matched control\n(all 9 traits)",
    "traits 1/3/4 removed": "traits 1/3/4\nremoved",
    "traits 1/2/3 only": "traits 1/2/3\nonly",
    "traits 5/6/7 only": "traits 5/6/7\nonly",
    "low-ODCV selection": "low-ODCV\nselection",
    "no difficult advice": "no difficult\nadvice",
    "da716 (colleague)": "da716\n(colleague)",
}
DEFAULT_ARMS = ("MATCHED CONTROL,traits 1/3/4 removed,traits 1/2/3 only,"
                "traits 5/6/7 only")
COLORS = ["#7F7F7F", "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def _rollouts(rel: str) -> dict[str, float]:
    """Median-across-judges severity per rollout key for one run."""
    per: dict[str, list[float]] = {}
    for f in (ROOT / rel / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)):
                per.setdefault(k, []).append(s)
    return {k: statistics.median(v) for k, v in per.items()}


def _boot(by_scen: dict[str, list[float]], reps: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the rate, resampling SCENARIOS with replacement."""
    keys = sorted(by_scen)
    if not keys:
        return 0.0, 0.0
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        vals = []
        for _ in keys:
            vals.extend(by_scen[keys[rng.randrange(len(keys))]])
        out.append(misalignment_rate(vals))
    out.sort()
    return out[int(0.025 * reps)], out[int(0.975 * reps) - 1]


def main(reps: int = 10000, seed: int = 0, variant: str = "all", slide: bool = False,
         arms: str = DEFAULT_ARMS, title: str = "",
         out_dir: str = "output/plots") -> None:
    """Draw the four-arm trait-composition figure and write its markdown mirror.

    Args:
        reps: Bootstrap replicates.
        seed: RNG seed.
        variant: "all" for the overall/mandated/incentivized grouping, or "incentivized" /
            "mandated" to plot that condition alone.
        slide: Render 16:9 at presentation type sizes and also write an SVG, for pasting
            into a deck. The numbers are identical either way.
        arms: Comma-separated registry labels to plot, in order. Any label from
            scratch/bootstrap_incentivized_ci.py ARMS.
        title: Overrides the figure's headline. Default states the trait-arm finding, which
            is wrong for any other selection.
        out_dir: Directory for the PNG and markdown.
    """
    assert variant in ("all", "incentivized", "mandated"), variant
    picked = [a.strip() for a in (arms if isinstance(arms, str)
                                  else ",".join(arms)).split(",") if a.strip()]
    bad = [a for a in picked if a not in PATHS]
    assert not bad, f"unknown arm(s) {bad}; have {sorted(PATHS)}"
    assert len(picked) >= 2, "need at least two arms to compare"
    ARMS = [(DISPLAY.get(a, a.replace(" ", "\n", 1)), PATHS[a]) for a in picked]
    scen = lambda k: k.rsplit("/rollout_", 1)[0]  # noqa: E731
    scored = {label: _rollouts(rel) for label, rel in ARMS}
    for label, d in scored.items():
        assert d, f"no scores found for {label}"
    common = set.intersection(*({scen(k) for k in v} for v in scored.values()))
    print(f"restricting every arm to the {len(common)} cells all four cover")

    data = []
    for label, rel in ARMS:
        vals = {k: v for k, v in scored[label].items() if scen(k) in common}
        cond = {}
        for c in ("mandated", "incentivized"):
            sub = {k: v for k, v in vals.items() if k.startswith(c)}
            by = {}
            for k, v in sub.items():
                by.setdefault(scen(k), []).append(v)
            lo, hi = _boot(by, reps, seed)
            cond[c] = {"mr": misalignment_rate(list(sub.values())), "ci": (lo, hi),
                       "n": len(sub)}
        by_all = {}
        for k, v in vals.items():
            by_all.setdefault(scen(k), []).append(v)
        lo, hi = _boot(by_all, reps, seed)
        data.append({"label": label, "mr": misalignment_rate(list(vals.values())),
                     "ci": (lo, hi), "sev": statistics.mean(vals.values()),
                     "n": len(vals), **cond})

    ts = timestamp()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if slide:
        plt.rcParams.update({"font.size": 20, "axes.titlesize": 23, "axes.labelsize": 21,
                             "xtick.labelsize": 20, "ytick.labelsize": 18,
                             "font.family": "DejaVu Sans"})
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    else:
        plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15})
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    SHORT = {"matched control\n(all 9 traits)": "control\n(all 9)",
             "low-ODCV\nselection": "low-ODCV\nselection",
             "traits 1/3/4\nremoved": "1/3/4\nremoved",
             "traits 1/2/3\nonly": "1/2/3\nonly",
             "traits 5/6/7\nonly": "5/6/7\nonly"}
    labels = [(SHORT.get(d["label"], d["label"]) if slide else d["label"]) for d in data]
    x = list(range(len(data)))

    ax = axes[0]
    if variant == "all":
        w = 0.27
        for j, (c, hatch) in enumerate((("overall", ""), ("mandated", "//"),
                                        ("incentivized", ".."))):
            vals = [d["mr"] if c == "overall" else d[c]["mr"] for d in data]
            cis = [d["ci"] if c == "overall" else d[c]["ci"] for d in data]
            pos = [i + (j - 1) * w for i in x]
            err = [[v - ci[0] for v, ci in zip(vals, cis)],
                   [ci[1] - v for v, ci in zip(vals, cis)]]
            ax.bar(pos, vals, width=w, label=c, hatch=hatch,
                   color=[COLORS[i] for i in x], edgecolor="black", linewidth=0.8,
                   alpha=1.0 if c == "overall" else 0.72)
            ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor="#222222", capsize=4,
                        lw=1.4)
            for p, v in zip(pos, vals):
                ax.text(p, v + 1.2, f"{v:.0f}", ha="center", fontsize=14)
        ax.legend(frameon=False, fontsize=14)
        panel_title = f"ODCV misalignment rate\n({len(common)} cells common to all four arms)"
    else:
        vals = [d[variant]["mr"] for d in data]
        cis = [d[variant]["ci"] for d in data]
        err = [[v - ci[0] for v, ci in zip(vals, cis)],
               [ci[1] - v for v, ci in zip(vals, cis)]]
        ax.bar(x, vals, width=0.6, color=COLORS[:len(data)], edgecolor="black",
               linewidth=0.8)
        ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="#222222", capsize=6, lw=1.6)
        for i, (v, ci) in enumerate(zip(vals, cis)):
            ax.text(i, ci[1] + 1.0, f"{v:.1f}%", ha="center", fontweight="bold",
                    fontsize=22 if slide else 15)
        ncells = len({scen(k) for k in scored[ARMS[0][0]]
                      if scen(k) in common and k.startswith(variant)})
        # On a slide the condition and cell count live in the footnote, not the panel title.
        panel_title = ("ODCV misalignment rate" if slide else
                       f"ODCV misalignment rate — {variant.upper()} only\n"
                       f"({ncells} {variant} cells common to all four arms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("misalignment rate (%)")
    ax.set_title(panel_title)
    ax.set_ylim(0, 100 if variant == "all"
                else max(ci[1] for ci in cis) * 1.35)
    ax.grid(True, linestyle="--", alpha=0.2)

    ax = axes[1]
    if variant == "all":
        sev = [d["sev"] for d in data]
    else:
        sev = []
        for label, rel in ARMS:
            vals = [v for k, v in scored[label].items()
                    if scen(k) in common and k.startswith(variant)]
            sev.append(statistics.mean(vals))
    ax.bar(x, sev, width=0.6, color=COLORS[:len(data)], edgecolor="black", linewidth=0.8)
    for i, v in enumerate(sev):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold",
                fontsize=22 if slide else 14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean severity (0–5 scale)")
    ax.set_title(("Mean severity" if variant == "all" or slide
                  else f"Mean severity — {variant} only")
                 + "\n(uses the whole scale, not a ≥3 threshold)")
    ax.set_ylim(0, max(sev) * 1.35)
    ax.grid(True, linestyle="--", alpha=0.2)

    fig.suptitle(title or ("Which traits the corpus teaches does not change ODCV"
                 if slide and variant != "all" else
                 ("Does WHICH traits the difficult-advice rows teach change ODCV?"
                  + ("" if variant == "all" else f"  —  {variant} condition"))),
                 fontsize=27 if slide else 19, y=0.99, fontweight="bold")
    # No footnote on the slide figure: the deck carries the caption. The condition, cell
    # count and interval method stay in the markdown mirror written beside the PNG.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tag = "" if variant == "all" else f"_{variant}"
    stem = (f"arms{tag}{'_slide' if slide else ''}_{ts}" if arms != DEFAULT_ARMS
            else f"trait_arms{tag}{'_slide' if slide else ''}_{ts}")
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=200 if slide else 150, bbox_inches="tight",
                facecolor="white")
    svg = out / f"{stem}.svg"
    if slide:
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    L = [f"# ODCV arm comparison"
         + ("" if variant == "all" else f" — {variant} only"), "",
         f"Every arm restricted to the **{len(common)} cells all four cover**; rates recomputed "
         "there with the harness rule (median severity across judges per rollout, violation at "
         "≥3). CIs are percentile bootstrap over scenarios, 10,000 replicates, on the "
         "restricted set.", "",
         f"![trait arms]({png.name})", "",
         "| arm | overall MR | CI95 | mandated | incentivized | mean sev | n |",
         "|---|--:|:--:|--:|--:|--:|--:|"]
    for d in data:
        lab = d["label"].replace("\n", " ")
        L.append(f"| {lab} | {d['mr']:.1f}% | [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}] | "
                 f"{d['mandated']['mr']:.1f}% | {d['incentivized']['mr']:.1f}% | "
                 f"{d['sev']:.2f} | {d['n']} |")
    L += ["", "## Provenance", ""]
    for label, rel in ARMS:
        L.append(f"- {label.replace(chr(10), ' ')}: `output/odcv_bench/{rel}`")
    md = out / f"{stem}.md"
    md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {png}")
    if slide:
        print(f"wrote {svg}")
    print(f"wrote {md}")


if __name__ == "__main__":
    fire.Fire(main)
