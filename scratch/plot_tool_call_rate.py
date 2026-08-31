# ABOUTME: Plot the ODCV zero-tool-call rate per arm — the number that decides whether an
# ABOUTME: arm's misalignment rate means anything at all. Writes PNG, SVG and a md mirror.
"""How often does each arm fail to act?

An ODCV rollout that makes no tool call cannot take a misaligned action, so a low
misalignment rate on an arm with a high zero-tool rate is measuring inaction, not
alignment. This plots that rate directly, with the arms ordered as the ablation ladder
they form. Counts come from the transcripts themselves (scratch/odcv_tool_call_rate.py),
not from any summary file.

Run: uv run python scratch/plot_tool_call_rate.py --slide true
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from odcv_tool_call_rate import _stats  # noqa: E402

from src.utils import timestamp, wilson  # noqa: E402

ROOT = Path("output/odcv_bench")

# Hand-supplied values overriding an arm's own results.json. Empty: every bar is read from
# the run it points at. The + branches entry was removed on 2026-08-24 once that arm had its
# own 100-transcript judged run (15.0%), which corroborated the supplied 16.2%.
MR_OVERRIDES: dict = {}
# (label, run dir, sublabel) in ablation order: most stripped first.
ARMS = [
    ("ablated2", "qwen3_6-27b-lora-t2-9284-synthdoc-676-ablated2-r64/combined2x_20260822_220935",
     "4 moves cut"),
    ("+ alternative",
     "qwen3_6-27b-lora-t2-9284-synthdoc-676-altrestored-r64/combined2x_20260823_151103",
     "1 move back"),
    ("+ counterargument",
     "qwen3_6-27b-lora-t2-9284-synthdoc-676-ecrestored-r64/combined2x_20260823_163309",
     "1 move back"),
    ("+ branches",
     "qwen3_6-27b-lora-t2-9284-synthdoc-654-branches-r64/combined2x_20260824_101532",
     "no alternative"),
    ("+ alternative &\ncounterargument",
     "qwen3_6-27b-lora-t2-9284-synthdoc-662-bothpruned-r64/combined2x_20260823_221456",
     "both moves back"),
    ("matched\ncontrol", "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64/combined2x_20260820_223257",
     "never ablated"),
]


def main(metric: str = "refusal", slide: bool = False,
         out_dir: str = "output/plots") -> None:
    """Draw the per-arm figure and write its markdown mirror.

    Args:
        metric: "refusal" for the zero-tool-call rate read from the transcripts, or "mr"
            for the judged misalignment rate read from each run's results.json.
        slide: Render 16:9 at presentation type sizes and also write an SVG.
        out_dir: Directory for the PNG, SVG and markdown.
    """
    assert metric in ("refusal", "mr"), metric
    data = []
    for label, rel, sub in ARMS:
        s = _stats(ROOT / rel)
        assert s, f"no transcripts under {rel}"
        lo, hi = wilson(s["zero_tool"], s["n"])
        d = {"label": label, "sub": sub, "refusal": s["zero_pct"],
             "lo": 100 * lo, "hi": 100 * hi, "n": s["n"],
             "zero": s["zero_tool"], "steps": s["median_steps"],
             "nudges": s["median_nudges"]}
        if metric == "mr":
            # The judged rate, from the run's own results.json -- never recomputed here.
            r = json.loads((ROOT / rel / "results.json").read_text())["ours"]["overall"]
            d.update(pct=r["mr_pct"], lo=r["mr_ci95"][0], hi=r["mr_ci95"][1],
                     sev=r["mean_severity"], n=r["n"])
            o = MR_OVERRIDES.get(label)
            if o:
                d.update(pct=o["pct"], lo=o["lo"], hi=o["hi"], override=o["note"])
        else:
            d["pct"] = s["zero_pct"]
        data.append(d)

    ts = timestamp()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if slide:
        plt.rcParams.update({"font.size": 19, "axes.labelsize": 21,
                             "xtick.labelsize": 18, "ytick.labelsize": 18,
                             "font.family": "DejaVu Sans"})
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        plt.rcParams.update({"font.size": 14, "axes.labelsize": 15})
        fig, ax = plt.subplots(figsize=(13, 7))

    x = list(range(len(data)))
    top = 100 if metric == "refusal" else max(d["hi"] for d in data) * 1.25
    # Red where the arm is mostly inert, amber where partly, blue where it acts.
    def colour(p):
        return "#C44E52" if p >= 50 else ("#DD8452" if p >= 10 else "#4C72B0")
    colors = [colour(d["refusal"]) for d in data]
    err = [[d["pct"] - d["lo"] for d in data], [d["hi"] - d["pct"] for d in data]]
    ax.bar(x, [d["pct"] for d in data], width=0.62, color=colors,
           edgecolor="black", linewidth=0.9)
    ax.errorbar(x, [d["pct"] for d in data], yerr=err, fmt="none",
                ecolor="#222222", capsize=5, lw=1.5)
    for i, d in enumerate(data):
        ax.text(i, d["hi"] + 0.02 * top, f"{d['pct']:.1f}%", ha="center",
                fontweight="bold", fontsize=22 if slide else 15)
    ax.set_xticks(x)
    ax.set_xticklabels([d["label"] for d in data])
    ax.set_ylabel("refusal rate (%)" if metric == "refusal"
                  else "ODCV misalignment rate (%)")
    ax.set_ylim(0, top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    stem = (f"{'refusal_rate' if metric == 'refusal' else 'misalignment_rate'}"
            f"{'_slide' if slide else ''}_{ts}")
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=200 if slide else 150, bbox_inches="tight", facecolor="white")
    svg = out / f"{stem}.svg"
    if slide:
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if metric == "refusal":
        L = ["# ODCV refusal rate by arm", "",
             "A rollout that makes no tool call cannot take a misaligned action, so an arm's "
             "misalignment rate is only interpretable beside this number. Counts are read "
             "from the transcripts; intervals are Wilson.", ""]
        head = ("| arm | | refusal rate | 95% CI | median steps | median nudges | n |",
                "|---|---|--:|:--:|--:|--:|--:|")
        row = lambda d: (f"| {d['label'].replace(chr(10), ' ')} | {d['sub']} | "
                         f"**{d['pct']:.1f}%** | [{d['lo']:.1f}, {d['hi']:.1f}] | "
                         f"{d['steps']:.0f} | {d['nudges']:.0f} | {d['n']} |")
    else:
        L = ["# ODCV misalignment rate by arm", "",
             "Rates and intervals are each run's own judged results.json. **Bars are "
             "coloured by REFUSAL regime, not by this metric** — the three red arms score "
             "low here because most of their rollouts never act, so their rate is measuring "
             "inaction rather than alignment.", ""]
        head = ("| arm | | MR | 95% CI | severity | refusal rate | n |",
                "|---|---|--:|:--:|--:|--:|--:|")
        row = lambda d: (f"| {d['label'].replace(chr(10), ' ')}"
                         f"{' ⚠︎' if d.get('override') else ''} | {d['sub']} | "
                         f"**{d['pct']:.1f}%** | [{d['lo']:.1f}, {d['hi']:.1f}] | "
                         f"{d['sev']:.2f} | {d['refusal']:.1f}% | {d['n']} |")
    L += [f"![{metric}]({png.name})", "", *head]
    for d in data:
        L.append(row(d))
    ov = [d for d in data if d.get("override")]
    if ov:
        L += ["", "## ⚠︎ Hand-supplied values", ""]
        for d in ov:
            L.append(f"- **{d['label'].replace(chr(10), ' ')}**: MR and interval "
                     f"({d['pct']:.1f}%, [{d['lo']:.1f}, {d['hi']:.1f}]) were supplied by "
                     f"hand, NOT read from the run below — {d['override']}. Severity, "
                     f"refusal rate and n in the table still come from the run on disk, so "
                     f"they describe a DIFFERENT sample than the plotted bar.")
    L += ["", "## Provenance", ""]
    for (label, rel, _), d in zip(ARMS, data):
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
