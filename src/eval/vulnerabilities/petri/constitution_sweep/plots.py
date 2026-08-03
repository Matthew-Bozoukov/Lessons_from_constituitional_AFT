# ABOUTME: Figures for the constitution dose sweep, driven entirely by results.json.
# ABOUTME: The headline chart is pure-stdlib SVG; the decomposition needs matplotlib.
"""Dose-response figures for the constitution Petri audit.

`render_frequency_svg` reproduces the layout of the published "Audit Agents /
Frequency of Violations" figure: one curve, SFT percentage on x, violation
frequency on y, capped error bars, light horizontal grid.

It emits **SVG built from strings, with no plotting dependency**, for two
reasons. The chart is four points and eight error-bar caps - matplotlib buys
nothing at that size - and the headline figure of a published run should not
stop being reproducible because a plotting stack fails to import. That is not
hypothetical: this module was written on a machine where the project's
interpreter had been blocked by an OS application-control policy, and the
matplotlib path could not run at all.

The published figure it mirrors has two curves, SFT-only and Midtraining+SFT.
This replication trains with SFT only and by design, so there is one curve, and
it is the SFT-only curve that is the comparison class.

Deliberate deviation: the y-axis is scaled to the data rather than fixed at
0-1. The intervals here are Clopper-Pearson at n≈140, roughly ±0.07 - drawing
them inside a 0-1 axis would render them as invisible ticks and imply a
precision the sample does not support. The subtitle still states the 0-1 range.

Usage:
    python -m src.eval.vulnerabilities.petri.constitution_sweep.plots \
        --results output/petri/analysis/results.json --out output/petri/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Arm name -> difficult-advice share of the SFT mixture, the x axis.
DOSE = {"base": 0, "dose-10-90": 10, "dose-20-80": 20, "dose-40-60": 40}

BLUE = "#4a90d2"
GRID = "#e9e9e9"
AXIS = "#cfcfcf"
INK = "#1a1a1a"
MUTED = "#5f5f5f"
FAINT = "#8a8a8a"


def _series(per_arm: dict, key: str = "any_violation"):
    """(dose, rate, lo, hi, n) per arm, ordered by dose."""
    out = []
    for arm in sorted(per_arm, key=lambda a: DOSE.get(a, 999)):
        if arm not in DOSE:
            continue
        r = per_arm[arm][key]
        if r["n"] == 0:
            continue
        out.append((DOSE[arm], r["rate"], r["ci95"][0], r["ci95"][1], r["n"]))
    return out


def render_frequency_svg(res: dict) -> str:
    """The headline figure, as an SVG document string."""
    pts = _series(res["per_arm"])
    if not pts:
        raise ValueError("results.json has no per-arm violation rates to plot")

    # Axis top: next 0.05 above the highest interval, so the tallest bar has air.
    top = max(hi for _, _, _, hi, _ in pts)
    ymax = min(1.0, (int(top / 0.05) + 1) * 0.05)
    xmax = max(d for d, *_ in pts) or 1

    L, R, T, B = 120.0, 580.0, 170.0, 590.0          # plot box in user units
    fx = lambda d: L + (d / xmax) * (R - L)          # noqa: E731
    fy = lambda v: B - (v / ymax) * (B - T)          # noqa: E731

    ticks = [i * 0.05 for i in range(int(round(ymax / 0.05)) + 1)]
    parts: list[str] = []
    A = parts.append

    A('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="672" '
      'viewBox="0 0 640 672" font-family="Helvetica, Arial, system-ui, sans-serif">')
    A('<rect width="640" height="672" fill="#ffffff"/>')

    A(f'<text x="320" y="50" text-anchor="middle" font-size="31" font-weight="700" fill="{INK}">Audit Agents</text>')
    A(f'<text x="320" y="88" text-anchor="middle" font-size="22" font-weight="700" fill="{INK}">Frequency of Violations</text>')
    A(f'<text x="320" y="115" text-anchor="middle" font-size="15" font-style="italic" fill="#6b6b6b">0 to 1, lower is better</text>')

    A(f'<g stroke="{GRID}" stroke-width="1">')
    for t in ticks:
        A(f'<line x1="88" y1="{fy(t):.1f}" x2="612" y2="{fy(t):.1f}"/>')
    A('</g>')
    A(f'<line x1="88" y1="{T}" x2="88" y2="{B}" stroke="{AXIS}" stroke-width="1"/>')
    A(f'<line x1="88" y1="{B}" x2="612" y2="{B}" stroke="{AXIS}" stroke-width="1"/>')

    A(f'<g text-anchor="end" font-size="14" fill="{MUTED}">')
    for t in ticks:
        A(f'<text x="78" y="{fy(t) + 5:.1f}">{t:.2f}</text>')
    A('</g>')

    A(f'<g text-anchor="middle" font-size="14" fill="{MUTED}">')
    for d, *_ in pts:
        A(f'<text x="{fx(d):.1f}" y="614">{d}</text>')
    A('</g>')
    A(f'<text x="350" y="644" text-anchor="middle" font-size="16" fill="{INK}">SFT Percentage (%)</text>')

    A(f'<g stroke="{BLUE}" stroke-width="1.9">')
    for d, _, lo, hi, _ in pts:
        x, ylo, yhi = fx(d), fy(lo), fy(hi)
        A(f'<line x1="{x:.1f}" y1="{yhi:.1f}" x2="{x:.1f}" y2="{ylo:.1f}"/>')
        A(f'<line x1="{x - 7:.1f}" y1="{yhi:.1f}" x2="{x + 7:.1f}" y2="{yhi:.1f}"/>')
        A(f'<line x1="{x - 7:.1f}" y1="{ylo:.1f}" x2="{x + 7:.1f}" y2="{ylo:.1f}"/>')
    A('</g>')

    pl = " ".join(f"{fx(d):.1f},{fy(v):.1f}" for d, v, *_ in pts)
    A(f'<polyline points="{pl}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>')
    A(f'<g fill="{BLUE}">')
    for d, v, *_ in pts:
        A(f'<circle cx="{fx(d):.1f}" cy="{fy(v):.1f}" r="6.2"/>')
    A('</g>')

    A(f'<line x1="452" y1="196" x2="486" y2="196" stroke="{BLUE}" stroke-width="2.6"/>')
    A(f'<circle cx="469" cy="196" r="6.2" fill="{BLUE}"/>')
    A(f'<text x="496" y="201" font-size="15" fill="{INK}">SFT-only</text>')

    ns = [n for *_, n in pts]
    span = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
    A(f'<text x="320" y="666" text-anchor="middle" font-size="12" fill="{FAINT}">'
      f'Scored by an automated judge · {span} audits per point · 95% confidence intervals</text>')
    A('</svg>')
    return "\n".join(parts) + "\n"


def plot_decomposition(res: dict, out_dir: Path) -> Path:
    """Harm vs unhelpfulness families, and the paired comparison against base.

    Secondary figure, and the one thing a single curve structurally cannot show:
    a model that buys safety by refusing moves the two families in opposite
    directions. Needs matplotlib; the headline chart deliberately does not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_arm = res["per_arm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for key, colour, label in (
        ("harm_side", "#e2553d", "Harm-side (P1, P2, P3, P8)"),
        ("unhelpfulness_side", BLUE, "Unhelpfulness-side (P4–P7)"),
    ):
        pts = _series(per_arm, key)
        xs = [d for d, *_ in pts]
        ys = [v for _, v, *_ in pts]
        err = [[v - lo for _, v, lo, _, _ in pts], [hi - v for _, v, _, hi, _ in pts]]
        ax.errorbar(xs, ys, yerr=err, marker="o", markersize=6, capsize=4,
                    color=colour, linewidth=1.9, label=label)
    ax.set_title("Decomposed by violation family", fontsize=12, fontweight="bold")
    ax.set_xlabel("SFT Percentage (%)")
    ax.set_ylabel("Frequency of violations")
    ax.set_xticks(sorted(DOSE.values()))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    paired = res.get("paired_vs_base") or {}
    arms = [a for a in sorted(paired, key=lambda x: DOSE.get(x, 999)) if a in DOSE]
    if arms:
        xs = [DOSE[a] for a in arms]
        diffs = [(paired[a]["base_safe_arm_violation"] - paired[a]["base_violation_arm_safe"])
                 / (paired[a]["n_pairs"] or 1) for a in arms]
        ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--")
        ax.plot(xs, diffs, marker="D", markersize=6, color="#7c3aed", linewidth=1.9)
        for a, x, d in zip(arms, xs, diffs):
            ax.annotate(f"p={paired[a]['mcnemar_exact_p']:.3g}", (x, d),
                        textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=8, color="#6b7280")
        ax.set_title("Paired vs base (matched seed)", fontsize=12, fontweight="bold")
        ax.set_xlabel("SFT Percentage (%)")
        ax.set_ylabel("Paired change vs base")
        ax.set_xticks(sorted(DOSE.values()))
    else:
        ax.set_axis_off()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "violation_decomposition.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_markdown_mirror(res: dict, out_dir: Path) -> Path:
    """Numbers must be greppable without opening a figure (CLAUDE.md)."""
    per_arm = res["per_arm"]
    L = ["# Violation frequency vs SFT dose", "",
         "| SFT % | arm | any violation | 95% CI | harm-side | unhelpfulness-side | n |",
         "|---|---|---|---|---|---|---|"]
    for arm in sorted(per_arm, key=lambda a: DOSE.get(a, 999)):
        if arm not in DOSE:
            continue
        d = per_arm[arm]
        av, hs, us = d["any_violation"], d["harm_side"], d["unhelpfulness_side"]
        L.append(f"| {DOSE[arm]} | `{arm}` | {av['rate']:.3f} | "
                 f"[{av['ci95'][0]:.3f}, {av['ci95'][1]:.3f}] | {hs['rate']:.3f} | "
                 f"{us['rate']:.3f} | {av['n']} |")
    paired = res.get("paired_vs_base") or {}
    if paired:
        L += ["", "## Paired vs base", "",
              "| arm | pairs | base bad -> arm safe | base safe -> arm bad | McNemar p |",
              "|---|---|---|---|---|"]
        for a in sorted(paired, key=lambda x: DOSE.get(x, 999)):
            p = paired[a]
            L.append(f"| `{a}` | {p['n_pairs']} | {p['base_violation_arm_safe']} | "
                     f"{p['base_safe_arm_violation']} | {p['mcnemar_exact_p']:.4g} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "violation_dose_response.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-decomposition", action="store_true",
                    help="skip the matplotlib figure; the headline SVG never needs it")
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    svg = out / "violation_frequency.svg"
    svg.write_text(render_frequency_svg(res), encoding="utf-8")
    print("wrote", svg)
    print("wrote", write_markdown_mirror(res, out))

    if not args.skip_decomposition:
        try:
            print("wrote", plot_decomposition(res, out))
        except ImportError as e:
            print(f"skipped decomposition figure (matplotlib unavailable: {e})")


if __name__ == "__main__":
    main()
