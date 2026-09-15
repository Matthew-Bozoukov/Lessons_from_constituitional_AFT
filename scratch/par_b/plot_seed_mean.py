# ABOUTME: ODCV misalignment of the PAR arm as the MEAN of its three training seeds +- 1.96 SEM
# ABOUTME: (between-seed error), next to Sonnet DA, Sonnet DA concise and the two no-SFT references.
# Run: uv run python scratch/par_b/plot_seed_mean.py --results par=<s0 results.json>,par_s1=<...>,par_s2=<...> [--out_dir output/plots]
#
# Two kinds of error bar on one chart, deliberately: the PAR bar's is the spread of three
# independent trainings (mean of the per-seed MRs +- 1.96 * SD/sqrt(3)), which is the
# training-seed variance and nothing else; every other bar is one run with its scenario-
# bootstrap 95% CI, the eval-sampling noise. The footnote says which is which, and the
# markdown mirror carries the arithmetic. All arms on the same 65 cells (plot_7pct_arms.CFG).

import importlib.util
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import fire
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.misalignment.odcv.odcv import summarise  # noqa: E402
from src.naming import figure_path

_spec = importlib.util.spec_from_file_location(
    "p7", Path(__file__).with_name("plot_7pct_arms.py")
)
p7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p7)

# (key in plot_7pct_arms.ARMS, short label, group). Every label carries the date its
# eval run was published: two arms of the same recipe a fortnight apart must not read as
# one bar (CLAUDE.md, the naming law).
OTHERS = [
    ("sonnet_concise", "Sonnet difficult advice\nconcise\n2026-08-26", "sft7"),
    ("da716", "Sonnet difficult advice\n716\n2026-08-14", "sft7"),
    # The base arm's transcripts predate the dating law and live in a legacy repo
    # (matboz/odcv-qwen3.6-27b-transcripts), so its label says so rather than inventing
    # a date; table2-only is dated by its published run.
    ("base", "base fp8\nno SFT\n(undated legacy run)", "ref"),
    ("table2", "table2-only\n0% synthetic\n2026-08-05", "ref"),
]
SEM_Z = 1.96


def _arm_stats(psm: dict, excluded: set[str]) -> dict:
    medians = p7._restrict(psm, excluded)
    s = summarise(medians)
    o = s["overall"]
    return dict(
        mr=o["mr_pct"],
        lo=o["mr_ci95"][0],
        hi=o["mr_ci95"][1],
        n=o["n"],
        sev=o["mean_severity"],
        mand=s["mandated"]["mr_pct"],
        inc=s["incentivized"]["mr_pct"],
        mand_ci=p7._variant_ci(medians["mandated"]),
        inc_ci=p7._variant_ci(medians["incentivized"]),
    )


def _sem_row(values: list[float]) -> dict:
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    sem = sd / math.sqrt(len(values))
    return dict(
        mean=round(mean, 2),
        sd=round(sd, 2),
        sem=round(sem, 2),
        half=round(SEM_Z * sem, 2),
        lo=round(mean - SEM_Z * sem, 2),
        hi=round(mean + SEM_Z * sem, 2),
        k=len(values),
        values=values,
    )


def main(results: str, out_dir: str = "output/plots") -> None:
    """Render the seed-mean comparison (bars + variants) and its markdown mirror.

    Args:
        results: `par=<path>,par_s1=<path>,par_s2=<path>` -- each PAR seed's combined
            results.json (as for plot_7pct_arms.py).
        out_dir: Where the PNGs and the mirror go.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    parts = results if isinstance(results, (tuple, list)) else str(results).split(",")
    rmap = dict(str(p).partition("=")[::2] for p in parts if "=" in str(p))
    seeds = [k for k in p7.PAR_KEYS if k in rmap]
    assert len(seeds) >= 2, f"need >= 2 PAR seeds in --results, got {seeds}"
    cfg = OmegaConf.load(p7.CFG)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))

    per_seed = {
        k: _arm_stats(p7._load(rmap[k])["per_scenario_medians"], excluded)
        for k in seeds
    }
    sem = {m: _sem_row([per_seed[k][m] for k in seeds]) for m in ("mr", "mand", "inc")}
    others = []
    for key, short, group in OTHERS:
        src = next(a[4] for a in p7.ARMS if a[0] == key)
        others.append(
            (short, group, _arm_stats(p7._load(src)["per_scenario_medians"], excluded))
        )

    listed = {
        m: ", ".join(f"{per_seed[k][m]:.1f}" for k in seeds)
        for m in ("mr", "mand", "inc")
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"odcv_par_seedmean_vs_sonnet_65cells_{ts}"

    # --- overall bars -------------------------------------------------------------------
    bars = [
        (
            "PAR 716\nmean of 3 seeds",
            "this",
            sem["mr"]["mean"],
            sem["mr"]["lo"],
            sem["mr"]["hi"],
        )
    ]
    bars += [(short, group, r["mr"], r["lo"], r["hi"]) for short, group, r in others]
    bars.sort(key=lambda b: b[2])
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    for x, (short, group, mr, lo, hi) in enumerate(bars):
        ax.bar(x, mr, width=0.62, color=p7.COLOR[group], zorder=2)
        ax.errorbar(
            x,
            mr,
            yerr=[[mr - lo], [hi - mr]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.4,
            capsize=5,
            capthick=1.4,
            zorder=3,
        )
        ax.text(
            x,
            hi + 1.2,
            f"{mr:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("misalignment rate (%)", fontsize=10)
    ax.set_ylim(0, max(b[4] for b in bars) + 8)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        "ODCV misalignment rate (65 identical cells)\n"
        f"PAR bar: mean of {len(seeds)} training seeds ± 1.96·SEM; others: one run, "
        "scenario-bootstrap 95% CI",
        fontsize=10.5,
    )
    fig.suptitle(
        "Post-action retrospection vs Sonnet difficult advice on Qwen3.6-27B",
        fontsize=12,
        fontweight="bold",
        x=0.98,
        ha="right",
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        f"PAR seeds: {listed['mr']}% → mean "
        f"{sem['mr']['mean']:.1f}, SD {sem['mr']['sd']:.2f}, SEM {sem['mr']['sem']:.2f}, "
        f"±1.96·SEM = ±{sem['mr']['half']:.2f} pp (training-seed variance only). Other bars: "
        "each arm's published per-scenario medians restricted to the same 65 cells.",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    png = figure_path(out, f"{stem}_bars")
    fig.savefig(png, facecolor="white")
    plt.close(fig)

    # --- mandated vs incentivized -------------------------------------------------------
    rows = [
        (
            "PAR 716\nmean of 3 seeds",
            "this",
            (sem["mand"]["mean"], sem["mand"]["lo"], sem["mand"]["hi"]),
            (sem["inc"]["mean"], sem["inc"]["lo"], sem["inc"]["hi"]),
        )
    ]
    rows += [
        (short, group, (r["mand"], *r["mand_ci"]), (r["inc"], *r["inc_ci"]))
        for short, group, r in others
    ]
    rows.sort(key=lambda r: r[2][0] + r[3][0])
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
    w = 0.36
    for i, (short, group, mand, inc) in enumerate(rows):
        c = p7.COLOR[group]
        for dx, (v, lo, hi), alpha, hatch in (
            (-w / 2, mand, 1.0, None),
            (w / 2, inc, 0.45, "///"),
        ):
            ax.bar(
                i + dx,
                v,
                width=w,
                color=c,
                alpha=alpha,
                hatch=hatch,
                edgecolor=c,
                linewidth=0.8,
                zorder=2,
            )
            ax.errorbar(
                i + dx,
                v,
                yerr=[[v - lo], [hi - v]],
                fmt="none",
                ecolor="#222222",
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )
            ax.text(
                i + dx,
                hi + 1.0,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#111111",
            )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("misalignment rate (%)", fontsize=10)
    ax.set_ylim(0, max(max(r[2][2], r[3][2]) for r in rows) + 9)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=[
            Patch(
                facecolor="#555555", label="mandated  (operator orders the violation)"
            ),
            Patch(
                facecolor="#555555",
                alpha=0.45,
                hatch="///",
                edgecolor="#555555",
                label="incentivized  (violation is merely rewarded)",
            ),
        ],
        loc="upper left",
        fontsize=8.5,
        frameon=False,
    )
    ax.set_title(
        "ODCV misalignment rate by variant (same 65 cells: 34 mandated + 31 incentivized)\n"
        f"PAR bars: mean of {len(seeds)} seeds ± 1.96·SEM; others: one run, "
        "scenario-cluster bootstrap 95% CI",
        fontsize=10.5,
    )
    fig.suptitle(
        "Post-action retrospection vs Sonnet difficult advice on Qwen3.6-27B",
        fontsize=12,
        fontweight="bold",
        x=0.98,
        ha="right",
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        f"PAR mandated seeds {listed['mand']}% (±{sem['mand']['half']:.2f}); incentivized "
        f"{listed['inc']}% (±{sem['inc']['half']:.2f}).",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    png_v = figure_path(out, f"{stem}_variants")
    fig.savefig(png_v, facecolor="white")
    plt.close(fig)

    # --- mirror ---------------------------------------------------------------------------
    lines = [
        f"# PAR seed mean ± 1.96·SEM vs Sonnet DA arms and references, same 65 cells",
        "",
        f"Generated {ts}. Cell set: `{p7.CFG}` exclusions.",
        "",
        "## PAR per seed (each: 2 rollouts x 65 cells, scenario-bootstrap CI)",
        "",
        "| seed | MR | 95% CI | mandated | incentivized | n |",
        "|---|---|---|---|---|---|",
    ]
    for k in seeds:
        r = per_seed[k]
        lines.append(
            f"| {k} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] | {r['mand']:.1f}% | "
            f"{r['inc']:.1f}% | {r['n']} |"
        )
    lines += [
        "",
        "## Between-seed error (k = %d trainings)" % len(seeds),
        "",
        "| metric | values | mean | SD | SEM = SD/√k | ±1.96·SEM | interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, name in (
        ("mr", "overall MR"),
        ("mand", "mandated MR"),
        ("inc", "incentivized MR"),
    ):
        s = sem[m]
        lines.append(
            f"| {name} | {', '.join(f'{v:.1f}' for v in s['values'])} | {s['mean']:.2f} | "
            f"{s['sd']:.2f} | {s['sem']:.2f} | ±{s['half']:.2f} | [{s['lo']:.2f}, {s['hi']:.2f}] |"
        )
    lines += [
        "",
        "## Comparison arms (one run each, scenario-bootstrap 95% CI)",
        "",
        "| arm | MR | 95% CI | sev | mandated (95% CI) | incentivized (95% CI) | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for short, _, r in others:
        lines.append(
            f"| {short.replace(chr(10), ' ')} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] | "
            f"{r['sev']:.2f} | {r['mand']:.1f}% [{r['mand_ci'][0]}, {r['mand_ci'][1]}] | "
            f"{r['inc']:.1f}% [{r['inc_ci'][0]}, {r['inc_ci'][1]}] | {r['n']} |"
        )
    lines += [
        "",
        "The PAR error bar is training-seed variance only (three independent trainings, "
        "each evaluated once); it does not include the eval-sampling noise the other arms' "
        "bootstrap intervals show. The two are not comparable widths.",
        "",
        f"Plots: `{png.relative_to(ROOT)}`, `{png_v.relative_to(ROOT)}`",
    ]
    md = out / f"{stem}_results.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(sem, indent=1))
    print(f"wrote {png}\nwrote {png_v}\nwrote {md}")


if __name__ == "__main__":
    fire.Fire(main)
