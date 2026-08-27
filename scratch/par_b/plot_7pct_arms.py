# ABOUTME: ODCV misalignment for every 716-row (7%) SFT arm with a published results.json,
# ABOUTME: re-summarised on the shared 65-cell set: bar chart with 95% CI (+ a dot-plot variant).
# Run: uv run python scratch/par_b/plot_7pct_arms.py --results <par combined results.json> [--style bars|variants|dots] [--out_dir output/plots]
#
# Nothing is re-run: each arm's per-scenario medians are pulled from the results.json its
# eval published on the Hub and restricted to the cells this arm was scored on (the
# peer-critique 65-cell set), so every interval on the plot is over the SAME cells. The
# post-action-retrospection (design B) arm is `this run`; the t10 arm is pulled from the
# Hub like the other siblings. Arms
# without a pullable results.json (courtroom716, peercritique716) are not drawn; the
# c6masked number is the team's posted 65-cell figure and is marked as such.

import json
import sys
import time
from pathlib import Path

import fire
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import misalignment_rate, summarise  # noqa: E402


def _variant_ci(cells: dict[str, float], n_boot: int = 10_000, seed: int = 0) -> tuple:
    """95% CI on one variant's MR by resampling SCENARIOS (each with all its rollouts).

    The module's `bootstrap_ci` is paired across variants and only feeds the overall
    numbers; per variant it reports a point estimate alone. Rollouts of one scenario are
    not independent draws, so the cluster here is the scenario, as upstream's is.
    """
    import random

    by_scenario: dict[str, list[float]] = {}
    for key, score in cells.items():
        by_scenario.setdefault(key.split("/")[0], []).append(score)
    groups = list(by_scenario.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        draw = [s for g in rng.choices(groups, k=len(groups)) for s in g]
        stats.append(misalignment_rate(draw))
    stats.sort()
    return round(stats[int(0.025 * n_boot)], 1), round(stats[int(0.975 * n_boot)], 1)


CFG = "scratch/par_b/odcv_bench_t2_9284_par716_2x65.yaml"

# (short label, long label, group, source). group: "this" | "sft7" | "ref".
# (key, short label, long label, group, source). group: "this" | "sft7" | "ref". `--only`
# takes a comma-separated list of keys to draw a subset (e.g. the Sonnet-only view).
ARMS = [
    (
        "par",
        "PAR 716\n(this run)",
        "post-action-retrospection 716 (design B; this run, 2 rollouts)",
        "this",
        None,
    ),  # source filled from --results
    (
        "t10",
        "t10 curiosity\n716",
        "t10 curiosity 716 (trait ODCV cannot reward)",
        "sft7",
        (
            "LASR-Callum/2026-08-20-odcv-t10-curiosity-716-eval",
            "combined2x_20260820_185205/results.json",
        ),
    ),
    (
        "da716",
        "Sonnet DA\n(da716)",
        "Sonnet DA - da716 (difficult advice v2, 9 traits)",
        "sft7",
        (
            "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
            "combined4x_20260814_230249/results.json",
        ),
    ),
    (
        "sonnet_concise",
        "Sonnet DA\nconcise",
        "Sonnet DA concise - arm C (Sonnet replies capped to grok's length, 703 paired rows)",
        "sft7",
        (
            "LASR-Callum/2026-08-26-odcv-sonnetconcise703-paired-eval",
            "combined2x_20260826_174216/results.json",
        ),
    ),
    (
        "lessswap",
        "lessswap716",
        "lessswap716 (LESS-selected rows, 3 traits)",
        "sft7",
        ("LASR-Callum/2026-08-18-odcv-lessswap716-eval", "results.json"),
    ),
    (
        "synthdoc_v1",
        "synthdoc-716\n(v1)",
        "synthdoc-716 (difficult advice v1)",
        "sft7",
        (
            "matboz/2026-08-08-difficult-advice-5pct-qwen36-odcv-rollouts",
            "results.json",
        ),
    ),
    (
        "table2",
        "table2-only\n(0% SFT)",
        "table2-only 9284 (0% SFT control)",
        "ref",
        (
            "LASR-Callum/qwen3.6-27b-table2-only-9284-r64",
            "combined5x_20260805_132959/results.json",
        ),
    ),
    (
        "base",
        "base fp8\n(no SFT)",
        "Qwen3.6-27B base fp8 (no SFT)",
        "ref",
        ("matboz/odcv-qwen3.6-27b-transcripts", "base_fp8/results.json"),
    ),
]
# Posted on the same 65 cells (team thread, 2026-08-18); no results.json to pull.
POSTED = [
    (
        "c6masked",
        "c6masked (synthdoc-716, C6 spans unsupervised)",
        "sft7",
        9.7,
        (5.5, 13.9),
        195,
        "posted 65-cell figure, MR ±4.2",
    )
]

BLUE, RED, GRAY = "#3a63a8", "#c23b3b", "#8a8985"
COLOR = {"this": RED, "sft7": BLUE, "ref": GRAY}


def _restrict(psm: dict, excluded: set[str]) -> dict:
    return {
        v: {k: s for k, s in cells.items() if f"{v}/{k.split('/')[0]}" not in excluded}
        for v, cells in psm.items()
    }


def _load(source) -> dict:
    path = (
        ROOT / source
        if isinstance(source, str)
        else Path(hf_hub_download(source[0], source[1], repo_type="dataset"))
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _collect(results: str, only: tuple[str, ...] = ()) -> list[dict]:
    cfg = OmegaConf.load(CFG)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))
    rows = []
    unknown = set(only) - {a[0] for a in ARMS}
    if unknown:
        raise SystemExit(
            f"unknown arm key(s) {sorted(unknown)}; known: {[a[0] for a in ARMS]}"
        )
    for key, short, label, group, source in ARMS:
        if only and key not in only:
            continue
        source = results if source is None else source
        medians = _restrict(_load(source)["per_scenario_medians"], excluded)
        s = summarise(medians)
        o = s["overall"]
        rows.append(
            dict(
                short=short,
                label=label,
                group=group,
                mr=o["mr_pct"],
                lo=o["mr_ci95"][0],
                hi=o["mr_ci95"][1],
                n=o["n"],
                sev=o["mean_severity"],
                mand=s["mandated"]["mr_pct"],
                inc=s["incentivized"]["mr_pct"],
                mand_ci=_variant_ci(medians["mandated"]),
                inc_ci=_variant_ci(medians["incentivized"]),
                mand_n=s["mandated"]["n"],
                inc_n=s["incentivized"]["n"],
                note="re-summarised on the 65 cells from published medians",
            )
        )
    for short, label, group, mr, (lo, hi), n, note in [] if only else POSTED:
        rows.append(
            dict(
                short=short,
                label=label,
                group=group,
                mr=mr,
                lo=lo,
                hi=hi,
                n=n,
                sev=None,
                mand=None,
                inc=None,
                mand_ci=None,
                inc_ci=None,
                mand_n=None,
                inc_n=None,
                note=note,
            )
        )
    rows.sort(key=lambda r: r["mr"])
    return rows


def _mirror(rows: list[dict], md: Path, png: Path, ts: str) -> None:
    lines = [
        "# ODCV misalignment, 716-row (7%) SFT arms, same 65 cells",
        "",
        f"Generated {ts}. Re-summarised from published per-scenario medians (no reruns). "
        f"Cell set: `{CFG}` exclusions.",
        "",
        "| arm | MR | 95% CI | sev | mandated (95% CI) | incentivized (95% CI) | n | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sev = f"{r['sev']:.2f}" if r["sev"] is not None else "—"
        mand = (
            f"{r['mand']:.1f}% [{r['mand_ci'][0]}, {r['mand_ci'][1]}] n={r['mand_n']}"
            if r["mand"] is not None
            else "—"
        )
        inc = (
            f"{r['inc']:.1f}% [{r['inc_ci'][0]}, {r['inc_ci'][1]}] n={r['inc_n']}"
            if r["inc"] is not None
            else "—"
        )
        lines.append(
            f"| {r['label']} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] | "
            f"{sev} | {mand} | {inc} | {r['n']} | {r['note']} |"
        )
    lines += [
        "",
        "Not drawn (no pullable results.json): courtroom716, peercritique716.",
        "",
        f"Plot: `{png.relative_to(ROOT)}`",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bars(rows: list[dict], png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    xs = list(range(len(rows)))
    for x, r in zip(xs, rows):
        ax.bar(x, r["mr"], width=0.62, color=COLOR[r["group"]], zorder=2)
        ax.errorbar(
            x,
            r["mr"],
            yerr=[[r["mr"] - r["lo"]], [r["hi"] - r["mr"]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.4,
            capsize=5,
            capthick=1.4,
            zorder=3,
        )
        ax.text(
            x,
            r["hi"] + 1.2,
            f"{r['mr']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([r["short"] for r in rows], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("misalignment rate (%)", fontsize=10)
    ax.set_ylim(0, max(r["hi"] for r in rows) + 8)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("ODCV misalignment rate\n(95% CI, 65 identical cells)", fontsize=11)
    fig.suptitle(
        "716-row (7%) synthetic-SFT arms on Qwen3.6-27B",
        fontsize=12,
        fontweight="bold",
        x=0.98,
        ha="right",
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        "blue = other 716-row arms · red = this run · gray = no-SFT references. All "
        "intervals recomputed on the same 65 cells from each arm's published medians "
        "(no reruns)"
        + (
            "; c6masked is the team's posted figure."
            if any(r["sev"] is None for r in rows)
            else "."
        ),
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(png, facecolor="white")


def _variants(rows: list[dict], png: Path) -> None:
    """Grouped bars: each arm's mandated and incentivized MR side by side, own CIs."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = [r for r in rows if r["mand"] is not None]  # posted-only arms have no split
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
    w = 0.36
    for i, r in enumerate(rows):
        c = COLOR[r["group"]]
        for dx, key, ci_key, alpha, hatch in (
            (-w / 2, "mand", "mand_ci", 1.0, None),
            (w / 2, "inc", "inc_ci", 0.45, "///"),
        ):
            v, (lo, hi) = r[key], r[ci_key]
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
    ax.set_xticklabels([r["short"] for r in rows], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("misalignment rate (%)", fontsize=10)
    ax.set_ylim(0, max(max(r["mand_ci"][1], r["inc_ci"][1]) for r in rows) + 9)
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
        "ODCV misalignment rate by variant\n(95% CI, scenario-cluster bootstrap; "
        "same 65 cells: 34 mandated + 31 incentivized)",
        fontsize=11,
    )
    fig.suptitle(
        "716-row (7%) synthetic-SFT arms on Qwen3.6-27B",
        fontsize=12,
        fontweight="bold",
        x=0.98,
        ha="right",
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        "blue = other 716-row arms · red = this run · gray = no-SFT references. Solid = "
        "mandated, hatched = incentivized. Recomputed from each arm's published "
        "per-scenario medians (no reruns). Arms with only an overall figure posted are "
        "omitted.",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(png, facecolor="white")


def _dots(rows: list[dict], png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(rows) + 2.2), dpi=200)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        c = COLOR[r["group"]]
        ax.plot(
            [r["lo"], r["hi"]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2
        )
        ax.plot(r["mr"], y, "o", ms=9, color=c, mec="white", mew=1.5, zorder=3)
        ax.annotate(
            f"{r['mr']:.1f}%  [{r['lo']:.1f}, {r['hi']:.1f}]   n={r['n']}",
            (r["hi"], y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#52514e",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 72)
    ax.set_xlabel("ODCV-Bench misalignment rate (median judge score ≥ 3), %  — 95% CI")
    ax.xaxis.grid(True, color="#e4e3df", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        "716-row (7%) SFT arms on Qwen3.6-27B: ODCV misalignment on the same 65 cells",
        loc="left",
        fontsize=12,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(png, facecolor="white")


def main(
    results: str,
    style: str = "bars",
    out_dir: str = "output/plots",
    only: str = "",
    tag: str = "",
) -> None:
    """Render the comparison.

    Args:
        results: The PAR arm's combined results.json (scratch/odcv_combine_passes.py output).
        style: `bars` (vertical bars + CI whiskers, value labels), `variants` (mandated vs
            incentivized grouped bars, per-variant CIs) or `dots` (dot-and-interval).
        out_dir: Where the PNG and its markdown mirror go.
        only: Comma-separated arm keys to draw (see ARMS), e.g.
            `par,da716,sonnet_concise,base,table2`. Empty = every arm plus the posted one.
        tag: Name segment for the output files when `only` is set (default: the keys joined).
    """
    import matplotlib

    matplotlib.use("Agg")

    # fire hands `--only a,b,c` over as a tuple already; a quoted string arrives as str.
    parts = only if isinstance(only, (tuple, list)) else str(only).split(",")
    keys = tuple(str(k).strip() for k in parts if str(k).strip())
    rows = _collect(results, keys)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = (
        "odcv_7pct_arms_par_65cells"
        if not keys
        else (f"odcv_{tag or '_'.join(keys)}_65cells")
    )
    png = out / f"{stem}_{style}_{ts}.png"
    md = out / f"{stem}_{style}_{ts}_results.md"
    {"bars": _bars, "variants": _variants, "dots": _dots}[style](rows, png)
    _mirror(rows, md, png, ts)
    print(f"wrote {png}\nwrote {md}")


if __name__ == "__main__":
    fire.Fire(main)
