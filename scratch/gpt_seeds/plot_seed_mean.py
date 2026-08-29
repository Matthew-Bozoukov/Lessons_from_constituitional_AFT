# ABOUTME: ODCV misalignment per GENERATOR arm as the mean of its training seeds +- 1.96 SEM (GPT: 3
# ABOUTME: seeds; grok/Sonnet: their seeds when published, else one run), next to Sonnet concise + refs.
# Run: uv run python scratch/gpt_seeds/plot_seed_mean.py [--results gpt.42=<results.json>,gpt.69=<...>,grok.42=<repo::file>,...] [--out_dir output/gpt_seeds/plots]
#
# Sibling of scratch/par_b/plot_seed_mean.py (PAR seeds), generalised to several seed groups:
# every arm with >= 2 seeds is drawn as the MEAN of its per-seed MRs +- 1.96 * SD/sqrt(k)
# (training-seed variance, no eval noise); an arm with one run keeps its scenario-bootstrap
# 95% CI (eval-sampling noise). Whisker style + footnote say which is which; the markdown
# mirror carries the arithmetic and the t-based 95% interval (k=3 -> t=4.303) alongside
# the 1.96 one, because 1.96 is an ~81% interval at df=2 (scratch/stats/odcv_seed_sem.py).
# All arms on the same 65 cells (the gptresp eval config's 15 exclusions, shared by every
# paired arm), re-summarised from each run's published per-scenario medians -- no reruns.

from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import fire
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.misalignment.odcv.odcv import misalignment_rate, summarise  # noqa: E402

CFG = "configs/eval/odcv_bench_t2_9284_gptresp685_r64_paired_2x65.yaml"
SEM_Z = 1.96

# Validated categorical palette (dataviz skill, light mode, all pairs pass; CVD 7.2 in the
# warn band is covered by the per-bar value labels + tick labels). Refs are gray outlines:
# they are references, not series. Colour follows the GENERATOR that wrote the data.
RED, BLUE, GREEN, GRAY = "#e34948", "#2a78d6", "#008300", "#8a8985"

# key -> (short label, colour, hatch, seeds{seed: source}); source = local path str or
# (repo, file) on the Hub. --results adds/overrides `key.seed=<path | repo::file>`.
ARMS: dict[str, dict] = {
    "gpt": dict(
        short="GPT DA\n(gptresp685)",
        long="GPT-responder paired arm (gpt-5.6-luna draft, gpt-5.6-terra revise; 685 rows)",
        color=RED,
        hatch=None,
        seeds={
            0: (
                "LASR-Callum/2026-08-25-odcv-gptresp685-paired-eval",
                "combined2x_20260825_181731/results.json",
            )
        },
    ),
    "grok": dict(
        short="grok DA\n(grokresp703)",
        long="grok-responder paired arm (grok-4.6 draft + revise; 703 rows)",
        color=BLUE,
        hatch=None,
        # Seeds 42/69 published by the sibling seed-replicate run (matboz, 2026-08-28); same
        # 65 cells, same judges, same protocol as seed 0, so the three pool as a seed mean.
        seeds={
            0: (
                "LASR-Callum/2026-08-24-odcv-grokresp703-paired-eval",
                "results/results.json",
            ),
            42: (
                "matboz/2026-08-27-odcv-grokresp703-paired-seed42",
                "results/results.json",
            ),
            69: (
                "matboz/2026-08-27-odcv-grokresp703-paired-seed69",
                "results/results.json",
            ),
        },
    ),
    "sonnet": dict(
        short="Sonnet DA\n(da716)",
        long="Sonnet difficult advice (Haiku 4.5 draft, Sonnet 5 revise; da716)",
        color=GREEN,
        hatch=None,
        seeds={
            0: (
                "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
                "combined4x_20260814_230249/results.json",
            )
        },
    ),
    "sonnet_concise": dict(
        short="Sonnet DA\nconcise",
        long="Sonnet DA concise (arm C: Sonnet replies capped to grok's length; 703 rows)",
        color=GREEN,
        hatch="///",
        seeds={
            0: (
                "LASR-Callum/2026-08-26-odcv-sonnetconcise703-paired-eval",
                "combined2x_20260826_174216/results.json",
            )
        },
    ),
    "base": dict(
        short="base fp8\n(no SFT)",
        long="Qwen3.6-27B base fp8 (no SFT)",
        color=GRAY,
        hatch="...",
        seeds={0: ("matboz/odcv-qwen3.6-27b-transcripts", "base_fp8/results.json")},
    ),
    "table2": dict(
        short="table2-only\n(0% SFT)",
        long="table2-only 9,284 (0% synthetic control)",
        color=GRAY,
        hatch="...",
        seeds={
            0: (
                "LASR-Callum/qwen3.6-27b-table2-only-9284-r64",
                "combined5x_20260805_132959/results.json",
            )
        },
    ),
}


def _load(source) -> dict:
    if isinstance(source, str) and "::" in source:
        source = tuple(source.split("::", 1))
    path = (
        ROOT / source
        if isinstance(source, str)
        else Path(hf_hub_download(source[0], source[1], repo_type="dataset"))
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _restrict(psm: dict, excluded: set[str]) -> dict:
    return {
        v: {k: s for k, s in cells.items() if f"{v}/{k.split('/')[0]}" not in excluded}
        for v, cells in psm.items()
    }


def _variant_ci(cells: dict, n_boot: int = 10_000, seed: int = 0) -> tuple:
    """95% CI on one variant's MR, resampling SCENARIOS (each with all its rollouts).

    Two published shapes exist and both appear among the arms plotted here: older runs key
    each rollout separately (`"<Scenario>/rollout_NNN": 3.0`), current ones nest the
    rollouts (`"<Scenario>": [3.5, 3.0]`). Normalise to {scenario: [scores]} first --
    `summarise` already accepts both, so a plot that assumed one shape produced a TypeError
    on the other rather than a wrong number.
    """
    by_scenario: dict[str, list[float]] = {}
    for key, score in cells.items():
        runs = score if isinstance(score, list) else [score]
        by_scenario.setdefault(key.split("/")[0], []).extend(float(s) for s in runs)
    groups = list(by_scenario.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        draw = [s for g in rng.choices(groups, k=len(groups)) for s in g]
        draws.append(misalignment_rate(draw))
    draws.sort()
    return round(draws[int(0.025 * n_boot)], 1), round(draws[int(0.975 * n_boot)], 1)


def _run_stats(psm: dict, excluded: set[str]) -> dict:
    medians = _restrict(psm, excluded)
    s = summarise(medians)
    o = s["overall"]
    return dict(
        mr=o["mr_pct"],
        lo=o["mr_ci95"][0],
        hi=o["mr_ci95"][1],
        n=o["n_rollouts"],
        sev=o["mean_severity"],
        mand=s["mandated"]["mr_pct"],
        inc=s["incentivized"]["mr_pct"],
        mand_ci=_variant_ci(medians["mandated"]),
        inc_ci=_variant_ci(medians["incentivized"]),
    )


def _sem_row(values: list[float]) -> dict:
    k = len(values)
    mean = st.mean(values)
    sd = st.stdev(values) if k > 1 else 0.0
    sem = sd / math.sqrt(k)
    t = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    cov = 100 * (2 * stats.t.cdf(SEM_Z, k - 1) - 1) if k > 1 else float("nan")
    return dict(
        k=k,
        values=values,
        mean=round(mean, 2),
        sd=round(sd, 2),
        sem=round(sem, 2),
        half=round(SEM_Z * sem, 2),
        lo=round(mean - SEM_Z * sem, 2),
        hi=round(mean + SEM_Z * sem, 2),
        t=round(t, 3),
        half_t=round(t * sem, 2) if k > 1 else float("nan"),
        lo_t=round(mean - t * sem, 2) if k > 1 else float("nan"),
        hi_t=round(mean + t * sem, 2) if k > 1 else float("nan"),
        coverage_of_1p96_pct=round(cov, 1) if k > 1 else float("nan"),
    )


def _collect(results: dict[str, str]) -> list[dict]:
    cfg = OmegaConf.load(ROOT / CFG)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))
    rows = []
    for key, arm in ARMS.items():
        seeds = dict(arm["seeds"])
        for rk, src in results.items():
            k, _, sd = rk.partition(".")
            if k == key:
                seeds[int(sd or 0)] = src
        per_seed = {
            sd: _run_stats(_load(src)["per_scenario_medians"], excluded)
            for sd, src in sorted(seeds.items())
        }
        row = dict(
            key=key,
            short=arm["short"],
            long=arm["long"],
            color=arm["color"],
            hatch=arm["hatch"],
            per_seed=per_seed,
            sources={
                str(sd): (src if isinstance(src, str) else "::".join(src))
                for sd, src in seeds.items()
            },
        )
        if len(per_seed) >= 2:
            row["kind"] = "seed_mean"
            row["sem"] = {
                m: _sem_row([per_seed[s][m] for s in per_seed])
                for m in ("mr", "mand", "inc")
            }
            row.update(
                mr=row["sem"]["mr"]["mean"],
                lo=row["sem"]["mr"]["lo"],
                hi=row["sem"]["mr"]["hi"],
                mand=(
                    row["sem"]["mand"]["mean"],
                    row["sem"]["mand"]["lo"],
                    row["sem"]["mand"]["hi"],
                ),
                inc=(
                    row["sem"]["inc"]["mean"],
                    row["sem"]["inc"]["lo"],
                    row["sem"]["inc"]["hi"],
                ),
            )
        else:
            ((sd, r),) = per_seed.items()
            row["kind"] = "single"
            row.update(
                mr=r["mr"],
                lo=r["lo"],
                hi=r["hi"],
                mand=(r["mand"], *r["mand_ci"]),
                inc=(r["inc"], *r["inc_ci"]),
            )
        rows.append(row)
    rows.sort(key=lambda r: r["mr"])
    return rows


def _bar(ax, x, r, v, lo, hi, width, alpha=1.0, hatch=None, fs=9.5):
    seedmean = r["kind"] == "seed_mean"
    h = hatch if hatch is not None else r["hatch"]
    # Hatch lines take the edge colour; on a coloured fill they must contrast with it or
    # the texture (the only thing separating Sonnet from Sonnet-concise) vanishes.
    edge = "#555555" if r["color"] == GRAY else ("#ffffff" if h else r["color"])
    ax.bar(
        x,
        v,
        width=width,
        color=r["color"],
        alpha=alpha,
        hatch=h,
        edgecolor=edge,
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        x,
        v,
        yerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
        fmt="none",
        ecolor="#111111" if seedmean else "#555555",
        elinewidth=2.0 if seedmean else 1.2,
        capsize=7 if seedmean else 4,
        capthick=2.0 if seedmean else 1.2,
        zorder=3,
    )
    ax.text(
        x,
        hi + 1.0,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontsize=fs,
        fontweight="bold",
        color="#111111",
    )


def main(results: str = "", out_dir: str = "output/gpt_seeds/plots") -> None:
    """Render the generator seed-mean comparison (bars + variants) and its mirror.

    Args:
        results: Comma-separated `key.seed=<results.json path | repo::file>` entries adding
            seeds to an arm (keys: gpt, grok, sonnet, sonnet_concise, base, table2), e.g.
            `gpt.42=output/odcv_bench/<key>/combined2x_<ts>/results.json,gpt.69=...`.
        out_dir: Where the PNGs, results.json and the markdown mirror go.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    parts = results if isinstance(results, (tuple, list)) else str(results).split(",")
    rmap = dict(str(p).partition("=")[::2] for p in parts if "=" in str(p))
    rows = _collect(rmap)
    seed_arms = [r for r in rows if r["kind"] == "seed_mean"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"odcv_generator_seedmean_65cells_{ts}"
    title = "Generator ablation on Qwen3.6-27B: who wrote the difficult-advice answers"

    # --- overall bars ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    for x, r in enumerate(rows):
        _bar(ax, x, r, r["mr"], r["lo"], r["hi"], 0.62)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [
            r["short"]
            + (
                "\nmean of %d seeds" % r["sem"]["mr"]["k"]
                if r["kind"] == "seed_mean"
                else ""
            )
            for r in rows
        ],
        fontsize=8.8,
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("misalignment rate (%)", fontsize=10)
    ax.set_ylim(0, max(r["hi"] for r in rows) + 8)
    ax.yaxis.grid(True, color="#e4e3df", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#111111",
                lw=2.0,
                label="mean of training seeds ± 1.96·SEM",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                lw=1.2,
                label="one run, scenario-bootstrap 95% CI",
            ),
            Patch(facecolor=RED, label="GPT-written answers"),
            Patch(facecolor=BLUE, label="grok-written answers"),
            Patch(
                facecolor=GREEN, label="Sonnet-written answers (hatched: length-capped)"
            ),
            Patch(
                facecolor=GRAY,
                hatch="...",
                edgecolor="#555555",
                label="no-SFT references",
            ),
        ],
        loc="upper left",
        fontsize=7.8,
        frameon=False,
    )
    ax.set_title(
        "ODCV misalignment rate on the same 65 cells (34 mandated + 31 incentivized)",
        fontsize=10.5,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.98, ha="right", y=0.995)
    foot = " · ".join(
        f"{r['key']}: seeds {', '.join(f'{v:.1f}' for v in r['sem']['mr']['values'])}% → "
        f"{r['sem']['mr']['mean']:.1f} ± {r['sem']['mr']['half']:.1f} (SD {r['sem']['mr']['sd']:.2f})"
        for r in seed_arms
    )
    fig.text(
        0.01,
        0.005,
        (foot + ". " if foot else "")
        + "Seed bars carry training-seed variance only (each seed "
        "evaluated once, 2 rollouts × 65 cells); ±1.96·SEM at k=3 is an ~81% interval "
        "(t=4.30 for 95%). Others: published per-scenario medians on the same cells.",
        fontsize=7.2,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    png = out / f"{stem}_bars.png"
    fig.savefig(png, facecolor="white")
    plt.close(fig)

    # --- mandated vs incentivized --------------------------------------------------------
    vrows = sorted(rows, key=lambda r: r["mand"][0] + r["inc"][0])
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=200)
    w = 0.36
    for i, r in enumerate(vrows):
        _bar(ax, i - w / 2, r, *r["mand"], w, fs=8)
        _bar(ax, i + w / 2, r, *r["inc"], w, alpha=0.45, hatch="\\\\\\", fs=8)
    ax.set_xticks(range(len(vrows)))
    ax.set_xticklabels([r["short"] for r in vrows], fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("misalignment rate (%)", fontsize=10)
    ax.set_ylim(0, max(max(r["mand"][2], r["inc"][2]) for r in vrows) + 9)
    ax.yaxis.grid(True, color="#e4e3df", lw=0.8, zorder=0)
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
                hatch="\\\\\\",
                edgecolor="#555555",
                label="incentivized  (violation is merely rewarded)",
            ),
            Line2D([], [], color="#111111", lw=2.0, label="seed mean ± 1.96·SEM"),
            Line2D(
                [],
                [],
                color="#555555",
                lw=1.2,
                label="one run, scenario-cluster bootstrap 95% CI",
            ),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    ax.set_title("ODCV misalignment rate by variant (same 65 cells)", fontsize=10.5)
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.98, ha="right", y=0.995)
    fig.text(
        0.01,
        0.005,
        " · ".join(
            f"{r['key']} mandated {', '.join(f'{v:.1f}' for v in r['sem']['mand']['values'])}% "
            f"(±{r['sem']['mand']['half']:.1f}); incentivized "
            f"{', '.join(f'{v:.1f}' for v in r['sem']['inc']['values'])}% (±{r['sem']['inc']['half']:.1f})"
            for r in seed_arms
        ),
        fontsize=7.2,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    png_v = out / f"{stem}_variants.png"
    fig.savefig(png_v, facecolor="white")
    plt.close(fig)

    # --- mirror + json ------------------------------------------------------------------
    lines = [
        f"# Generator arms: seed mean ± 1.96·SEM vs single runs, same 65 cells",
        "",
        f"Generated {ts}. Cell set: `{CFG}` exclusions. No reruns: every number is "
        "re-summarised from the run's published per-scenario medians.",
        "",
        "## Per run (each: rollouts × 65 cells, scenario-bootstrap 95% CI)",
        "",
        "| arm | seed | MR | 95% CI | sev | mandated | incentivized | n rollouts | source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for sd, s in r["per_seed"].items():
            lines.append(
                f"| {r['key']} | {sd} | {s['mr']:.1f}% | [{s['lo']:.1f}, {s['hi']:.1f}] "
                f"| {s['sev']:.2f} | {s['mand']:.1f}% [{s['mand_ci'][0]}, {s['mand_ci'][1]}] "
                f"| {s['inc']:.1f}% [{s['inc_ci'][0]}, {s['inc_ci'][1]}] | {s['n']} "
                f"| `{r['sources'][str(sd)]}` |"
            )
    lines += [
        "",
        "## Between-seed error (arms with >= 2 seeds)",
        "",
        "| arm | metric | per-seed values | mean | SD | SEM = SD/√k | **±1.96·SEM** | "
        "interval | ±t·SEM (true 95%) | interval | coverage of ±1.96 at df=k-1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in seed_arms:
        for m, name in (
            ("mr", "overall MR"),
            ("mand", "mandated MR"),
            ("inc", "incentivized MR"),
        ):
            s = r["sem"][m]
            lines.append(
                f"| {r['key']} | {name} | {', '.join(f'{v:.1f}' for v in s['values'])} | "
                f"{s['mean']:.2f} | {s['sd']:.2f} | {s['sem']:.2f} | **±{s['half']:.2f}** | "
                f"[{s['lo']:.2f}, {s['hi']:.2f}] | ±{s['half_t']:.2f} (t={s['t']}) | "
                f"[{s['lo_t']:.2f}, {s['hi_t']:.2f}] | {s['coverage_of_1p96_pct']:.0f}% |"
            )
    lines += [
        "",
        "## Bars as drawn",
        "",
        "| arm | kind | bar | whisker |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['long']} | {r['kind']} | {r['mr']:.1f}% | [{r['lo']:.1f}, {r['hi']:.1f}] |"
        )
    lines += [
        "",
        "Seed-mean whiskers are training-seed variance only (each seed evaluated once); "
        "single-run whiskers are eval-sampling noise. The two are not comparable widths. "
        "At k=3, ±1.96·SEM covers ~81%, not 95% (t=4.303 for 95%), per "
        "scratch/stats/odcv_seed_sem.py.",
        "",
        f"Plots: `{png.relative_to(ROOT)}`, `{png_v.relative_to(ROOT)}`",
    ]
    md = out / f"{stem}_results.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = [
        {k: v for k, v in r.items() if k not in ("color", "hatch")} for r in rows
    ]
    (out / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    for r in rows:
        print(
            f"{r['key']:15s} {r['kind']:9s} {r['mr']:5.1f}%  [{r['lo']:.1f}, {r['hi']:.1f}]"
            + (
                f"   seeds {r['sem']['mr']['values']}"
                if r["kind"] == "seed_mean"
                else ""
            )
        )
    print(f"wrote {png}\nwrote {png_v}\nwrote {md}")


if __name__ == "__main__":
    fire.Fire(main)
