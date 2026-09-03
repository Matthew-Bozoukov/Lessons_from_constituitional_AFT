# ABOUTME: ODCV misalignment per TRAINING-DATA arm, every interval from src/eval/stats:
# ABOUTME: a seeded arm pools its seeds on the design's checkpoint axis, so all bars are comparable.
# Run: uv run python scratch/gpt_seeds/plot_seed_mean.py [--results gpt.42=<results.json>,gpt.69=<...>,grok.42=<repo::file>,...] [--out_dir output/gpt_seeds/plots]
#
# Supersedes scratch/par_b/plot_seed_mean.py (PAR seeds only): every arm with >= 2 seeds is
# scored through odcv.summarise_pooled, which puts the training seed on its own axis of the
# ODCV design. That replaced a mean-of-per-seed-MRs +- 1.96*SEM: the old figure drew two
# different quantities as the same whisker (training variance for seeded arms, eval-sampling
# noise for single runs), so the widths could not be compared. Now every bar carries scenario
# sampling, and a seeded arm additionally carries seed-to-seed variance, in one interval.
# Same exclusion list for every arm (the gptresp eval config's 15 -- verified
# identical to the PAR arms' scratch/par_b/odcv_bench_t2_9284_par716_2x65.yaml exclusions),
# re-summarised from each run's published per-scenario medians -- no reruns.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The statistics live in src/ now -- shape normalisation, the cell-level bootstrap, the
# one-pass-per-seed rule and the seed mean are all reusable and were duplicated across
# three scratch scripts before. Nothing in this file recomputes them.
from src.naming import figure_path  # noqa: E402
from src.eval.misalignment.odcv.odcv import (  # noqa: E402
    passes_by_index,
    pick_most_complete_pass,
    shared_cells,
    summarise,
    summarise_pooled,
)

CFG = "configs/eval/2026-08-24_odcv_bench_table2_9284_gpt_responder_685_rank64_paired_2_65.yaml"
SEM_Z = 1.96

# Validated categorical palette (dataviz skill, light mode, `--pairs all`: all checks pass;
# the two WARNs -- CVD 7.2 green/red, TEAL contrast 2.99 -- both take the same relief, the
# per-bar value labels + tick labels + the markdown mirror). Refs are gray outlines: they
# are references, not series. Colour follows the GENERATOR that wrote the data, except TEAL
# = the one arm that is a different DOCUMENT TYPE (Sonnet-written retrospection, not advice).
RED, BLUE, GREEN, TEAL, GRAY = "#e34948", "#2a78d6", "#008300", "#00a3ad", "#8a8985"

# THE difficult-advice baseline, named once so a plot cannot disagree with docs/BASELINES.md
# about which arm everything is compared against. tests/test_baselines.py asserts the two
# agree. Changing the baseline means changing this line, that file, and nothing else.
BASELINE_ARM = "principle_scoped"

# key -> (short label, colour, hatch, seeds{seed: source}); source = local path str or
# (repo, file) on the Hub. --results adds/overrides `key.seed=<path | repo::file>`.
ARMS: dict[str, dict] = {
    "par": dict(
        short="PAR 716\n(retrospection)",
        long="post-action retrospection 716 (Sonnet-written; design B, 3 training seeds)",
        color=TEAL,
        hatch=None,
        # Seeds 0/1/2 = the three PAR trainings (2026-08-27/28); same exclusion list, same
        # judges (grok-4.20 + gemini-3.1-pro) and the same protocol as every arm here.
        seeds={
            0: (
                "LASR-Callum/2026-08-27-odcv-post-action-retrospection-716-eval",
                "combined2x_20260827_023241/results.json",
            ),
            1: (
                "LASR-Callum/2026-08-27-odcv-post-action-retrospection-716-seed-1-eval",
                "combined2x_20260827_161549/results.json",
            ),
            2: (
                "LASR-Callum/2026-08-27-odcv-post-action-retrospection-716-seed-2-eval",
                "combined3x_20260828_003554/results.json",
            ),
        },
    ),
    "gpt": dict(
        short="GPT DA\n(gptresp685)",
        long="GPT-responder paired arm (gpt-5.6-luna draft, gpt-5.6-terra revise; 685 rows)",
        color=RED,
        hatch=None,
        # Seeds 42/69 published by this repo's own seed-replicate run (2026-08-29); both are
        # checkpoint-600 adapters, the arm's documented protocol (it crashes at step 624/624).
        seeds={
            0: (
                "LASR-Callum/2026-08-25-odcv-gpt-responder-685-paired-eval",
                "combined2x_20260825_181731/results.json",
            ),
            42: (
                "LASR-Callum/2026-08-28-odcv-gpt-responder-685-seed42-paired-eval",
                "results/results.json",
            ),
            69: (
                "LASR-Callum/2026-08-28-odcv-gpt-responder-685-seed69-paired-eval",
                "results/results.json",
            ),
        },
    ),
    "grok": dict(
        short="grok DA\n(grokresp703)",
        long="grok-responder paired arm (grok-4.6 draft + revise; 703 rows)",
        color=BLUE,
        hatch=None,
        # Seeds 42/69 published by the sibling seed-replicate run (matboz, 2026-08-28); same
        # exclusions, judges and protocol as seed 0, so the three pool as a seed mean.
        seeds={
            0: (
                "LASR-Callum/2026-08-24-odcv-grok-responder-703-paired-eval",
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
    "principle_scoped": dict(
        short="Sonnet DA\nprinciple-scoped 702",
        long="THE DA BASELINE: principle-scoped 702 (da716's recipe with the constitution removed "
        "from both refine stages; see docs/BASELINES.md)",
        color=GREEN,
        hatch="..",
        # Matched fork of da716, not a separate generation: it resumed da716's run directory
        # from stages 1-4, so scenarios and draft prompts are byte-identical and only
        # revise_prompts / revise_responses differ. Seeds 42/69 training 2026-08-31.
        seeds={
            0: (
                "LASR-Callum/2026-08-21-odcv-difficult-advice-principle-scoped-702-eval",
                "combined2x_20260824_130511/results.json",
            )
        },
    ),
    "sonnet": dict(
        short="Sonnet DA\n(da716)",
        long="Sonnet difficult advice, da716 -- SUPERSEDED as the baseline by principle-scoped 702; "
        "kept because the generator swaps freeze ITS stages 1-4 and can only be read against it",
        color=GREEN,
        hatch=None,
        seeds={
            0: (
                "LASR-Callum/2026-08-14-qwen36-lora-table2-9284-difficult-advice-716-rank-64-dynbatch",
                "combined4x_20260814_230249/results.json",
            )
        },
    ),
    "synthdoc": dict(
        short="Sonnet DA v1\n(synthdoc716)",
        long="Sonnet difficult advice v1 (synthdoc-716; the ONLY Sonnet DA arm with seeds)",
        color=GREEN,
        hatch="xxx",
        # da716 (v2, below) has exactly ONE training -- no seed 42/69 adapter or eval
        # exists anywhere under LASR-Callum or matboz. v1 is the Sonnet difficult-advice
        # arm that does have three trainings, so it is the one that can carry a seed
        # whisker. Different corpus from da716: do not read them as the same arm.
        seeds={
            0: (
                "matboz/2026-08-24-odcv-synthdoc-716-seed0-5pass",
                "results/results.json",
            ),
            42: (
                "LASR-Callum/2026-08-26-odcv-synthdoc-716-seed42",
                "results/results.json",
            ),
            69: (
                "LASR-Callum/2026-08-26-odcv-synthdoc-716-seed69",
                "results/results.json",
            ),
        },
    ),
    "sonnet_concise": dict(
        short="Sonnet DA\nconcise",
        long="Sonnet DA concise (arm C: Sonnet replies capped to grok's length; 703 rows)",
        color=GREEN,
        hatch="///",
        seeds={
            0: (
                "LASR-Callum/2026-08-26-odcv-sonnet-concise-703-paired-eval",
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
                "LASR-Callum/2026-08-05-qwen36-table2-only-9284-rank-64",
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


def _cells(psm: dict, excluded: set[str]) -> dict[str, dict[str, list[float]]]:
    """{pass_id: {"variant/scenario": [severity]}} from published medians, exclusions cut.

    Thin wrapper over `passes_by_index` in src/. The shape normalisation it does is the
    whole ballgame: a combined multi-pass run keys cells "<Scenario>/rollout_NNN", and
    taking those at face value makes each rollout its own scenario -- which is how the
    earlier version of this figure got single-run CIs that were far too narrow.
    """
    return {
        pid: {c: s for c, s in cells.items() if c not in excluded}
        for pid, cells in passes_by_index(psm).items()
    }


def _merge(passes: dict) -> dict[str, list[float]]:
    """Every pass pooled into one {cell: [scores]} -- used for single-training arms."""
    merged: dict[str, list[float]] = {}
    for cells in passes.values():
        for cell, scores in cells.items():
            merged.setdefault(cell, []).extend(scores)
    return merged


def _split(keys) -> dict[str, list[str]]:
    return {
        v: [k for k in keys if k.startswith(f"{v}/")]
        for v in ("mandated", "incentivized")
    }


def _to_medians(cells: dict[str, list[float]], keys) -> dict[str, dict[str, list[float]]]:
    """{"variant/scenario": [s]} restricted to `keys` -> the {variant: {scenario: [s]}}
    shape src/eval/misalignment/odcv/odcv.py's summarise takes."""
    out: dict[str, dict[str, list[float]]] = {"mandated": {}, "incentivized": {}}
    for k in keys:
        variant, _, scenario = k.partition("/")
        out[variant][scenario] = list(cells[k])
    return out


def _arm_stats(cells_by_seed: dict, keys) -> dict:
    """MR + per-variant MR with intervals from src/eval/stats, via odcv.summarise[_pooled].

    This is the whole reason the figure changed shape. It used to draw two incomparable
    whiskers: a mean-of-per-seed-MRs +- 1.96*SEM for seeded arms, and a cell bootstrap for
    single runs -- different quantities, same visual. `summarise_pooled` instead puts the
    CHECKPOINT (the training seed) on its own axis of the ODCV design, so a seeded arm's
    interval carries scenario sampling AND seed-to-seed variance in one number, on the same
    scale as a single run's. One kind of whisker, and the seeded arms' are honestly wider.
    """
    by_ckpt = {str(sd): _to_medians(c, keys) for sd, c in cells_by_seed.items()}
    s = summarise_pooled(by_ckpt) if len(by_ckpt) > 1 else summarise(next(iter(by_ckpt.values())))
    o = s["overall"]
    out = dict(
        mr=o["mr_pct"],
        lo=o["mr_ci95"][0],
        hi=o["mr_ci95"][1],
        n_scenarios=o["n_scenarios"],
        n_rollouts=o["n_rollouts"],
        sev=o["mean_severity"],
        n_checkpoints=o.get("n_checkpoints", 1),
        claims=s["stats"]["overall"]["mr"].get("claims", []),
    )
    for v in ("mandated", "incentivized"):
        b = s.get(v) or {}
        ci = b.get("mr_ci95") or [b.get("mr_pct", 0.0), b.get("mr_pct", 0.0)]
        out["mand" if v == "mandated" else "inc"] = (b.get("mr_pct", 0.0), ci[0], ci[1])
    return out


def _collect(results: dict[str, str], only: tuple[str, ...] = ()) -> list[dict]:
    """Every arm on the cells EVERY arm kept, scored through src/eval/stats.

    Two intersections, in order. Within an arm, a cell missing from any seed is dropped from
    all its seeds. Then ACROSS arms, a cell missing from any arm is dropped from all arms --
    without which the bars are comparable in shape but not in level: arms sat on 57 to 65
    surviving cells, and the arm that lost the most read 2.6 pp low against its own figure
    purely because the cells it lost were ones it FAILED. Dropout is not random w.r.t. MR --
    the cells that fail are the long, agentic rollouts -- so a smaller set is biased LOW.

    `--only` restricts to a subset of arms; the intersection is then over just those, which
    is what makes a six-arm view legitimately comparable rather than a crop of a nine-arm one.
    """
    cfg = OmegaConf.load(ROOT / CFG)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))
    unknown = set(only) - set(ARMS)
    assert not unknown, f"unknown arm(s) {sorted(unknown)}; known: {list(ARMS)}"

    prepared = {}
    for key, arm in ARMS.items():
        if only and key not in only:
            continue
        seeds = dict(arm["seeds"])
        for rk, src in results.items():
            k, _, sd = rk.partition(".")
            if k == key:
                seeds[int(sd or 0)] = src
        passes = {
            sd: _cells(_load(src)["per_scenario_medians"], excluded)
            for sd, src in sorted(seeds.items())
        }
        if len(passes) >= 2:
            # ONE pass per seed: a seed scored over several passes gets a less noisy point
            # estimate than a one-pass sibling, which would bias the seed axis of the design.
            cells = {sd: pick_most_complete_pass(ps) for sd, ps in passes.items()}
        else:
            # A single training has no seed axis to protect, so every pass it ran is kept.
            cells = {sd: _merge(ps) for sd, ps in passes.items()}
        prepared[key] = dict(
            arm=arm, seeds=seeds, passes=passes, cells=cells, own=shared_cells(cells)
        )

    common = sorted(set.intersection(*(set(p["own"]) for p in prepared.values())))
    assert common, "no cell survives every arm"

    rows = []
    for key, pr in prepared.items():
        arm, seeds, passes, cells = pr["arm"], pr["seeds"], pr["passes"], pr["cells"]
        st = _arm_stats(cells, common)
        rows.append(
            dict(
                key=key,
                short=arm["short"],
                long=arm["long"],
                color=arm["color"],
                hatch=arm["hatch"],
                n_cells=len(common),
                n_cells_own=len(pr["own"]),
                n_seeds=len(cells),
                kind="pooled" if len(cells) > 1 else "single",
                n_passes={str(sd): len(ps) for sd, ps in passes.items()},
                sources={
                    str(sd): (src if isinstance(src, str) else "::".join(src))
                    for sd, src in seeds.items()
                },
                per_seed={
                    sd: _arm_stats({sd: c}, common) for sd, c in cells.items()
                },
                **st,
            )
        )
    rows.sort(key=lambda r: r["mr"])
    return rows


def _bar(ax, x, r, v, lo, hi, width, alpha=1.0, hatch=None, fs=9.5):
    seedmean = r["kind"] == "pooled"
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


def main(results: str = "", out_dir: str = "output/gpt_seeds/plots",
         only: str = "") -> None:
    """Render the arm comparison (bars + variants) and its markdown mirror.

    Args:
        results: Comma-separated `key.seed=<results.json path | repo::file>` entries adding
            seeds to an arm (keys: par, gpt, grok, sonnet, sonnet_concise, base, table2), e.g.
            `gpt.42=output/odcv_bench/<key>/combined2x_<ts>/results.json,gpt.69=...`.
        out_dir: Where the PNGs, results.json and the markdown mirror go.
        only: Comma-separated arm keys to draw, e.g. "principle_scoped,grok,gpt".
            The shared-cell intersection is then taken over JUST those arms, so a subset
            view is genuinely comparable rather than a crop of the full one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    parts = results if isinstance(results, (tuple, list)) else str(results).split(",")
    rmap = dict(str(p).partition("=")[::2] for p in parts if "=" in str(p))
    # fire hands a comma list back as a TUPLE, not a string (see docs/GOTCHAS.md), so
    # accept both or `--only a,b` arrives as the repr of a tuple and every key is unknown.
    raw = only if isinstance(only, (tuple, list)) else str(only).split(",")
    picks = tuple(str(k).strip() for k in raw if str(k).strip())
    rows = _collect(rmap, picks)
    seed_arms = [r for r in rows if r["kind"] == "pooled"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    n_common = rows[0]["n_cells"]  # same for every arm by construction
    # Every output path comes from src.utils.figure_path: <out>/<YYYY-MM-DD>_<subject>.
    # A plot outlives the chat that made it, so the date and the arm set are IN the
    # filename rather than in whoever remembers running it.
    subject = f"odcv_arms_seed_mean_{n_common}_cells"
    title = "Synthetic-SFT arms on Qwen3.6-27B: who wrote the answers, and what kind of document"

    # --- overall bars ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    for x, r in enumerate(rows):
        _bar(ax, x, r, r["mr"], r["lo"], r["hi"], 0.62)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [
            r["short"]
            + (
                "\n%d seeds pooled" % r["n_seeds"]
                if r["kind"] == "pooled"
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
    # The baseline gets a rule across the whole axis, not just a bar among bars: every other
    # arm on this chart is read as a distance from it, and a reader should not have to know
    # which bar that is. Drawn under the bars so it never crosses a mark.
    base_row = next((r for r in rows if r["key"] == BASELINE_ARM), None)
    if base_row:
        ax.axhline(
            base_row["mr"],
            color="#111111",
            lw=1.0,
            ls=(0, (5, 4)),
            zorder=1,
            alpha=0.55,
        )
        ax.text(
            len(rows) - 0.4,
            base_row["mr"],
            f" baseline {base_row['mr']:.1f}%",
            va="center",
            ha="left",
            fontsize=7.6,
            color="#111111",
            alpha=0.75,
        )
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
                label="seeds pooled on the design checkpoint axis",
            ),
            Line2D(
                [],
                [],
                color="#555555",
                lw=1.2,
                label="one training (scenario variance only)",
            ),
            # Only the colours this figure actually draws: with --only, listing a family
            # that is not on the chart invites the reader to hunt for a bar that is not there.
            *[
                p
                for c, p in (
                    (RED, Patch(facecolor=RED, label="GPT-written difficult advice")),
                    (BLUE, Patch(facecolor=BLUE, label="grok-written difficult advice")),
                    (
                        GREEN,
                        Patch(
                            facecolor=GREEN,
                            label="Sonnet-written difficult advice (hatched: a variant)",
                        ),
                    ),
                    (
                        TEAL,
                        Patch(
                            facecolor=TEAL,
                            label="Sonnet-written post-action retrospection",
                        ),
                    ),
                    (
                        GRAY,
                        Patch(
                            facecolor=GRAY,
                            hatch="...",
                            edgecolor="#555555",
                            label="no-SFT references",
                        ),
                    ),
                )
                if c in {r["color"] for r in rows}
            ],
        ],
        loc="upper left",
        fontsize=7.8,
        frameon=False,
    )
    ax.set_title(
        f"ODCV misalignment rate on the {n_common} cells EVERY arm scored\n"
        "(the shared exclusion list, then the cells no arm lost)",
        fontsize=10.5,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.98, ha="right", y=0.995)
    foot = " · ".join(
        f"{r['key']}: {r['n_seeds']} seeds" for r in seed_arms
    )
    fig.text(
        0.01,
        0.005,
        (foot + ". " if foot else "")
        + "Each seed is evaluated once, on ONE pass. Every interval comes from src/eval/stats: a "
        "pooled arm's carries seed-to-seed variance, a single arm's does not.",
        fontsize=7.2,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    png = figure_path(out, f"{subject}_bars")
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
    ax.set_xticklabels(
        [
            r["short"]
            + (
                "\n%d seeds pooled" % r["n_seeds"]
                if r["kind"] == "pooled"
                else ""
            )
            for r in vrows
        ],
        fontsize=8.8,
        rotation=12,
        ha="right",
    )
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
            Line2D([], [], color="#111111", lw=2.0, label="seeds pooled (scenario + seed variance)"),
            Line2D(
                [],
                [],
                color="#555555",
                lw=1.2,
                label="one training (scenario variance only)",
            ),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    ax.set_title(
        f"ODCV misalignment rate by variant, on the {n_common} cells EVERY arm scored",
        fontsize=10.5,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.98, ha="right", y=0.995)
    fig.text(
        0.01,
        0.005,
        " · ".join(
            f"{r['key']}: mandated {r['mand'][0]:.1f}% [{r['mand'][1]:.1f}, {r['mand'][2]:.1f}], "
            f"incentivized {r['inc'][0]:.1f}% [{r['inc'][1]:.1f}, {r['inc'][2]:.1f}]"
            for r in rows
        ),
        fontsize=7.2,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    png_v = figure_path(out, f"{subject}_variants")
    fig.savefig(png_v, facecolor="white")
    plt.close(fig)

    # --- mirror + json ------------------------------------------------------------------
    lines = [
        f"# Synthetic-SFT arms: ODCV misalignment, intervals from src/eval/stats",
        "",
        f"Generated {ts}. Cell set: `{CFG}` exclusions. No reruns: every number is "
        "re-summarised from the run's published per-scenario medians.",
        "",
        "## Per run",
        "",
        "One row per training. A multi-seed arm keeps ONE pass per seed (the most complete) "
        "and every seed is scored on the cells they all kept; a single-training arm keeps "
        "every pass it ran. Each row's interval is that ONE training scored alone -- the "
        "arm's bar pools them, so its interval is wider than any row here. `passes` is how "
        "many the published run actually contained.",
        "",
        "| arm | seed | MR | 95% CI | sev | mandated | incentivized | cells | rollouts | passes | source |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for sd, s in r["per_seed"].items():
            ci = f"[{s['lo']:.1f}, {s['hi']:.1f}]"
            mand = f"{s['mand'][0]:.1f}% [{s['mand'][1]:.1f}, {s['mand'][2]:.1f}]"
            inc = f"{s['inc'][0]:.1f}% [{s['inc'][1]:.1f}, {s['inc'][2]:.1f}]"
            lines.append(
                f"| {r['key']} | {sd} | {s['mr']:.1f}% | {ci} | {s['sev']:.2f} | {mand} "
                f"| {inc} | {s['n_scenarios']} | {s['n_rollouts']} "
                f"| {r['n_passes'][str(sd)]} | `{r['sources'][str(sd)]}` |"
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
        "Every interval comes from src/eval/stats via odcv.summarise[_pooled]: scenarios are "
        "the sampled unit, variants are enumerated and mixed 50/50, rollouts average into the "
        "cell. An arm with several seeds pools them on the checkpoint axis, so its interval "
        "ALSO carries seed-to-seed variance -- which is why the pooled arms are wider here "
        "than they were as a mean-of-means, and why the two kinds are now comparable.",
        "",
        f"Plots: `{png.relative_to(ROOT)}`, `{png_v.relative_to(ROOT)}`",
    ]
    md = figure_path(out, f"{subject}_results", ext="md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = [
        {k: v for k, v in r.items() if k not in ("color", "hatch")} for r in rows
    ]
    figure_path(out, f"{subject}_results", ext="json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    for r in rows:
        print(
            f"{r['key']:15s} {r['kind']:9s} {r['mr']:5.1f}%  [{r['lo']:.1f}, {r['hi']:.1f}]"
            + (
                "   seeds "
                + ", ".join(f"{s['mr']:.1f}" for s in r["per_seed"].values())
                if r["kind"] == "pooled"
                else ""
            )
        )
    print(f"wrote {png}\nwrote {png_v}\nwrote {md}")


if __name__ == "__main__":
    fire.Fire(main)
