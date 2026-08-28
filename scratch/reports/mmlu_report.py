# ABOUTME: Aggregate MMLU records across arms into the paired comparison vs the Qwen
# ABOUTME: base: gate check, dose-response plot, per-category plot, markdown mirror.

"""Reporting for the MMLU absolute-benchmark eval.

Reads every arm's graded records, checks that they were all given the *same exam*, and
produces the deliverables:

1. `mmlu_accuracy.png` — accuracy vs synthetic fraction with Wilson intervals, against a
   base-model reference band, plus the paired difference with its non-inferiority margin.
2. `mmlu_by_category.png` — the same dose-response split across MMLU's four subject
   groups, because a uniform drop and a drop concentrated in STEM are different findings.
3. `mmlu_results.md` — the markdown mirror, so every number is greppable without opening
   a PNG.

The parity check in `_load_arm_records` is not a formality. Every statistic here is
*paired* — it assumes arm i's question k and arm j's question k are the same question.
If two arms were run against different subset sizes or a different seed, the pairing is
silently wrong and the intervals are nonsense while still looking reasonable. So the
uid sets are compared explicitly and a mismatch is fatal.

    uv run python scratch/reports/mmlu_report.py --config configs/eval/mmlu.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.capabilities.mmlu.runner import resolve_arms  # noqa: E402
from src.eval.capabilities.mmlu.mmlu import (  # noqa: E402
    CATEGORIES,
    LETTERS,
    mcnemar,
    paired_diff,
    parse_answer,
    score_records,
)
from src.utils import read_jsonl, timestamp, write_run_meta  # noqa: E402

# One data series per panel, so colour carries no categorical load and no legend box is
# needed — the title names the series and points are directly labelled. These are the
# repo's validated palette slots (see src/eval/misalignment/internalization/plots_theme.py); duplicated rather than
# imported to keep src/ independent of internalization/, matching arena_hard_report.py.
_SERIES = "#2a78d6"
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_GRID = "#e6e5e1"
_CRITICAL = "#d03b3b"  # reserved status hue; ships with a label, never colour alone
_SURFACE = "#fcfcfb"


def _regrade(record: dict) -> dict:
    """Re-derive `parsed`/`parse_tier`/`correct` from the stored answer text.

    Grading is a pure function of the saved generation, so it belongs here rather than
    being frozen at generation time. Re-deriving it means an improvement to the parser
    — say, learning that the Tulu-SFT arms end in `\\boxed{}` rather than "Answer:" —
    applies to every historical run for free, with no GPU, no endpoint and no re-spend.
    Freezing `correct` into the record would instead require re-generating 2,850 answers
    to fix a regex.

    Records written before a field existed are tolerated: `answer_letter` has been
    written since the first version, and without it there is nothing to grade against.
    """
    gold = record.get("answer_letter")
    if not gold:
        return record
    n_choices = len(LETTERS) if record.get("n_choices") is None else int(record["n_choices"])
    parsed, tier = parse_answer(record.get("answer", ""), n_choices)
    return {**record, "parsed": parsed, "parse_tier": tier, "correct": parsed == gold}


def _load_arm_records(cfg: DictConfig, arms: list[dict], mode: str) -> dict[str, list[dict]]:
    """Load each arm's graded records, failing loudly if the arms disagree on the exam.

    Returns:
        `{arm_name: records}`, every list sorted by uid and covering an identical uid set.
    """
    root = Path(str(cfg.output_dir)) / mode
    loaded: dict[str, list[dict]] = {}
    for arm in arms:
        path = root / arm["name"] / "records.jsonl"
        if not path.exists():
            continue
        loaded[arm["name"]] = sorted(
            (_regrade(r) for r in read_jsonl(path)), key=lambda r: r["uid"]
        )

    if not loaded:
        raise SystemExit(
            f"No records under {root}. Run the eval first (mode={mode} is inferred "
            f"from the artifact):\n"
            f"  uv run scripts/run_eval.py --target <hf_path> --name mmlu"
        )

    # Paired statistics require identical question sets. A mismatch here produces
    # intervals that look fine and mean nothing, so it is fatal rather than a warning.
    uid_sets = {name: {r["uid"] for r in recs} for name, recs in loaded.items()}
    reference_name, reference = next(iter(uid_sets.items()))
    for name, uids in uid_sets.items():
        if uids != reference:
            only_a = len(reference - uids)
            only_b = len(uids - reference)
            raise SystemExit(
                f"Arms were not given the same questions: {reference_name} has {only_a} "
                f"question(s) {name} lacks, {name} has {only_b} that {reference_name} "
                f"lacks. Paired statistics are invalid across different subsets — re-run "
                f"the smaller arm at the same seed and per_subject, or delete its stale "
                f"records.jsonl."
            )
    return loaded


def _compare(baseline: list[dict], arm: list[dict], cfg: DictConfig) -> dict[str, Any]:
    """Paired comparison of one arm against the baseline over identical questions."""
    stats = cfg.statistics
    base_correct = [bool(r["correct"]) for r in baseline]
    arm_correct = [bool(r["correct"]) for r in arm]
    # Closed form, clustered by subject: the difference of two means has an exact SE, and
    # the spread is taken over SUBJECTS because the 57 are a sample of domains, not the
    # population. Replaced paired_bootstrap_diff (per-question resampling) on 2026-08-28.
    boot = paired_diff(baseline, arm)
    test = mcnemar(base_correct, arm_correct)
    margin = float(cfg.thresholds.max_accuracy_drop_pp) / 100.0
    # Non-inferiority: the WORST plausible drop must be inside the margin. `boot` is
    # oriented arm - baseline, so the worst case is its lower bound.
    return {
        **boot,
        **test,
        "diff_pp": boot["diff"] * 100.0,
        "ci_lower_pp": boot["ci_lower"] * 100.0,
        "ci_upper_pp": boot["ci_upper"] * 100.0,
        "max_plausible_drop_pp": -boot["ci_lower"] * 100.0,
        "non_inferior": boot["ci_lower"] >= -margin,
    }


def _health(scores: dict, cfg: DictConfig) -> list[str]:
    """Instrument-health breaches for one arm, checked before its accuracy is trusted."""
    issues = []
    if scores["parse_rate"] < float(cfg.thresholds.min_parse_rate):
        issues.append(
            f"parse rate {scores['parse_rate']:.1%} below "
            f"{float(cfg.thresholds.min_parse_rate):.0%}"
        )
    if scores["truncation_rate"] > float(cfg.thresholds.max_truncation_rate):
        issues.append(
            f"truncation {scores['truncation_rate']:.1%} above "
            f"{float(cfg.thresholds.max_truncation_rate):.0%} — raise generation.max_tokens"
        )
    return issues


def _style(ax: plt.Axes) -> None:
    """Recessive grid and axes, so the marks carry the figure."""
    ax.grid(True, axis="y", color=_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d3")


def _dose_points(rows: list[dict]) -> tuple[list[float], list[str]]:
    """X positions for the SFT arms: synthetic fraction as a percentage."""
    return [float(r["synthetic_fraction"]) * 100.0 for r in rows], [r["arm"] for r in rows]


def plot_accuracy(
    sft: list[dict],
    base: dict | None,
    comparisons: dict[str, dict],
    cfg: DictConfig,
    path: Path,
    mode: str,
) -> None:
    """Dose-response accuracy panel plus the paired difference panel."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), facecolor="#ffffff")
    xs, names = _dose_points(sft)

    # --- Panel A: absolute accuracy, with the base model as a reference band ---------
    ax = axes[0]
    if base:
        ax.axhspan(
            base["ci_lower"] * 100,
            base["ci_upper"] * 100,
            color=_INK_MUTED,
            alpha=0.10,
            zorder=1,
        )
        ax.axhline(base["mean"] * 100, color=_INK_MUTED, lw=1.1, ls="--", zorder=2)
        # Anchored mid-axis in a BLENDED transform (x in axes fraction, y in data), so
        # the label tracks the reference line vertically while sitting in the horizontal
        # gap between dose points. Anchoring it to a data x — either edge — puts it on
        # top of a marker or a whisker, because the arms cluster near the base line and
        # that is exactly where the label wants to live.
        ax.annotate(
            f"Qwen base  {base['mean']:.1%}",
            xy=(0.62, base["mean"] * 100),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color=_INK_MUTED,
        )

    for x, row in zip(xs, sft):
        lower = (row["mean"] - row["ci_lower"]) * 100
        upper = (row["ci_upper"] - row["mean"]) * 100
        unmatched = row.get("role") == "unmatched_control"
        ax.errorbar(
            x,
            row["mean"] * 100,
            yerr=[[lower], [upper]],
            fmt="o",
            ms=8,
            lw=2,
            capsize=4,
            color=_SERIES,
            # Open marker is a second, non-colour channel marking the arm whose training
            # recipe differs from the rest — it is not on the same dose-response curve.
            mfc="#ffffff" if unmatched else _SERIES,
            mec=_SERIES,
            mew=2,
            zorder=4,
        )
        ax.annotate(
            f"{row['mean']:.1%}",
            xy=(x, row["ci_upper"] * 100),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=_INK,
        )
    if any(r.get("role") == "unmatched_control" for r in sft):
        ax.annotate(
            "open marker = different training recipe (not on the dose curve)",
            xy=(0.5, -0.30),
            xycoords="axes fraction",
            ha="center",
            fontsize=7.5,
            color=_INK_MUTED,
        )

    ax.set_xlabel("Constitution / difficult-advice data in SFT mixture (%)", fontsize=9.5, color=_INK)
    ax.set_ylabel("MMLU accuracy (%)", fontsize=9.5, color=_INK)
    ax.set_title("Absolute accuracy vs the base model", fontsize=10.5, color=_INK, pad=10)
    if xs:
        pad = max(2.0, (max(xs) - min(xs)) * 0.12)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    _style(ax)

    # --- Panel B: paired difference, with the non-inferiority margin -----------------
    ax = axes[1]
    margin = float(cfg.thresholds.max_accuracy_drop_pp)
    diffs = [comparisons[n] for n in names if n in comparisons]
    dx = [x for x, n in zip(xs, names) if n in comparisons]

    # Fix the limits from the DATA before drawing the fail-zone fill. Letting the fill
    # set them zooms the axis out to its own extent and squashes every interval — the
    # differences we are here to read become invisible next to a giant red rectangle.
    if diffs:
        lo = min([c["ci_lower_pp"] for c in diffs] + [-margin])
        hi = max([c["ci_upper_pp"] for c in diffs] + [0.0])
        span = max(hi - lo, 1.0)
        bottom, top = lo - span * 0.18, hi + span * 0.18
    else:
        bottom, top = -margin * 2, margin * 2
    ax.set_ylim(bottom, top)
    ax.axhspan(bottom, -margin, color=_CRITICAL, alpha=0.07, zorder=1)
    ax.axhline(-margin, color=_CRITICAL, lw=1.1, ls="--", zorder=2)
    ax.axhline(0, color=_INK_MUTED, lw=1.0, zorder=2)

    for x, cmp_ in zip(dx, diffs):
        ax.errorbar(
            x,
            cmp_["diff_pp"],
            yerr=[[cmp_["diff_pp"] - cmp_["ci_lower_pp"]], [cmp_["ci_upper_pp"] - cmp_["diff_pp"]]],
            fmt="o",
            ms=8,
            lw=2,
            capsize=4,
            color=_SERIES,
            mfc=_SERIES,
            mec=_SERIES,
            zorder=4,
        )
    if dx:
        ax.annotate(
            f"regression margin  −{margin:.0f}pp",
            xy=(min(dx), -margin),
            xytext=(0, -6),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.5,
            color=_CRITICAL,
        )
    ax.set_xlabel("Constitution / difficult-advice data in SFT mixture (%)", fontsize=9.5, color=_INK)
    ax.set_ylabel("Accuracy difference vs base (pp)", fontsize=9.5, color=_INK)
    ax.set_title("Paired difference, 95% CI over shared questions", fontsize=10.5, color=_INK, pad=10)
    if xs:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    _style(ax)

    n = sft[0]["n"] if sft else 0
    n_subjects = len(sft[0]["by_subject"]) if sft else 0
    fig.suptitle(
        f"MMLU capability check — constitution-SFT arms ({mode})", fontsize=12.5, color=_INK, y=1.03
    )
    # Derived from the records actually loaded, never from the config: a run made with
    # `--per_subject` set on the command line would otherwise be captioned with the
    # config's value, putting a wrong n on the figure.
    fig.text(
        0.5,
        -0.06,
        f"n = {n} questions, stratified across {n_subjects} subjects · "
        f"identical subset and decoding for every arm · "
        f"temperature {float(cfg.generation.temperature):g}",
        ha="center",
        fontsize=8,
        color=_INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def plot_by_category(
    sft: list[dict], base: dict | None, path: Path, mode: str
) -> None:
    """Small multiples: the dose-response curve within each MMLU subject group.

    Small multiples rather than four coloured series on one axis: the categories are not
    being compared to each other, each is being compared to its own base reference, and
    four overlapping series with four error bars is unreadable at this size.
    """
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4), sharey=True, facecolor="#ffffff")
    xs, _ = _dose_points(sft)

    for ax, category in zip(axes, CATEGORIES):
        if base and category in base["by_category"]:
            ref = base["by_category"][category]
            ax.axhspan(ref["ci_lower"] * 100, ref["ci_upper"] * 100, color=_INK_MUTED, alpha=0.10)
            ax.axhline(ref["mean"] * 100, color=_INK_MUTED, lw=1.0, ls="--")
        for x, row in zip(xs, sft):
            block = row["by_category"].get(category)
            if not block:
                continue
            ax.errorbar(
                x,
                block["mean"] * 100,
                yerr=[
                    [(block["mean"] - block["ci_lower"]) * 100],
                    [(block["ci_upper"] - block["mean"]) * 100],
                ],
                fmt="o",
                ms=6,
                lw=1.6,
                capsize=3,
                color=_SERIES,
                mfc="#ffffff" if row.get("role") == "unmatched_control" else _SERIES,
                mec=_SERIES,
                mew=1.6,
            )
        # Every arm answers the same questions, so any arm's count is the panel's n.
        n = int(sft[0]["by_category"].get(category, {}).get("n", 0)) if sft else 0
        ax.set_title(
            f"{category.replace('_', ' ')}\nn={n}", fontsize=9.5, color=_INK, pad=8
        )
        ax.set_xlabel("synthetic %", fontsize=8.5, color=_INK_MUTED)
        if xs:
            pad = max(2.0, (max(xs) - min(xs)) * 0.15)
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
        _style(ax)
    axes[0].set_ylabel("MMLU accuracy (%)", fontsize=9.5, color=_INK)

    fig.suptitle(
        f"MMLU by subject group — dashed line and band = Qwen base ({mode})",
        fontsize=11.5,
        color=_INK,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def _markdown(
    scores: dict[str, dict],
    comparisons: dict[str, dict],
    health: dict[str, list[str]],
    baseline_arm: str,
    cfg: DictConfig,
    mode: str,
    subset_note: str,
) -> str:
    """Build the greppable markdown mirror."""
    margin = float(cfg.thresholds.max_accuracy_drop_pp)
    lines = [
        f"# MMLU capability check — constitution-SFT arms ({mode})",
        "",
        f"Generated {timestamp()} · baseline = `{baseline_arm}` · {subset_note}",
        "",
        "Absolute benchmark complementing the pairwise preference eval in "
        "`configs/eval/arena_hard.yaml`, which by construction cannot detect both arms "
        "degrading together.",
        "",
        "## Headline",
        "",
        "| arm | synthetic % | n | accuracy | 95% CI | Δ vs base (pp) | paired 95% CI | "
        "McNemar p | non-inferior | parse rate | truncation |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in scores.items():
        frac = s.get("synthetic_fraction")
        frac_txt = "—" if frac is None else f"{float(frac) * 100:.0f}%"
        cmp_ = comparisons.get(name)
        if cmp_:
            delta = f"{cmp_['diff_pp']:+.1f}"
            ci = f"[{cmp_['ci_lower_pp']:+.1f}, {cmp_['ci_upper_pp']:+.1f}]"
            p = f"{cmp_['p_value']:.3f}"
            verdict = "**PASS**" if cmp_["non_inferior"] else "**FAIL**"
        else:
            delta = ci = p = verdict = "— (baseline)"
        lines.append(
            f"| `{name}` | {frac_txt} | {s['n']} | {s['mean']:.1%} | "
            f"[{s['ci_lower']:.1%}, {s['ci_upper']:.1%}] | {delta} | {ci} | {p} | {verdict} | "
            f"{s['parse_rate']:.1%} | {s['truncation_rate']:.1%} |"
        )

    lines += [
        "",
        f"Non-inferiority gate: the lower bound of the paired difference must sit above "
        f"−{margin:.0f}pp. A point estimate near zero is not sufficient — the interval has "
        f"to exclude a regression worth caring about.",
        "",
        "## Instrument health",
        "",
    ]
    breaches = {k: v for k, v in health.items() if v}
    if breaches:
        lines.append(
            "**These are harness failures, not model results. Fix and re-run rather than "
            "reporting the accuracy above.**"
        )
        lines.append("")
        for name, issues in breaches.items():
            lines.append(f"- `{name}`: " + "; ".join(issues))
    else:
        lines.append(
            f"All arms within gates (parse rate ≥ {float(cfg.thresholds.min_parse_rate):.0%}, "
            f"truncation ≤ {float(cfg.thresholds.max_truncation_rate):.0%})."
        )

    lines += ["", "## Reasoning-trace health", "",
              "Empty `<think>` blocks are CLAUDE.md gotcha 2 — the collapse plain SFT induces. "
              "A thinking arm at ~0 words has stopped reasoning regardless of its accuracy.",
              "",
              "| arm | mean think words | empty-think rate | parse tiers |",
              "|---|---|---|---|"]
    for name, s in scores.items():
        tiers = ", ".join(f"{k}={v}" for k, v in sorted(s["parse_tiers"].items()))
        lines.append(
            f"| `{name}` | {s['mean_think_words']:.0f} | {s['empty_think_rate']:.1%} | {tiers} |"
        )

    lines += ["", "## By subject group", "", "| arm | " + " | ".join(
        c.replace("_", " ") for c in CATEGORIES) + " |",
        "|---|" + "---|" * len(CATEGORIES)]
    for name, s in scores.items():
        cells = []
        for category in CATEGORIES:
            block = s["by_category"].get(category)
            cells.append(f"{block['mean']:.1%} (n={block['n']})" if block else "—")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    per_subject = min(
        (b["n"] for block in scores.values() for b in block["by_subject"].values()),
        default=0,
    )
    lines += [
        "",
        "## Per-subject accuracy",
        "",
        f"Directional only — at ~{per_subject} questions per subject a single question "
        f"moves a subject by {100 / per_subject if per_subject else 0:.0f}pp. Read the "
        f"subject groups above for anything load-bearing.",
        "",
        "| subject | " + " | ".join(f"`{n}`" for n in scores) + " |",
        "|---|" + "---|" * len(scores),
    ]
    all_subjects = sorted({s for block in scores.values() for s in block["by_subject"]})
    for subject in all_subjects:
        cells = [
            f"{scores[n]['by_subject'][subject]['mean']:.0%}"
            if subject in scores[n]["by_subject"]
            else "—"
            for n in scores
        ]
        lines.append(f"| {subject} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main(
    config: str = "configs/eval/mmlu.yaml",
    nothink: bool = False,
) -> None:
    """Build the MMLU comparison report from graded records.

    Args:
        config: Path to the MMLU eval config.
        nothink: Report on the `nothink/` results tree instead of `think/`.
    """
    cfg = OmegaConf.load(config)
    mode = "nothink" if nothink else "think"
    ladder = resolve_arms(cfg)
    records = _load_arm_records(cfg, ladder, mode)

    order = [a for a in ladder if a["name"] in records]
    scores: dict[str, dict] = {}
    for arm in order:
        block = score_records(records[arm["name"]])
        block |= {
            "arm": arm["name"],
            "synthetic_fraction": arm["synthetic_fraction"],
            "role": arm["role"],
            "adapter": arm["adapter"],
        }
        scores[arm["name"]] = block

    baseline_arm = str(cfg.baseline_arm)
    if baseline_arm not in scores:
        raise SystemExit(
            f"Baseline arm {baseline_arm!r} has no records under "
            f"{Path(str(cfg.output_dir)) / mode}. Every comparison is against it, so run "
            f"it first:\n  uv run scripts/run_eval.py --target <its hf_path> --name mmlu"
        )

    comparisons = {
        name: _compare(records[baseline_arm], records[name], cfg)
        for name in scores
        if name != baseline_arm
    }
    health = {name: _health(block, cfg) for name, block in scores.items()}

    out_dir = Path(str(cfg.output_dir)) / "report" / f"{mode}_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_block = scores.get(baseline_arm)
    sft = [
        s for s in scores.values()
        if s["arm"] != baseline_arm and s.get("synthetic_fraction") is not None
    ]
    sft.sort(key=lambda s: float(s["synthetic_fraction"]))

    if sft:
        plot_accuracy(sft, base_block, comparisons, cfg, out_dir / "mmlu_accuracy.png", mode)
        plot_by_category(sft, base_block, out_dir / "mmlu_by_category.png", mode)

    # Counted off the loaded records rather than the config, so a `--per_subject`
    # override on the eval run cannot leave a wrong n in the mirror.
    n = base_block["n"]
    subset_note = (
        f"{n} questions across {len(base_block['by_subject'])} subjects, seed "
        f"{int(cfg.seed)}, {int(cfg.prompt.n_shot)}-shot"
    )
    (out_dir / "mmlu_results.md").write_text(
        _markdown(scores, comparisons, health, baseline_arm, cfg, mode, subset_note)
    )
    (out_dir / "mmlu_scores.json").write_text(
        json.dumps({"scores": scores, "comparisons": comparisons, "health": health}, indent=2)
    )
    write_run_meta(
        out_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"mode": mode, "baseline_arm": baseline_arm, "arms": list(scores)},
    )

    print(f"=== MMLU report ({mode}) — baseline {baseline_arm} ===")
    for name, block in scores.items():
        cmp_ = comparisons.get(name)
        tail = ""
        if cmp_:
            verdict = "PASS" if cmp_["non_inferior"] else "FAIL"
            tail = (
                f"  Δ {cmp_['diff_pp']:+.1f}pp "
                f"[{cmp_['ci_lower_pp']:+.1f}, {cmp_['ci_upper_pp']:+.1f}]  "
                f"p={cmp_['p_value']:.3f}  {verdict}"
            )
        print(f"  {name:18} {block['mean']:6.1%} (n={block['n']}){tail}")
        for issue in health[name]:
            print(f"      !! {issue}")
    print(f"\n>>> {out_dir / 'mmlu_results.md'}")
    if sft:
        print(f">>> {out_dir / 'mmlu_accuracy.png'}")
        print(f">>> {out_dir / 'mmlu_by_category.png'}")


if __name__ == "__main__":
    fire.Fire(main)
