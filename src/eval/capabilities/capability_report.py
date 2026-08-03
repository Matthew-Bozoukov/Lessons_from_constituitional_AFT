# ABOUTME: Turn Arena-Hard judgments into the capability-eval deliverables: win-rate
# ABOUTME: plot, style-drift plot, markdown mirror, and a hashed artefact manifest.

"""Reporting for the capability regression eval.

Reads every arm's judgments out of the vendored harness and produces spec §13's
relative-family deliverables:

1. Style-controlled and uncontrolled win rate vs arm A, with CIs, per arm, with
   `hard_prompt` and `creative_writing` reported separately — the core plot, mirroring
   the reference work's near-50% line.
2. Judge-agreement figures from the §4 validation, for the methods section.
4. Style-drift metrics per arm, plotted against mixture ratio.
5. Per-subcategory win-rate breakdown within `hard_prompt`, marked directional.
7. A frozen artefact manifest: pinned base and per-arm checkpoints, prompt-set content
   hash, judge model id, rubric hash, and the observed tie rates.

Deliverables 3 and 6 (absolute benchmark deltas, and the alignment-vs-capability Pareto
scatter that depends on them) are out of scope for this build.

Item 7 matters more than it looks. Preference numbers are not comparable across judge
versions or rubric edits, and the pinned judge is a `-preview` endpoint. A number without
its methodology version attached is not reproducible six months from now.

    uv run python src/eval/capabilities/capability_report.py --config configs/eval/capability.yaml
"""

from __future__ import annotations

import hashlib
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

from src.eval.capabilities.capability_stats import (  # noqa: E402
    battles_from_judgments,
    evaluate_arm,
    subcategory_breakdown,
)
from src.utils import git_sha, read_jsonl, timestamp, write_run_meta  # noqa: E402

# Colour encodes the SERIES (controlled vs uncontrolled), not the arm. Arm identity is
# already carried by x-position, and spending colour on it would leave the legend unable
# to mean anything. Categorical slots 1 and 2, CVD-validated against the light surface
# (worst adjacent pair ΔE 24.7 protan, 33.6 normal). Shape is a redundant second channel
# so the pair never depends on colour alone.
_SERIES = {
    "controlled": {"colour": "#2a78d6", "marker": "o", "label": "style-controlled (primary)"},
    "uncontrolled": {"colour": "#eb6834", "marker": "s", "label": "uncontrolled"},
}
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_CRITICAL = "#d03b3b"  # reserved status hue, never reused as a series colour
_SERIES_1 = "#2a78d6"


def _sha256(path: Path) -> str:
    """Content hash of a file, for the frozen artefact manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(vendor: Path, bench: str, judge: str, arm: str) -> list[dict]:
    """Read one arm's judgment records, or an empty list if it has not been judged."""
    path = vendor / "data" / bench / "model_judgment" / judge / f"{arm}.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def _answer_meta(vendor: Path, bench: str, arm: str) -> dict[str, dict]:
    """Read `{uid: style metadata}` for one arm's answers."""
    path = vendor / "data" / bench / "model_answer" / f"{arm}.jsonl"
    if not path.exists():
        return {}
    return {rec["uid"]: rec["metadata"] for rec in read_jsonl(path)}


def _style_summary(meta: dict[str, dict], uids: set[str]) -> dict[str, float]:
    """Mean style features over a set of answers."""
    rows = [meta[u] for u in uids if u in meta]
    if not rows:
        return {}
    n = len(rows)
    return {
        "mean_token_len": sum(r["token_len"] for r in rows) / n,
        "mean_header_count": sum(sum(r["header_count"].values()) for r in rows) / n,
        "mean_list_count": sum(sum(r["list_count"].values()) for r in rows) / n,
        "mean_bold_count": sum(sum(r["bold_count"].values()) for r in rows) / n,
    }


def _plot_win_rates(results: dict, cfg: DictConfig, out_dir: Path) -> Path:
    """Deliverable 1: controlled vs uncontrolled win rate per arm, per slice.

    The 50% line and the threshold band are drawn because this plot is read for whether
    intervals *clear a line*, not for where the dots sit. Absent the band, a reader has
    to do the comparison in their head and will get it wrong.
    """
    slices = [s for s in ("hard_prompt", "creative_writing") if any(
        s in r["by_slice"] for r in results.values()
    )]
    if not slices:
        raise SystemExit("No judged slices found; nothing to plot.")

    threshold = float(cfg.thresholds.relative.win_rate_ci_lower_min) * 100
    fig, axes = plt.subplots(1, len(slices), figsize=(5.2 * len(slices), 4.4), squeeze=False)

    for ax, slice_name in zip(axes[0], slices):
        arms = [
            a
            for a in results
            if slice_name in results[a]["by_slice"]
            and results[a]["synthetic_fraction"] is not None
        ]
        arms.sort(key=lambda a: results[a]["synthetic_fraction"])
        # Mixture ratio is a genuine quantity, so it gets a real numeric axis rather than
        # evenly-spaced categories. That is what turns a row of dots into a dose-response
        # curve — the actual deliverable (spec §7), and how the reference work plots it.
        fractions = [results[a]["synthetic_fraction"] * 100 for a in arms]

        # 50 is the whole story: this is a treated checkpoint against its untreated
        # sibling, so the reference line is the expected result, not a target to beat.
        ax.axhline(50, color=_INK_MUTED, lw=1.0, ls="--", zorder=1)
        ax.axhline(threshold, color=_CRITICAL, lw=1.0, ls=":", zorder=1, alpha=0.7)

        for key, spec in _SERIES.items():
            means = [results[a]["by_slice"][slice_name][key]["mean"] * 100 for a in arms]
            lo = [
                means[i] - results[a]["by_slice"][slice_name][key]["ci_lower"] * 100
                for i, a in enumerate(arms)
            ]
            hi = [
                results[a]["by_slice"][slice_name][key]["ci_upper"] * 100 - means[i]
                for i, a in enumerate(arms)
            ]
            ax.errorbar(
                fractions,
                means,
                yerr=[lo, hi],
                fmt=spec["marker"] + "-",
                ms=6,
                lw=1.6,
                mfc=spec["colour"],
                mec=spec["colour"],
                ecolor=spec["colour"],
                color=spec["colour"],
                capsize=3,
                elinewidth=1.4,
                zorder=3,
                label=spec["label"],
            )

        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(-2, max(fractions) + 4 if fractions else 50)
        ax.set_xticks(fractions)
        ax.set_xlabel("Difficult-advice percentage (%)", fontsize=9.5, color=_INK)
        ax.set_ylabel("Win rate (%)", fontsize=9.5, color=_INK)
        n_primary = results[arms[0]]["by_slice"][slice_name]["controlled"]["n"] if arms else 0
        ax.set_title(
            f"{slice_name}  ·  n≈{n_primary} prompts", fontsize=9, color=_INK_MUTED, pad=8
        )
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95, edgecolor="#e3e2df")
        ax.grid(axis="y", alpha=0.18, zorder=0)
        ax.tick_params(colors=_INK_MUTED, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d3")

    baseline_pct = next(
        (
            r["synthetic_fraction"] * 100
            for a, r in results.items()
            if a == str(cfg.baseline_arm) and r["synthetic_fraction"] is not None
        ),
        None,
    )
    ref = f"{baseline_pct:.0f}% arm" if baseline_pct is not None else str(cfg.baseline_arm)
    fig.suptitle("LMSYS SxS: Win Rate vs Baseline", fontsize=12.5, color=_INK, y=1.02)
    fig.text(
        0.5,
        0.965,
        f"50 = tie, higher is better  ·  baseline = the {ref}",
        ha="center",
        fontsize=8.5,
        color=_INK_MUTED,
        style="italic",
    )
    fig.tight_layout()
    path = out_dir / "win_rates.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return path


def _plot_style_drift(results: dict, out_dir: Path) -> Path | None:
    """Deliverable 4: style drift against mixture ratio.

    Expect drift to grow monotonically with synthetic fraction. If it does, that is a
    clean supporting result; if the most extreme arm shows large drift but a flat
    controlled win rate, the honest reading is "capability preserved, voice changed".
    """
    arms = [a for a in results if results[a]["synthetic_fraction"] is not None
            and results[a].get("style", {}).get("hard_prompt")]
    if len(arms) < 2:
        return None
    arms.sort(key=lambda a: results[a]["synthetic_fraction"])
    fractions = [results[a]["synthetic_fraction"] for a in arms]

    metrics = ["mean_token_len", "mean_header_count", "mean_list_count", "mean_bold_count"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.6))
    # Small multiples, one measure each: these quantities have different units, so a
    # shared axis would be meaningless and a second y-axis would be worse.
    for ax, metric in zip(axes, metrics):
        values = [results[a]["style"]["hard_prompt"].get(metric, 0.0) for a in arms]
        # Single series per panel, so the panel title names it and no legend is needed.
        ax.plot(fractions, values, "o-", color=_SERIES_1, lw=2.0, ms=8)
        ax.set_xlabel("synthetic fraction", color=_INK_MUTED, fontsize=9)
        ax.set_title(metric.replace("mean_", "").replace("_", " "), fontsize=10, color=_INK)
        ax.grid(alpha=0.18)
        ax.tick_params(colors=_INK_MUTED, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d3")
    fig.suptitle(
        "Style drift vs mixture ratio (hard_prompt) — measured, not controlled for",
        fontsize=11,
        color=_INK,
    )
    fig.tight_layout()
    path = out_dir / "style_drift.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _markdown(results: dict, cfg: DictConfig, validation: dict | None, manifest: dict) -> str:
    """Markdown mirror so every number is greppable without opening a PNG."""
    threshold = float(cfg.thresholds.relative.win_rate_ci_lower_min)
    lines = [
        "<!-- ABOUTME: Capability regression eval results (relative family). -->",
        "<!-- ABOUTME: Generated by src/eval/capabilities/capability_report.py. -->",
        "",
        "# Capability regression eval — results",
        "",
        f"- Base model: `{cfg.base_model}`",
        f"- Baseline arm: `{cfg.baseline_arm}` (all win rates are *vs this arm*)",
        f"- Judge: `{manifest['judge_model']}` (rubric sha256 `{manifest['rubric_sha256'][:12]}`)",
        f"- Prompt set: `{cfg.bench_name}` (sha256 `{manifest['question_set_sha256'][:12]}`)",
        f"- Gate: style-controlled 95% CI lower bound ≥ {threshold:.0%} on `hard_prompt`",
        "",
        "**50% is the target, not 100%** — this is a treated checkpoint measured against",
        "its own untreated sibling, so a result near 50% means no regression.",
        "",
        "> Scope: relative family only. Absolute benchmarks (IFEval, MMLU-Pro, GSM8K,",
        "> HumanEval+) are deferred, so this cannot detect both arms degrading together.",
        "",
        "## Win rates",
        "",
        "| arm | mixture | slice | controlled | 95% CI | uncontrolled | style gap | win/tie/loss | swap | pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in sorted(results, key=lambda a: (results[a]["synthetic_fraction"] is None,
                                              results[a]["synthetic_fraction"] or 0.0)):
        res = results[arm]
        for slice_name, block in res["by_slice"].items():
            c, u, s = block["controlled"], block["uncontrolled"], block["split"]
            lines.append(
                f"| `{arm}` | {res['label']} | {slice_name} | "
                f"{c['mean']:.1%} | [{c['ci_lower']:.1%}, {c['ci_upper']:.1%}] | "
                f"{u['mean']:.1%} | {block['style_gap_pp']:+.1f}pp | "
                f"{s['win_rate']:.0%}/{s['tie_rate']:.0%}/{s['loss_rate']:.0%} | "
                f"{s['swap_consistency']:.0%} | "
                f"{'PASS' if block['passes'] else '**FAIL**'} |"
            )

    lines += [
        "",
        "`style gap` = uncontrolled − controlled. A large positive gap means the arm's",
        "apparent standing came from length and formatting rather than substance.",
        "`swap` is swap consistency: how often the judge gave the same verdict with the",
        "responses presented in the opposite order. Low values mean the judge is reading",
        "position, not quality.",
        "",
    ]

    degenerate = {
        (arm, slice_name): block["controlled"]["degenerate_features"]
        for arm, res in results.items()
        for slice_name, block in res["by_slice"].items()
        if block["controlled"].get("degenerate_features")
    }
    if degenerate:
        lines += [
            "### Style control could not act on every feature",
            "",
            "These features had no variance across prompts, so length/formatting and model",
            "identity are the same column and the regression cannot separate them. The",
            "'controlled' number is uncontrolled in these dimensions:",
            "",
        ]
        lines += [f"- `{arm}` / {slice_name}: {', '.join(feats)}" for (arm, slice_name), feats in degenerate.items()]
        lines.append("")

    if validation:
        lines += [
            "## Judge validation (spec §4)",
            "",
            f"- Primary judge: `{validation['primary_judge']}`",
            f"- Reference judge: `{validation['reference_judge']}` "
            f"(third family — Claude generated our corpus, so a Claude validator would",
            f"  import the self-preference confound we avoided by choosing Gemini)",
            f"- Compared on {validation['n_compared']} `{validation['slice']}` questions "
            f"from the `{validation['comparison_arm']}` comparison",
            "",
            f"| metric | value | threshold | |",
            f"|---|---|---|---|",
            f"| verdict agreement | {validation['verdict_agreement']:.1%} | "
            f"≥ {validation['agreement_threshold']:.0%} | "
            f"{'PASS' if validation['verdict_agreement'] >= validation['agreement_threshold'] else '**FAIL**'} |",
            f"| win-rate gap | {validation['win_rate_gap_pp']:.1f}pp | "
            f"≤ {validation['gap_threshold_pp']:.0f}pp | "
            f"{'PASS' if validation['win_rate_gap_pp'] <= validation['gap_threshold_pp'] else '**FAIL**'} |",
            "",
        ]

    if any(res.get("subcategories") for res in results.values()):
        lines += [
            "## Per-subcategory breakdown — DIRECTIONAL ONLY",
            "",
            "At n≈100 per cell the interval is roughly ±8pp even with a healthy tie rate.",
            "These say where to look, not whether a category regressed. **Do not gate on them.**",
            "",
            "| arm | subcategory | n | win rate | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for arm, res in results.items():
            for sub, block in (res.get("subcategories") or {}).items():
                lines.append(
                    f"| `{arm}` | {sub} | {block['n']} | {block['mean']:.1%} | "
                    f"[{block['ci_lower']:.1%}, {block['ci_upper']:.1%}] |"
                )
        lines.append("")

    lines += [
        "## Frozen artefact manifest (deliverable 7)",
        "",
        "Preference numbers are not comparable across judge versions or rubric edits, and",
        "the pinned judge is a `-preview` endpoint that may be retired on short notice.",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
        "## Stopping rule (spec §9)",
        "",
        f"Staged sampling: {OmegaConf.to_container(cfg.staging, resolve=True)}. Extending an",
        "inconclusive stage is optional stopping and inflates type-I error. This is",
        "acceptable for a non-inferiority guardrail **only because it is disclosed here**.",
        "Do not present these intervals as fixed-n intervals.",
        "",
    ]
    return "\n".join(lines)


def main(
    config: str = "configs/eval/capability.yaml",
    validation_json: str = "",
) -> None:
    """Build the capability-eval report from judgments already on disk.

    Args:
        config: Path to the capability-eval config.
        validation_json: Optional path to a `judge_validation.json` from
            `capability_judge.py --mode validate`, to fold into the methods section.
    """
    cfg = OmegaConf.load(config)
    vendor = Path(cfg.vendor_dir)
    judge = str(cfg.judge.model)
    baseline = str(cfg.baseline_arm)
    stats = cfg.statistics
    threshold = float(cfg.thresholds.relative.win_rate_ci_lower_min)

    question_file = vendor / "data" / cfg.bench_name / "question.jsonl"
    questions = {q["uid"]: q for q in read_jsonl(question_file)}
    baseline_meta = _answer_meta(vendor, str(cfg.bench_name), baseline)
    if not baseline_meta:
        raise SystemExit(
            f"No answers for the baseline arm {baseline!r}. Generate them first with "
            f"capability_gen.py — every comparison is against this arm."
        )

    results: dict[str, Any] = {}
    for arm_cfg in cfg.arms:
        arm = str(arm_cfg.name)
        records = _load(vendor, str(cfg.bench_name), judge, arm)
        if not records:
            continue
        model_meta = _answer_meta(vendor, str(cfg.bench_name), arm)
        battles = battles_from_judgments(records)
        fraction = arm_cfg.synthetic_fraction

        block: dict[str, Any] = {
            "synthetic_fraction": fraction,
            "label": "base, no SFT" if fraction is None else f"{(1 - fraction) * 100:.0f}/{fraction * 100:.0f}",
            "adapter": arm_cfg.adapter,
            "by_slice": {},
            "style": {},
            "subcategories": {},
        }
        for slice_name in ("hard_prompt", "creative_writing"):
            subset = [b for b in battles if b["category"] == slice_name]
            if not subset:
                continue
            block["by_slice"][slice_name] = evaluate_arm(
                subset,
                model_meta,
                baseline_meta,
                threshold=threshold,
                rounds=int(stats.bootstrap_rounds),
                alpha=float(stats.alpha),
                seed=int(cfg.seed),
            )
            block["style"][slice_name] = _style_summary(
                model_meta, {b["uid"] for b in subset}
            )
            if slice_name == "hard_prompt":
                block["subcategories"] = subcategory_breakdown(
                    subset, questions, rounds=int(stats.bootstrap_rounds),
                    alpha=float(stats.alpha), seed=int(cfg.seed),
                )
        if block["by_slice"]:
            results[arm] = block

    if not results:
        raise SystemExit(
            f"No judgments found under {vendor}/data/{cfg.bench_name}/model_judgment/{judge}/. "
            f"Run capability_judge.py first."
        )

    out_dir = Path(cfg.output_dir) / "report" / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = json.loads(Path(validation_json).read_text()) if validation_json else None

    manifest = {
        "generated": timestamp(),
        "git_sha": git_sha(),
        "base_model": str(cfg.base_model),
        "baseline_arm": baseline,
        "judge_model": judge,
        "judge_extra_body": OmegaConf.to_container(cfg.judge.extra_body, resolve=True)
        if cfg.judge.get("extra_body")
        else None,
        "bench_name": str(cfg.bench_name),
        "question_set_sha256": _sha256(question_file),
        "rubric_sha256": _sha256(vendor / "utils" / "judge_utils.py"),
        "arena_hard_upstream_sha": "196f6b826783b3da7310e361a805fa36f0be83f3",
        "arms": {
            arm: {
                "adapter": res["adapter"],
                "synthetic_fraction": res["synthetic_fraction"],
                "observed_tie_rate": {
                    s: b["split"]["tie_rate"] for s, b in res["by_slice"].items()
                },
                "n_prompts": {s: b["controlled"]["n"] for s, b in res["by_slice"].items()},
            }
            for arm, res in results.items()
        },
        "decoding": OmegaConf.to_container(cfg.generation, resolve=True),
        "statistics": OmegaConf.to_container(stats, resolve=True),
        "scope_note": "Relative family only; absolute benchmarks deferred.",
    }

    win_plot = _plot_win_rates(results, cfg, out_dir)
    drift_plot = _plot_style_drift(results, out_dir)

    (out_dir / "capability_results.md").write_text(_markdown(results, cfg, validation, manifest))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True))

    print(f">>> report:      {out_dir / 'capability_results.md'}")
    print(f">>> win rates:   {win_plot}")
    if drift_plot:
        print(f">>> style drift: {drift_plot}")
    print()
    for arm, res in results.items():
        for slice_name, block in res["by_slice"].items():
            c = block["controlled"]
            print(
                f"  {arm:16} {slice_name:17} controlled={c['mean']:.1%} "
                f"[{c['ci_lower']:.1%}, {c['ci_upper']:.1%}] "
                f"tie={block['split']['tie_rate']:.0%} "
                f"{'PASS' if block['passes'] else 'FAIL'}"
            )

    failures = [
        (arm, s) for arm, res in results.items()
        for s, b in res["by_slice"].items()
        if not b["passes"] and s == str(cfg.thresholds.relative.primary_slice)
    ]
    if failures:
        print(
            f"\n  {len(failures)} arm(s) breached the primary gate. Per spec §3 this blocks "
            f"claiming the alignment result FOR THOSE ARMS — it is not grounds for dropping "
            f"them from the writeup. A regression locates the ceiling on usable mixture "
            f"ratio, which is itself a result."
        )


if __name__ == "__main__":
    fire.Fire(main)
