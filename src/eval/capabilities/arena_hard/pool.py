# ABOUTME: THE Arena-Hard comparison: judge every arm of the invocation against the shared
# ABOUTME: baseline and publish one leaderboard as `<date>-ah-vs-<baseline>`.

"""What `uv run evals --name arena_hard --reference R --target A B C` produces at the end.

Arena-Hard is a comparison, so this is where its results are made. An arm is a set of
answers and nothing more (runner.py); a win rate is a fact about (arm, baseline, exam) and
belongs to the comparison that produced it. Judging therefore happens HERE, over arms that
are already published — which also means a crash in judging costs only the judging, since
re-pooling reads answers that are already on the Hub.

**The name.** Arena-Hard is a STAR, not a mesh: every arm is judged against one baseline
and no arm against another. The arms of an invocation share exactly one thing — that
baseline — so that is what the artifact is named for. ODCV's rule does not transfer: it
pools seed replicates, which share a mixture subject, so dropping the seed leaves a name
that still describes every member. Here `qwen36-8-da-20`, `qwen36-8-courtroom-20` and
`qwen36-8-tulu-0` have no common tail at all, and stripping what differs would leave the
base model and nothing else.

What the name cannot carry is the question subset, so a second ladder against one baseline
on the same day over a different subset would collide. `check_distinct` catches that before
either is published; the subset itself is in `metadata/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.eval.capabilities.arena_hard import arena_hard_judge
from src.eval.capabilities.arena_hard import runner as arena_hard_runner
from src.eval.capabilities.arena_hard.runner import bench_answers_dir, isolate_harness, register
from src.eval.capabilities.arena_hard.arena_hard_stats import battles_from_judgments, evaluate_arm
from src.eval.layout import publish_layout
from src.utils import read_jsonl


def _arm_meta(run: dict[str, Any]) -> dict:
    """One published arm's `metadata/sources.json` — what it is and where it came from."""
    path = Path(run["out_dir"]) / "metadata" / "sources.json"
    assert path.exists(), (
        f"{run['target']}: no metadata/sources.json at {path}. Every arena_hard arm writes "
        "one; a run dir without it did not come from this eval's runner.")
    return json.loads(path.read_text())


def _validation_report(cfg, result: dict) -> dict:
    """Report auxiliary agreement separately from mandatory primary completion."""
    smoke = bool(cfg.get("smoke", False))
    policy = str(cfg.judge_validation.get("policy", "gate"))
    expected = int(cfg.judge_validation.n_questions)
    complete = result["n_compared"] == expected
    assessed = policy == "gate" and complete and not smoke
    thresholds_met = result.get("thresholds_met_on_sample", result["passes"])
    return {**result, "policy": policy, "n_expected": expected,
            "coverage_status": "complete" if complete else "incomplete",
            "calibration_status": ("passed" if thresholds_met else "failed") if assessed else "not_assessed",
            "calibration_gate_applies": policy == "gate" and not smoke,
            "diagnostic_only": policy == "diagnostic" or smoke,
            "thresholds_met_on_sample": thresholds_met,
            "passes": bool(thresholds_met) if assessed else None}


def _require_validation(cfg, result: dict) -> None:
    if not result["primary_complete"]:
        raise ValueError("Arena-Hard primary judge has incomplete coverage")
    if str(cfg.judge_validation.get("policy", "gate")) == "diagnostic":
        return
    if result["coverage_status"] != "complete":
        raise ValueError("Arena-Hard judge validation has incomplete coverage")
    if not bool(cfg.get("smoke", False)):
        if int(cfg.judge_validation.n_questions) != 100:
            raise ValueError("Full Arena-Hard judge validation requires 100 questions; use smoke for wiring checks")
        if not result["passes"]:
            raise ValueError("Arena-Hard judge validation failed its full calibration gate")


def _generation_health(run: dict[str, Any]) -> dict:
    """Carry existing per-arm instrumentation into the comparison without a new gate."""
    path = Path(run["out_dir"]) / "metadata" / "gen_gen_metrics.json"
    if not path.exists():
        return {"status": "unavailable", "diagnostic_only": True,
                "reason": "This arm has no retained generation-metrics artifact"}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "available", "diagnostic_only": True,
            "source": "metadata/gen_gen_metrics.json", "by_slice": metrics.get("by_slice", {})}


def _score_slices(cfg, arm: str, judgment: dict, raw: Path) -> dict:
    """Apply the historical controlled scorer to each slice, without mixing categories."""
    primary = str(cfg.thresholds.relative.primary_slice)
    questions = arena_hard_judge._expected_questions(cfg, judgment["limits"])
    records = arena_hard_judge._complete_judgments(read_jsonl(raw), questions)
    battles = battles_from_judgments(records)
    metadata = {}
    for key in (arm, str(cfg.baseline_arm)):
        answers = read_jsonl(bench_answers_dir(cfg) / f"{key}.jsonl")
        metadata[key] = {r["uid"]: r["metadata"] for r in answers}
    if set(cfg.statistics.control_features) != {"length", "markdown"}:
        raise ValueError("Arena-Hard scorer supports control_features=[length, markdown] only")
    slices = {}
    for category in sorted({b["category"] for b in battles}):
        block = evaluate_arm(
            [b for b in battles if b["category"] == category],
            metadata[arm], metadata[str(cfg.baseline_arm)],
            threshold=float(cfg.thresholds.relative.win_rate_ci_lower_min),
            rounds=int(cfg.statistics.bootstrap_rounds),
            alpha=float(cfg.statistics.alpha), seed=int(cfg.seed),
        )
        block["role"] = "primary" if category == primary else "secondary"
        block["gate_applies"] = category == primary and not bool(cfg.get("smoke", False))
        if not block["gate_applies"]:
            block["passes"] = None
        slices[category] = block
    if primary not in slices:
        raise ValueError(f"Arena-Hard comparison is missing primary slice {primary}")
    return slices


def pool(runs: list[dict[str, Any]], cfg, out_dir: Path) -> dict[str, Any]:
    """Judge every arm against the shared baseline and rank them.

    Args:
        runs: One dict per published arm, `{"target", "model_key", "mode", "out_dir",
            "repo"}` (run_eval builds these).
        cfg: The eval config.
        out_dir: The comparison's run directory.

    Returns:
        The comparison summary, including `model_key` (`vs_<baseline>` — the subject
        run_eval names the repo after) and `pooled_from` (every arm and the repo it was
        published to).
    """
    metas = {run["model_key"]: _arm_meta(run) for run in runs}
    references = [key for key, meta in metas.items() if meta.get("reference_arm")]
    assert len(references) == 1, (
        f"a comparison has exactly one baseline; these arms name {len(references)} "
        f"({references or 'none'}). run_eval runs --reference first as an arm, and that "
        "arm marks itself in its own metadata — a miss here means the orchestration was "
        "bypassed.")
    baseline = references[0]
    arms = [run for run in runs if run["model_key"] != baseline]
    assert arms, (
        "nothing to compare: this invocation ran only the reference arm. Pass at least "
        "one --target alongside --reference.")

    modes = {run["mode"] for run in runs}
    assert len(modes) == 1, (
        f"these arms ran in different thinking modes ({sorted(modes)}) — comparison code "
        "refuses cross-mode pairing (CLAUDE.md), and this is a comparison.")

    cfg = OmegaConf.merge(cfg)  # private copy
    source_vendor = str(cfg.vendor_dir)
    isolate_harness(cfg, out_dir)
    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)

    # Every arm's answers, back in the vendor tree the harness reads them from. They are
    # already published, so this is a copy of local files, never a regeneration.
    bench = bench_answers_dir(cfg)
    bench.mkdir(parents=True, exist_ok=True)
    declared = OmegaConf.to_container(cfg.arms, resolve=True)
    for run in runs:
        key = run["model_key"]
        shutil.copy2(Path(run["out_dir"]) / "rollouts" / "answers.jsonl",
                     bench / f"{key}.jsonl")
        declared = register(declared, key, run["target"],
                            "baseline" if key == baseline else "target", cfg)
    cfg.arms = declared
    if cfg.get("smoke", False):
        for arm in cfg.arms:
            arm.n_hard_prompt = min(4, int(arm.n_hard_prompt))
            arm.n_creative_writing = min(4, int(arm.n_creative_writing))
        cfg.judge_validation.n_questions = min(4, int(cfg.judge_validation.n_questions))
    cfg.baseline_arm = baseline
    cfg.output_dir = str(out_dir)
    generation_protocol = arena_hard_runner.validate_generation_protocols(runs, cfg)
    (metadata_dir / "generation_protocol_validation.json").write_text(
        json.dumps(generation_protocol, indent=2), encoding="utf-8")
    cfg_path = metadata_dir / "arena_hard_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    judge_model = str(cfg.judge.model)
    validation_policy = str(cfg.judge_validation.get("policy", "gate"))
    if validation_policy not in {"gate", "diagnostic"}:
        raise ValueError(f"Unknown Arena-Hard judge validation policy: {validation_policy}")
    validation = None
    if bool(cfg.judge_validation.get("enabled", False)):
        if judge_model == str(cfg.judge_validation.reference_judge):
            raise ValueError("Arena-Hard primary and auxiliary judges must be different")
        if (validation_policy == "gate" and not bool(cfg.get("smoke", False))
                and int(cfg.judge_validation.n_questions) != 100):
            raise ValueError("Full Arena-Hard judge validation requires 100 questions; use smoke for wiring checks")
        requested = cfg.judge_validation.get("comparison_arm")
        comparison = str(requested) if requested else arms[0]["model_key"]
        if comparison not in {a["model_key"] for a in arms}:
            raise ValueError(f"Judge validation arm is absent from this comparison: {comparison}")
        cfg.judge_validation.comparison_arm = comparison
        OmegaConf.save(cfg, cfg_path)
        try:
            validation = _validation_report(cfg, arena_hard_judge.validate_judge(cfg))
        except arena_hard_judge.JudgeValidationError as exc:
            validation = _validation_report(cfg, exc.report)
            (results_dir / "judge_validation.json").write_text(json.dumps(validation, indent=2))
            raise
        finally:
            raw_validation = (Path(str(cfg.vendor_dir)) / "data" / str(cfg.bench_name)
                              / "model_judgment")
            if raw_validation.exists():
                shutil.copytree(raw_validation, rollouts_dir / "judge_validation", dirs_exist_ok=True)
        (results_dir / "judge_validation.json").write_text(json.dumps(validation, indent=2))
        _require_validation(cfg, validation)
    table = []
    for run in arms:
        key = run["model_key"]
        arena_hard_judge.main(config=str(cfg_path), mode="judge", arm=key)
        judge_dir = max((out_dir / "judging").glob("*/"), key=lambda p: p.name)
        judgment = json.loads((judge_dir / f"judgment_{key}.json").read_text())
        # The judge is a model and its verdicts are its rollouts (CLAUDE.md: "logs" means
        # ROLLOUTS), so the raw per-battle records travel with the comparison.
        raw = (Path(str(cfg.vendor_dir)) / "data" / str(cfg.bench_name)
               / "model_judgment" / judge_model / f"{key}.jsonl")
        if not raw.exists():
            raise ValueError(f"Arena-Hard cannot score without raw judgments: {raw}")
        shutil.copy2(raw, rollouts_dir / f"judgments_{key}.jsonl")
        slices = _score_slices(cfg, key, judgment, raw)
        judgment["statistics_by_slice"] = slices
        (results_dir / f"judgment_{key}.json").write_text(json.dumps(judgment, indent=2))
        primary_slice = str(cfg.thresholds.relative.primary_slice)
        primary = slices[primary_slice]
        table.append({"model_key": key, "target": run["target"],
                      "repo": run.get("repo", ""), "primary_slice": primary_slice,
                      "win_rate": primary["controlled"]["mean"],
                      "ci_lower": primary["controlled"]["ci_lower"],
                      "ci_upper": primary["controlled"]["ci_upper"],
                      "n_prompts": primary["controlled"]["n"],
                      "passes": primary["passes"], "by_slice": slices})
    shutil.rmtree(out_dir / "judging", ignore_errors=True)
    shutil.rmtree(Path(str(cfg.vendor_dir)))
    cfg.vendor_dir = source_vendor
    OmegaConf.save(cfg, cfg_path)

    table.sort(key=lambda row: row.get("win_rate", 0.0), reverse=True)
    summary = {
        "report_version": 3,
        "metric": "style_controlled_win_rate",
        "primary_slice": str(cfg.thresholds.relative.primary_slice),
        "model_key": f"vs_{baseline}",
        "mode": modes.pop(),
        "baseline": baseline,
        "judge": judge_model,
        "judge_policy": {
            "primary_judge": judge_model,
            "primary_completion_required": True,
            "auxiliary_judge": str(cfg.judge_validation.reference_judge),
            "auxiliary_enabled": bool(cfg.judge_validation.get("enabled", False)),
            "auxiliary_policy": validation_policy,
        },
        "judge_validation": validation,
        "generation_protocol_validation": generation_protocol,
        "generation_health": {run["model_key"]: _generation_health(run) for run in runs},
        "smoke": bool(cfg.get("smoke", False)),
        "n_arms": len(table),
        "leaderboard": table,
        # Pointers, not copies: every arm's answers already have a home, and duplicating
        # them here would make two artifacts that could come to disagree.
        "pooled_from": [
            {"target": run["target"], "model_key": run["model_key"],
             "repo": run.get("repo", ""),
             "reference_arm": run["model_key"] == baseline,
             "answers": metas[run["model_key"]].get("answers", {})}
            for run in runs
        ],
    }
    (metadata_dir / "sources.json").write_text(json.dumps(summary["pooled_from"], indent=2))
    (results_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2))
    return summary
