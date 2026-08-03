# ABOUTME: Scores ODCV-Bench trajectories with the benchmark's four LLM judges and
# ABOUTME: reports misalignment rate / severity against the published Qwen3.6-27B result.

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.openrouter import map_threaded  # noqa: E402
from src.eval.misalignment.odcv import VARIANTS, load_published_medians, median_score, summarise  # noqa: E402
from src.eval.misalignment.odcv_rollout import openrouter_usage  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402

load_dotenv()


def _load_upstream_judge(bench_dir: Path):
    """Import the vendored evaluate_all_results module.

    Reusing it verbatim keeps the rubric prompt, retry loop and JSON parsing
    byte-identical to the published pipeline.

    Args:
        bench_dir: Vendored ODCV-Bench checkout.

    Returns:
        The imported module object.
    """
    path = bench_dir / "evaluate_all_results.py"
    assert path.is_file(), f"missing upstream judge script: {path}"
    spec = importlib.util.spec_from_file_location("odcv_upstream_judge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "evaluation_routine"), "upstream judge API changed"
    return mod


def _transcripts(rollout_dir: Path, model_key: str) -> list[tuple[str, str, Path]]:
    """Collect (variant, scenario, transcript_path) for a completed rollout."""
    out = []
    for variant in VARIANTS:
        root = rollout_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
        assert root.is_dir(), f"no rollouts for {variant}: {root}"
        for scen in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            rec = scen / "messages_record.txt"
            if rec.is_file() and rec.stat().st_size > 0:
                out.append((variant, scen.name, rec))
    assert out, f"no transcripts under {rollout_dir}"
    return out


def _score_with_judge(mod, judge_model: str, items, out_path: Path, max_workers: int) -> dict:
    """Score every transcript with one judge, caching to disk for resumability.

    Args:
        mod: The upstream judge module.
        judge_model: OpenRouter model id of the judge.
        items: List of (variant, scenario, path).
        out_path: JSON cache for this judge's scores.
        max_workers: Thread pool size.

    Returns:
        {"<variant>/<scenario>": {"score": ..., "reasoning": ...}}
    """
    cache = json.loads(out_path.read_text()) if out_path.is_file() else {}
    todo = [it for it in items if f"{it[0]}/{it[1]}" not in cache]
    print(f"  judge {judge_model}: {len(cache)} cached, {len(todo)} to score", flush=True)
    if not todo:
        return cache

    mod.OPENAI_MODEL = judge_model

    def run(i: int):
        _, _, path = todo[i]
        return mod.evaluation_routine(str(path))

    results = map_threaded(run, len(todo), max_workers=max_workers, desc=judge_model)
    for (variant, scenario, _), (score, reasoning) in zip(todo, results):
        cache[f"{variant}/{scenario}"] = {"score": score, "reasoning": reasoning}
        out_path.write_text(json.dumps(cache, indent=2))
    return cache


def main(
    rollout_dir: str,
    config: str = "configs/eval/odcv_bench_base_openrouter.yaml",
    max_workers: int = 8,
    smoke: bool = False,
) -> None:
    """Judge a rollout with the benchmark's 4 judges and compare to the paper.

    Args:
        rollout_dir: Output directory produced by odcv_rollout.py.
        config: Path to the OmegaConf YAML config.
        max_workers: Concurrent judge calls per judge model.
        smoke: Judge only one transcript per variant, with all judges, to verify wiring.
    """
    cfg = OmegaConf.load(config)
    bench_dir = Path(cfg.bench_dir).resolve()
    rollout = Path(rollout_dir).resolve()
    mod = _load_upstream_judge(bench_dir)

    items = _transcripts(rollout, cfg.model_key)
    if smoke:
        items = [next(it for it in items if it[0] == v) for v in VARIANTS]
    judges = dict(cfg.judges)

    # Upstream excludes a judge from scoring its own trajectories (median-of-3).
    self_judge = [k for k, v in judges.items() if k == cfg.model_key]
    for k in self_judge:
        judges.pop(k)
        print(f"[self-exclusion] dropping judge {k} for model {cfg.model_key}")

    eval_dir = rollout / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    usage_before = openrouter_usage()
    print("=" * 72)
    print(f"Judging {len(items)} transcripts with {len(judges)} judges "
          f"({len(items) * len(judges)} calls)")
    print(f"OpenRouter spend so far: ${usage_before:.4f}")
    print("=" * 72, flush=True)

    per_judge = {}
    for key, judge_model in judges.items():
        per_judge[key] = _score_with_judge(
            mod, judge_model, items, eval_dir / f"scores_{key}.json", max_workers
        )
        first = next(iter(per_judge[key].values()))
        print(f"  first score from {key}: {first['score']} | "
              f"{str(first['reasoning'])[:160]}...", flush=True)

    usage_after = openrouter_usage(settle_s=90)

    medians: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    dropped = []
    for variant, scenario, _ in items:
        key = f"{variant}/{scenario}"
        scores = [
            float(per_judge[j][key]["score"])
            for j in per_judge
            if per_judge[j].get(key, {}).get("score") not in (None, "N/A")
        ]
        if not scores:
            dropped.append(key)
            continue
        medians[variant][scenario] = median_score(scores)

    ours = summarise(medians)
    # Comparison baseline, in precedence order:
    #   `baseline_results` — a prior run's results.json (same "ours" schema), for comparing
    #     against a model we ran ourselves rather than a paper row.
    #   `published_key`    — a paper row other than this model's own, for a fine-tune that
    #     has no row of its own but was trained from a model that does.
    #   otherwise          — this model's own published row.
    baseline_results = cfg.get("baseline_results", None)
    if baseline_results:
        published = json.loads(Path(baseline_results).read_text())["ours"]
    else:
        published_key = str(cfg.get("published_key", cfg.model_key))
        published = summarise(load_published_medians(
            bench_dir / "existing_results/current/evaluations/judge_all/scores_final_median.csv",
            published_key,
        ))

    results = {
        "model": cfg.model,
        "model_key": cfg.model_key,
        "judges": dict(judges),
        "n_judged": len(items),
        "n_dropped_all_na": len(dropped),
        "dropped": dropped,
        "judging_cost_usd": round(usage_after - usage_before, 4),
        "ours": ours,
        "published": published,
        "delta_mr_pct": round(ours["overall"]["mr_pct"] - published["overall"]["mr_pct"], 1),
        "published_within_our_ci": (
            ours["overall"]["mr_ci95"][0]
            <= published["overall"]["mr_pct"]
            <= ours["overall"]["mr_ci95"][1]
        ),
        "per_scenario_medians": medians,
    }

    out = rollout / "results.json"
    out.write_text(json.dumps(results, indent=2))
    write_run_meta(
        eval_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"command": " ".join(sys.argv), "smoke": smoke, "timestamp": timestamp()},
    )

    print("\n" + "=" * 72)
    print(f"REPLICATION: {cfg.model_key}")
    print(f"  ours      MR = {ours['overall']['mr_pct']}%  "
          f"CI95 {ours['overall']['mr_ci95']}  Sev = {ours['overall']['mean_severity']}")
    print(f"  published MR = {published['overall']['mr_pct']}%  "
          f"Sev = {published['overall']['mean_severity']}")
    print(f"  delta = {results['delta_mr_pct']:+.1f} pp | "
          f"published inside our CI: {results['published_within_our_ci']}")
    for variant in VARIANTS:
        print(f"  {variant:<13} ours {ours[variant]['mr_pct']:>5}% / "
              f"published {published[variant]['mr_pct']:>5}%")
    if dropped:
        print(f"  WARNING: {len(dropped)} trajectories had no usable judge score: {dropped}")
    print(f"  judging cost ${results['judging_cost_usd']:.2f}")
    print(f">>> {out}")

