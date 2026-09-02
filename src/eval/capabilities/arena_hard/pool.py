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
pools seed replicates, which share a style-type, so dropping the seed leaves a name that
still describes every member. Here `difficult_advice_0`, `courtroom_716_0` and
`tulu_100_0` have no common prefix at all, and stripping what differs would leave nothing.

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
from src.eval.capabilities.arena_hard.runner import bench_answers_dir, register
from src.eval.layout import publish_layout


def _arm_meta(run: dict[str, Any]) -> dict:
    """One published arm's `metadata/sources.json` — what it is and where it came from."""
    path = Path(run["out_dir"]) / "metadata" / "sources.json"
    assert path.exists(), (
        f"{run['target']}: no metadata/sources.json at {path}. Every arena_hard arm writes "
        "one; a run dir without it did not come from this eval's runner.")
    return json.loads(path.read_text())


def _overall(judgment: dict) -> dict[str, float]:
    """Win/tie/loss and mean score across every slice, weighted by prompts in each.

    A leaderboard needs one number per arm, and averaging the slice rates unweighted would
    let a 150-prompt slice count for as much as a 500-prompt one.
    """
    slices = (judgment.get("by_slice") or {}).values()
    total = sum(float(s.get("n_prompts", 0)) for s in slices)
    if not total:
        return {}
    return {
        key: round(sum(float(s.get(key, 0)) * float(s.get("n_prompts", 0))
                       for s in slices) / total, 4)
        for key in ("win_rate", "tie_rate", "loss_rate")
    }


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
    cfg.baseline_arm = baseline
    cfg.output_dir = str(out_dir)
    cfg_path = metadata_dir / "arena_hard_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    judge_model = str(cfg.judge.model)
    table = []
    for run in arms:
        key = run["model_key"]
        arena_hard_judge.main(config=str(cfg_path), mode="judge", arm=key)
        judge_dir = max((out_dir / "judging").glob("*/"), key=lambda p: p.name)
        judgment = json.loads((judge_dir / f"judgment_{key}.json").read_text())
        (judge_dir / f"judgment_{key}.json").rename(results_dir / f"judgment_{key}.json")
        # The judge is a model and its verdicts are its rollouts (CLAUDE.md: "logs" means
        # ROLLOUTS), so the raw per-battle records travel with the comparison.
        raw = (Path(str(cfg.vendor_dir)) / "data" / str(cfg.bench_name)
               / "model_judgment" / judge_model / f"{key}.jsonl")
        if raw.exists():
            shutil.copy2(raw, rollouts_dir / f"judgments_{key}.jsonl")
        table.append({"model_key": key, "target": run["target"],
                      "repo": run.get("repo", ""), **_overall(judgment)})
    shutil.rmtree(out_dir / "judging", ignore_errors=True)

    table.sort(key=lambda row: row.get("win_rate", 0.0), reverse=True)
    summary = {
        "model_key": f"vs_{baseline}",
        "mode": modes.pop(),
        "baseline": baseline,
        "judge": judge_model,
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
