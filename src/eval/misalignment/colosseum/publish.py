# ABOUTME: Finish a Colosseum run dir after the fact — judge its episodes and push it — under
# ABOUTME: exactly the name, card and tags run_eval's own epilogue would have given it.

"""The recovery path, not the normal one.

`uv run evals --name colosseum_hospital ...` is the whole eval: the runner judges its own
episodes (hospital/runner.py) and run_eval names, tags and pushes the arm. This module is
for a run dir that invocation left unfinished — `--no-push` on a host with no route to the
Hub (Killarney's compute nodes, the Jira eval's home, where the judge cannot run either), a
merged cell assembled from several pods' pieces (scratch/colosseum_hospital/merge_cells.py),
a run whose push died — and it is held to the same contract. The NAME is
`src.naming.eval_name` over the run's own `model_key` and the eval's registered name facets
(`src.eval.run_variant`, read off the config in `metadata/run_meta.json`); the tags are
`src.eval.layout.run_tags`; the org is `HF_ORG`. Nothing is typed, so a run finished here is
indistinguishable from one run_eval finished, and the two can never disagree about what a
run is called.

Re-running is idempotent: the judge overwrites its own output and the push overwrites the
repo.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from src.eval import EVALS, run_variant
from src.eval.layout import run_tags
from src.eval.misalignment.colosseum import judge as jira_judge
from src.eval.misalignment.colosseum.hospital import judge as hospital_judge
from src.infra.huggingface import hf_repo_id, push_run_dir
from src.naming import eval_name as _eval_name
from src.utils import git_sha

__all__ = ["JUDGES", "finish_run_dir", "find_run_dirs", "model_key_of", "repo_name_for"]


def _judge_jira(run_dir: Path, root: Path, cfg, summary: dict, workers: int) -> dict:
    return jira_judge.judge_arm(run_dir, root, cfg, max_workers=workers)


def _judge_hospital(
    run_dir: Path, root: Path, cfg, summary: dict, workers: int
) -> dict:
    return hospital_judge.judge_arm(
        run_dir,
        root,
        cfg,
        condition=str(summary["condition"]),
        pair=summary.get("pair"),
        max_workers=workers,
    )


# registry key -> the eval's own judge_arm, called `(run_dir, root, cfg, summary, workers)`.
# The Hospital's needs the cell and the watched seats from the arm's summary; the Jira's
# needs neither. Each is the SAME function the eval's runner calls during a live run.
JUDGES = {
    "colosseum_jira": _judge_jira,
    "colosseum_hospital": _judge_hospital,
}


def model_key_of(meta: dict) -> str:
    """The arm's name token, as run_eval resolved it for this run.

    Recorded in run_meta.json (`model_key`) since 2026-09-25. A run dir older than that
    resolves its target again — metadata only, the way run_eval's preflight does — so the
    key is the one the framework would use today, not a string cut off the repo id here.
    """
    key = meta.get("model_key")
    if key:
        return str(key)
    from src.infra.endpoints.vllm import resolve_target

    return resolve_target(str(meta["target"])).model_key


def repo_name_for(eval_name: str, meta: dict, *, produced: str | None = None) -> str:
    """The fully qualified `org/name` this run publishes to — run_eval's name, rebuilt.

    Args:
        eval_name: The registry key (`colosseum_jira`, `colosseum_hospital`).
        meta: The run's `metadata/run_meta.json`: `target`, `model_key` (or resolvable),
            and `config` — the RESOLVED config the run was launched with, which is where
            the eval's name facets (the condition, the experiment) are read from.
        produced: The day the episodes were run, for the date the name carries and the
            card's `date_generated`; default today, for a push on the day of the run.

    Qualified HERE, the way run_eval does it, because `push_run_dir` gates the NAME before
    it touches the network. The org comes from `HF_ORG` in the environment
    (src.infra.huggingface.hf_org) — never from a config.
    """
    variant = run_variant(EVALS[eval_name], OmegaConf.create(meta.get("config") or {}))
    return hf_repo_id(
        _eval_name(eval_name, model_key_of(meta), date=produced, variant=variant)
    )


def find_run_dirs(root: Path) -> list[Path]:
    """Every per-arm Colosseum run directory under `output/<eval>/`.

    A run directory is recognised by the metadata run_eval wrote, not by its name: the
    `pooled/` subtree has the same shape and must not be judged (it holds contrasts, not
    episodes).
    """
    return sorted(
        d
        for d in root.glob("*/")
        if (d / "metadata" / "run_meta.json").is_file()
        and (d / "rollouts" / "colosseum").is_dir()
    )


def _card(
    meta: dict,
    cfg,
    summary: dict,
    *,
    eval_name: str,
    variant: str,
    produced: str | None = None,
) -> dict:
    """The dataset card, rebuilt from the metadata the run left behind.

    Rebuilt rather than re-derived: the values that matter — the target, the mode, the
    exact command, the commit — are facts about the run that already happened, and
    recomputing them here would silently describe THIS machine instead.
    """
    from datetime import date

    target = meta.get("target", "")
    return {
        "experiment": f"{eval_name}{f' ({variant})' if variant else ''} eval of {target} "
        f"(mode={meta.get('mode')}), mixed-checkpoint team; "
        f"peer={summary.get('peer', '')}",
        "date_generated": produced or date.today().isoformat(),
        "constitution": str(cfg.get("constitution", "none")),
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": f"target={target} base={meta.get('base_model')} "
        f"judge={cfg.judge.model}",
        "generation_config": json.dumps(
            {
                "max_tokens": int(cfg.max_tokens),
                "temperature": float(cfg.temperature),
                "seeds": list(cfg.seeds),
            }
        ),
        "schema": "rollouts/: Colosseum episode trees (prompts, blackboards, "
        "trajectories, tool events); results/: per_seed.json, episodes.json, "
        "judge.json, results.json + .md; metadata/: run_meta.json + config",
        "provenance": meta.get("command", ""),
    }


def finish_run_dir(
    run_dir: Path,
    cfg,
    *,
    judge: bool = True,
    push: bool = True,
    judge_workers: int = 8,
    eval_name: str = "colosseum_jira",
    produced: str | None = None,
) -> dict:
    """Judge one arm's episodes and push the run dir to the Hub.

    Args:
        run_dir: A per-arm directory under `output/<eval_name>/`.
        cfg: The eval config (for the `judge:` block and the card).
        judge: Run the judge pass. Off for a re-push that only needs the upload, or for
            an arm `uv run evals` already judged.
        push: Upload to HF. Off for a judge-only pass.
        judge_workers: Concurrent judge calls.
        eval_name: The registry key the run dir belongs to; picks the judge and names
            the repo and the tags.
        produced: The day the episodes were run, for the repo name and the card's
            `date_generated` (default today).

    Returns:
        What was done: the judge summary (when run) and the repo URL (when pushed).
    """
    meta = json.loads((run_dir / "metadata" / "run_meta.json").read_text())
    summary = json.loads((run_dir / "results" / "results.json").read_text())
    out: dict = {"run_dir": str(run_dir), "target": meta.get("target", "")}

    if judge:
        # `colosseum_run_root` is recorded relative to the run dir by the runner, so the
        # tree stays findable after the directory is moved or copied off the cluster.
        root = run_dir / summary["colosseum_run_root"]
        assert root.is_dir(), f"{root} is missing; nothing to judge"
        summary["judge"] = JUDGES[eval_name](run_dir, root, cfg, summary, judge_workers)
        if (run_dir / "results" / "per_seed.json").is_file():
            # The judge's per-seed measures joined the harvest's; the canonical summary
            # mirrors that file, as the runner's does.
            summary["measures"] = json.loads(
                (run_dir / "results" / "per_seed.json").read_text()
            )
        (run_dir / "results" / "results.json").write_text(json.dumps(summary, indent=2))
        out["judge"] = summary["judge"]

    if push:
        variant = run_variant(
            EVALS[eval_name], OmegaConf.create(meta.get("config") or {})
        )
        model_key = model_key_of(meta)
        repo_id = repo_name_for(eval_name, meta, produced=produced)
        out["repo"] = push_run_dir(
            run_dir,
            repo_id,
            _card(
                meta,
                cfg,
                summary,
                eval_name=eval_name,
                variant=variant,
                produced=produced,
            ),
            front_matter={
                "tags": run_tags(
                    eval_name, model_key, str(meta.get("mode")), variant=variant
                )
            },
        )
        print(f">>> pushed {out['repo']}")
    return out
