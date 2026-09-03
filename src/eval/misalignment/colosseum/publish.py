# ABOUTME: The online half: judge the finished episodes and push each run dir to HF.
# ABOUTME: Runs from a Killarney LOGIN node, because compute nodes have no network.

"""Finish a run that a GPU node could only half-finish.

Killarney compute nodes have no outbound network, so `uv run evals` runs there with
`--no-push` and the judge does not run at all. Everything a network needs is deferred to
here and driven over the same run directories, which are already in the published layout
(`rollouts/ results/ metadata/`) because run_eval's epilogue homed them before the push it
skipped.

Splitting the run this way is not just a workaround for one cluster: the judge is a
post-hoc read of `blackboards.json` and the push is a file upload, so neither belongs
inside a GPU allocation that costs an H100 per hour. A dead pod loses no judgement here
because none had been made yet, and re-running is idempotent — the judge overwrites its
own output and the push overwrites the repo.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.eval.misalignment.colosseum.judge import judge_run_root
from src.huggingface import push_run_dir
from src.utils import git_sha, hub_name

__all__ = ["finish_run_dir", "find_run_dirs"]


def find_run_dirs(root: Path) -> list[Path]:
    """Every per-arm Colosseum run directory under `output/colosseum_jira/`.

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


def _card(meta: dict, cfg, summary: dict) -> dict:
    """The dataset card, rebuilt from the metadata the GPU node left behind.

    Rebuilt rather than re-derived: the values that matter — the target, the mode, the
    exact command, the commit — are facts about the run that already happened, and
    recomputing them here would silently describe THIS machine instead.
    """
    target = meta.get("target", "")
    return {
        "experiment": f"colosseum_jira {summary.get('experiment', '')} of {target} "
        f"(mode={meta.get('mode')}), six-agent Jira team; "
        f"peer={summary.get('peer', '')}",
        "date_generated": date.today().isoformat(),
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
    run_dir: Path, cfg, *, judge: bool = True, push: bool = True, judge_workers: int = 8
) -> dict:
    """Judge one arm's episodes and push the run dir to the Hub.

    Args:
        run_dir: A per-arm directory under `output/colosseum_jira/`.
        cfg: The eval config (for the `judge:` block and the card).
        judge: Run the judge pass. Off for a re-push that only needs the upload.
        push: Upload to HF. Off for a judge-only pass.
        judge_workers: Concurrent judge calls.

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
        verdicts = judge_run_root(root, cfg, max_workers=judge_workers)
        (run_dir / "results" / "judge.json").write_text(json.dumps(verdicts, indent=2))
        # The canonical summary gains the secondary measure, so results.json stays the
        # one file that describes the run.
        summary["judge"] = {k: v for k, v in verdicts.items() if k != "per_run"}
        (run_dir / "results" / "results.json").write_text(json.dumps(summary, indent=2))
        out["judge"] = summary["judge"]

    if push:
        model_key = meta.get("target", "").split("/")[-1] or run_dir.name
        repo_id = hub_name(
            f"colosseum_jira {summary.get('experiment', '')} {model_key}"
        )
        out["repo"] = push_run_dir(
            run_dir,
            repo_id,
            _card(meta, cfg, summary),
            front_matter={
                "tags": [
                    "eval-run",
                    "eval:colosseum_jira",
                    f"model:{model_key}",
                    f"mode:{meta.get('mode')}",
                    f"experiment:{summary.get('experiment', '')}",
                ]
            },
        )
        print(f">>> pushed {out['repo']}")
    return out
