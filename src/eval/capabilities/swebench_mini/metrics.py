# ABOUTME: pass@1 with a Wilson interval, plus the instrumentation that distinguishes "solved
# ABOUTME: nothing" from "never produced a valid action" — the two look identical in the score.

"""Scoring the baseline, and the counters that stop a score being misread.

A single resolved rate is not enough to act on. A fine-tune that has lost tool-call
formatting, one that runs out of context on every long trajectory, and one that genuinely
cannot fix bugs all report the same number, and they demand completely different responses.
So every rollout-side counter here is reported next to pass@1, in the same spirit as the MMLU
eval reporting parse and truncation rates beside accuracy (CLAUDE.md gotcha 5).

The trajectory schema is upstream's and is read defensively: fields we cannot find come back
`None` rather than 0, because a silent zero reads as "no problems" and is exactly the wrong
message. Whichever fields survive the first smoke run are the ones the report should quote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.capabilities.stats import wilson_ci


def load_preds(preds_path: Path) -> dict[str, dict]:
    """Read mini-SWE-agent's preds.json (a dict keyed by instance id)."""
    if not preds_path.exists():
        return {}
    return json.loads(preds_path.read_text(encoding="utf-8"))


def patch_stats(preds: dict[str, dict], selected_ids: list[str]) -> dict[str, Any]:
    """Did the agent even produce a candidate patch for each selected instance?

    An instance missing from preds.json never finished; one present with an empty
    `model_patch` finished without editing anything. Both score unresolved, and neither is
    evidence about the model's ability to fix bugs.
    """
    attempted = [i for i in selected_ids if i in preds]
    non_empty = [i for i in attempted if (preds[i].get("model_patch") or "").strip()]
    return {
        "n_selected": len(selected_ids),
        "n_attempted": len(attempted),
        "n_with_patch": len(non_empty),
        "patch_rate": len(non_empty) / len(selected_ids) if selected_ids else 0.0,
        "n_missing": len(selected_ids) - len(attempted),
        "missing_instance_ids": sorted(set(selected_ids) - set(attempted)),
    }


def exit_status_counts(rollouts_dir: Path) -> dict[str, int]:
    """Aggregate mini-SWE-agent's per-batch exit-status files.

    These name the failure — context window exceeded, step limit, an API abort — which is the
    difference between "the model gave up" and "our serving config was too small".
    """
    import yaml

    counts: dict[str, int] = {}
    for path in sorted(rollouts_dir.glob("exit_statuses_*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for status in (data.values() if isinstance(data, dict) else []):
            key = str(status)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _messages(traj: Any) -> list[dict]:
    """Pull the message list out of a trajectory file, tolerating upstream's shape."""
    if isinstance(traj, dict):
        for key in ("messages", "trajectory", "history"):
            value = traj.get(key)
            if isinstance(value, list):
                return [m for m in value if isinstance(m, dict)]
    return traj if isinstance(traj, list) else []


def trajectory_stats(rollouts_dir: Path) -> dict[str, Any]:
    """Per-trajectory instrumentation: steps taken, and whether actions/reasoning appeared.

    `None` means "this file did not carry the field", never "zero occurrences" — see the
    module docstring. Confirm which fields are real after the first smoke run before quoting
    any of them in a report.
    """
    files = sorted(rollouts_dir.rglob("*.traj.json"))
    if not files:
        return {"n_trajectories": 0, "mean_steps": None, "no_tool_call_rate": None,
                "empty_reasoning_rate": None}

    steps: list[int] = []
    assistant_turns = no_tool_call = with_reasoning = empty_reasoning = 0
    saw_tool_field = saw_reasoning_field = False
    for path in files:
        try:
            messages = _messages(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        turns = [m for m in messages if m.get("role") == "assistant"]
        steps.append(len(turns))
        for message in turns:
            assistant_turns += 1
            if "tool_calls" in message:
                saw_tool_field = True
                if not message.get("tool_calls"):
                    no_tool_call += 1
            for field in ("reasoning_content", "reasoning"):
                if field in message:
                    saw_reasoning_field = True
                    if (message.get(field) or "").strip():
                        with_reasoning += 1
                    else:
                        empty_reasoning += 1
                    break
    reasoned = with_reasoning + empty_reasoning
    return {
        "n_trajectories": len(files),
        "mean_steps": sum(steps) / len(steps) if steps else None,
        "max_steps": max(steps) if steps else None,
        "assistant_turns": assistant_turns,
        # A fine-tune that lost tool-call formatting shows up here FIRST, as a wave of turns
        # that carry no action, long before it shows up as a low resolved rate.
        "no_tool_call_rate": (no_tool_call / assistant_turns
                              if saw_tool_field and assistant_turns else None),
        # CLAUDE.md gotcha 2: an arm that stopped reasoning has collapsed, and that is
        # invisible in pass@1.
        "empty_reasoning_rate": (empty_reasoning / reasoned
                                 if saw_reasoning_field and reasoned else None),
    }


def rollout_summary(preds: dict[str, dict], selected_ids: list[str],
                    rollouts_dir: Path) -> dict[str, Any]:
    """Everything knowable before grading — the half that needs the GPU."""
    return {**patch_stats(preds, selected_ids),
            "exit_statuses": exit_status_counts(rollouts_dir),
            **trajectory_stats(rollouts_dir)}


def resolution_summary(report: dict, selected_ids: list[str]) -> dict[str, Any]:
    """pass@1 from the official harness's report, scored over the SELECTED instances.

    Scoring over the selection rather than over what the harness happened to evaluate is
    deliberate: an instance whose rollout crashed, or whose patch failed to apply, is an
    unresolved instance. Dropping it from the denominator would quietly inflate pass@1 by
    exactly the failures we most want to see.

    Args:
        report: The harness's `*.json` run report.
        selected_ids: The subset the run was defined over.

    Returns:
        `pass_at_1` plus its Wilson interval, and the harness's own tallies for cross-check.
    """
    resolved = set(report.get("resolved_ids") or [])
    unresolved = set(report.get("unresolved_ids") or [])
    selected = set(selected_ids)
    # Loud rather than silent: a resolved id outside the selection means the filter and the
    # grading input disagree, which invalidates the number rather than shrinking it.
    stray = sorted((resolved | unresolved) - selected)
    n_resolved = len(resolved & selected)
    interval = wilson_ci(n_resolved, len(selected))
    return {
        "pass_at_1": interval["mean"],
        "pass_at_1_ci_lower": interval["ci_lower"],
        "pass_at_1_ci_upper": interval["ci_upper"],
        "n_resolved": n_resolved,
        "n_scored": len(selected),
        "resolved_ids": sorted(resolved & selected),
        "harness_completed": report.get("completed_instances"),
        "harness_unresolved": report.get("unresolved_instances"),
        "harness_error": report.get("error_instances"),
        "instances_outside_selection": stray,
    }


def report_line(model: str, provenance: dict, selection: dict, scores: dict) -> str:
    """The one-line result string, in the agreed reporting format."""
    return (f"{model} + mini-SWE-agent {provenance['scaffold_version']} "
            f"(config {provenance['scaffold_config_sha256'][:12]}), "
            f"{selection['dataset']}@{selection['dataset_revision'][:12]} "
            f"[{selection['n_selected']}/{selection['split_size']} instances, "
            f"subset {selection['subset_hash']}], "
            f"pass@1 = {scores['pass_at_1']:.1%} "
            f"[{scores['pass_at_1_ci_lower']:.1%}, {scores['pass_at_1_ci_upper']:.1%}]")
