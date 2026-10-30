# ABOUTME: From Petri's .eval logs to this repo's layout: one self-contained rollout per sample with
# ABOUTME: the target's system prompt first, and the in-run gate that the preamble really led it.

"""What leaves the Inspect run.

Inspect's `.eval` logs stay under metadata/inspect/ as the full record; this module reads
them back into the published contract: `rollouts/<seed>.md` (or `pass<k>/<seed>.md` over
several passes) holds the seed the auditor was handed, the system prompt the target was
actually served, every message of the target's last call with its reply, and the judge's
scores. On a constitution arm the system prompt is checked to START with the composed
preamble -- the check that used to live in a separate verify script, now part of the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from src.eval.inspect_messages import message_text, tool_calls_json
from src.utils import transcript_markdown

from .scores import sample_concerning, target_events, valid_score

__all__ = ["rollout_markdown", "write_rollouts"]

_FRONT_MATTER = ("tenet", "principle", "family", "archetype")


def _message_body(message) -> str:
    body = message_text(message)
    calls = tool_calls_json(message)
    if calls:
        body = f"{body}\n\ntool_calls: {json.dumps(calls, ensure_ascii=False, default=str)}".lstrip("\n")
    return body


def _system_message(event) -> str:
    """The system message the target saw on this call, or "" when the auditor never set one."""
    return next((message_text(m) for m in event.input if m.role == "system"), "")


def rollout_markdown(sample, *, preamble_text: str | None) -> str:
    """One sample's self-contained transcript.

    Args:
        sample: An Inspect EvalSample from a Petri audit log.
        preamble_text: The composed system preamble this arm was run with, or None. When given
            and the target was called, the system message the target saw MUST start with it.
    """
    meta = sample.metadata or {}
    title = f"{sample.id} (pass {sample.epoch})"
    front = " | ".join(f"{k}: {meta[k]}" for k in _FRONT_MATTER if k in meta)
    # The same score results.json refuses is shown here with the reason, so a reader of a
    # dead-endpoint transcript never takes its `1.0` for a result.
    verdict = f"concerning = {sample_concerning(sample)}" + (
        "" if valid_score(sample) is not None else
        " (EXCLUDED from results: unscored, or the last target call errored)")
    intro = f"{front}\n\n{verdict}" if front else verdict
    sections: list[tuple[int, str, str, str]] = [(2, "Seed", "fenced", str(sample.input))]
    events = target_events(sample)
    if events:
        ev = events[-1]
        sys_msg = _system_message(ev)
        if preamble_text is not None and sys_msg:
            assert sys_msg.startswith(preamble_text), (
                f"{sample.id}: the constitution preamble is not first in the target's system message")
        sections.append((2, "Target system prompt", "fenced", sys_msg))
        sections.append((2, "Target transcript", "text", ""))
        messages = [*ev.input, ev.output.message]
        sections.extend((3, f"{m.role} {i}", "fenced", _message_body(m)) for i, m in enumerate(messages))
    else:
        sections.append((2, "Target transcript", "text", "the auditor never called the target"))
    judge = {name: {"value": sc.value, "explanation": sc.explanation}
             for name, sc in (sample.scores or {}).items()}
    sections.append((2, "Judge", "json", json.dumps(judge, indent=2, ensure_ascii=False, default=str)))
    if sample.error is not None:
        sections.append((2, "Error", "fenced", str(sample.error)))
    return transcript_markdown(title, intro, sections)


def write_rollouts(logs: list[Path], rollout_dir: Path, *, preamble_text: str | None, passes: int) -> dict:
    """One rollout file per sample over every log; `pass<k>/` subdirectories over several passes.

    Logs are read in the order given (name-sorted, oldest first), so a seed rerun in a later
    log overwrites its earlier rollout exactly as it overrides its score.

    Returns:
        `n_rollouts` written, `excluded_rollouts` (samples `valid_score` refuses) and
        `targets_without_system_prompt` (the target was called but the auditor never set a
        system message -- on a constitution arm, a sample the preamble gate could not check).
    """
    final: dict[Path, object] = {}
    for log in logs:
        for sample in read_eval_log(str(log), resolve_attachments=True).samples or []:
            dest = rollout_dir if passes == 1 else rollout_dir / f"pass{sample.epoch}"
            final[dest / f"{sample.id}.md"] = sample
    for dest, sample in final.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rollout_markdown(sample, preamble_text=preamble_text), encoding="utf-8")
    calls = [target_events(s) for s in final.values()]
    return {"n_rollouts": len(final),
            "excluded_rollouts": sum(valid_score(s) is None for s in final.values()),
            "targets_without_system_prompt": sum(bool(ev) and not _system_message(ev[-1]) for ev in calls)}
