# ABOUTME: From Inspect's logs to this repo's layout: one self-contained rollout per episode, a rows
# ABOUTME: table of every measure, and the summary (cell rates, ladder, hit vs false-alarm calibration).

"""What leaves the Inspect run.

Inspect writes its own `.eval` logs (kept under metadata/inspect/, the full record). This
module reads them back into the contract every eval here publishes: `rollouts/` holds one
markdown transcript per episode — prompts, the target's conversation with its reasoning
and tool calls, the board, the peer's conversation, the scores — `results/` holds
`episodes.jsonl` (one row per episode with every measure) and the summary run_eval writes
as results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.log import EvalLog
from inspect_ai.scorer import CORRECT

from src.eval.misalignment.whistlebench_team import measures as M
from src.utils import transcript_markdown

__all__ = ["export", "rows_from_logs", "rollout_markdown"]

_STORE_PREFIX = "TeamBoard:"


def _store(sample, key: str, default=None):
    return sample.store.get(f"{_STORE_PREFIX}{key}", default)


def _ci(value) -> bool | None:
    if value is None:
        return None
    return value == CORRECT


def _dict_value(score) -> dict:
    if score is None or not isinstance(score.value, dict):
        return {}
    return score.value


def _row(log: EvalLog, sample) -> dict:
    meta = sample.metadata or {}
    scores = sample.scores or {}
    beh = scores.get("behaviour")
    beh_meta = (beh.metadata or {}) if beh is not None else {}
    measured = dict(beh_meta.get("measures") or {})
    wb = _dict_value(scores.get("whistleblowing"))
    aware = _dict_value(
        scores.get("test_awareness") or scores.get("deliberation_analysis")
    )
    refusal = scores.get("refusal")
    row = {
        "storyline": meta.get("storyline"),
        "docs": meta.get("docs"),
        "team": meta.get("team"),
        "seed": int(sample.epoch) - 1,
        "task": log.eval.task,
        "sample_id": sample.id,
        "errored": sample.error is not None,
        "error": (sample.error.message if sample.error is not None else None),
        "limit": sample.limit.type if sample.limit is not None else None,
        "n_messages": len(sample.messages),
        "peer_calls": _store(sample, "peer_calls"),
        "peer_stop": _store(sample, "peer_stop"),
        "n_board_posts": len(_store(sample, "posts", []) or []),
        # UPSTREAM scorers, as the paper reports them
        "gov_whistleblowing": _ci(wb.get("gov_whistleblowing")),
        "media_whistleblowing": _ci(wb.get("media_whistleblowing")),
        "internal_whistleblowing": _ci(wb.get("internal_whistleblowing")),
        "refusal_upstream": _ci(refusal.value)
        if refusal is not None and not isinstance(refusal.value, dict)
        else None,
        "evaluation_awareness": _ci(aware.get("evaluation_awareness")),
        "hypothetical_recognition": _ci(aware.get("hypothetical_recognition")),
        # this eval's measures (behaviour scorer metadata)
        **measured,
        "judge_error": (beh_meta.get("verdict") or {}).get("judge_error"),
    }
    row["upstream_any"] = (
        None
        if row["gov_whistleblowing"] is None
        else bool(
            row["gov_whistleblowing"]
            or row["media_whistleblowing"]
            or row["internal_whistleblowing"]
        )
    )
    return row


def rows_from_logs(logs: list[EvalLog]) -> list[dict]:
    rows = []
    for log in logs:
        for sample in log.samples or []:
            rows.append(_row(log, sample))
    return rows


def _messages_json(messages) -> list[dict]:
    out = []
    for m in messages:
        d = {"role": m.role}
        if isinstance(m.content, str):
            d["content"] = m.content
        else:
            parts = []
            for c in m.content:
                if c.type == "reasoning":
                    parts.append({"type": "reasoning", "reasoning": c.reasoning})
                elif c.type == "text":
                    parts.append({"type": "text", "text": c.text})
                else:
                    parts.append({"type": c.type})
            d["content"] = parts
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [
                {"id": tc.id, "function": tc.function, "arguments": tc.arguments}
                for tc in m.tool_calls
            ]
        if m.role == "tool":
            d["tool_call_id"] = getattr(m, "tool_call_id", None)
            d["function"] = getattr(m, "function", None)
        out.append(d)
    return out


def rollout_markdown(log: EvalLog, sample, row: dict) -> str:
    meta = sample.metadata or {}
    title = f"{meta.get('storyline')}/{meta.get('docs')}/{meta.get('team')} — seed {row['seed']}"
    rung = row.get("rung")
    intro = (
        f"Rung: **{rung}**. Upstream whistleblowing (any email or outside write): "
        f"**{row.get('upstream_whistleblowing')}**. Judged: {row.get('judged')}. "
        f"Stop: {'error' if row['errored'] else (row['limit'] or 'submitted' if row.get('submitted') else 'ended')}."
    )
    messages = list(sample.messages)
    system = next((m for m in messages if m.role == "system"), None)
    users = [
        m
        for m in messages
        if m.role == "user" and not m.text.startswith("[Team board]")
    ][:2]
    posts = _store(sample, "posts", []) or []
    board = "\n\n".join(f"[{p['sender']}] {p['text']}" for p in posts) or "(no posts)"
    peer_messages = _store(sample, "peer_messages", []) or []
    scores = {
        name: {"value": s.value, "explanation": s.explanation}
        for name, s in (sample.scores or {}).items()
    }
    beh = (sample.scores or {}).get("behaviour")
    sections = [
        (
            2,
            "Cell",
            "json",
            json.dumps(
                {
                    **meta,
                    "seed": row["seed"],
                    "task": log.eval.task,
                    "model": log.eval.model,
                },
                indent=2,
            ),
        ),
        (
            2,
            "System prompt (target)",
            "fenced",
            system.text if system is not None else "(none recorded)",
        ),
        (
            2,
            "Task (the requester's message and the documents)",
            "fenced",
            "\n\n---\n\n".join(m.text for m in users),
        ),
        (2, "Team board", "fenced", board),
        (
            2,
            "Target conversation (every message, reasoning included)",
            "json",
            json.dumps(_messages_json(messages), indent=2, ensure_ascii=False),
        ),
        (
            2,
            "Peer conversation",
            "json",
            json.dumps(peer_messages, indent=2, ensure_ascii=False)
            if peer_messages
            else "[]",
        ),
        (
            2,
            "Scores",
            "json",
            json.dumps(scores, indent=2, ensure_ascii=False, default=str),
        ),
        (
            2,
            "Behaviour judge",
            "json",
            json.dumps(
                beh.metadata if beh is not None else {},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
        ),
        (
            2,
            "Measures",
            "json",
            json.dumps(row, indent=2, ensure_ascii=False, default=str),
        ),
    ]
    if sample.error is not None:
        sections.insert(1, (2, "Error", "fenced", str(sample.error.message)))
    return transcript_markdown(title, intro, sections)


def export(
    logs: list[EvalLog],
    rollout_dir: Path,
    results_dir: Path,
    endorsed: list[str],
    overreach: list[str],
    run_info: dict,
) -> dict:
    """Write rollouts + episodes.jsonl and return the summary run_eval publishes."""
    rows = []
    for log in logs:
        for sample in log.samples or []:
            row = _row(log, sample)
            rows.append(row)
            cell_dir = (
                rollout_dir
                / str(row["storyline"])
                / str(row["docs"])
                / str(row["team"])
            )
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / f"seed{row['seed']}.md").write_text(
                rollout_markdown(log, sample, row), encoding="utf-8"
            )
    with (results_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = M.aggregate(rows, endorsed, overreach)
    summary["health"] = {
        "episodes": len(rows),
        "errored": sum(1 for r in rows if r["errored"]),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
        "hit_limit": sum(1 for r in rows if r.get("limit")),
        "not_submitted": sum(
            1 for r in rows if not r.get("submitted") and not r["errored"]
        ),
        "inspect_status": {log.eval.task: log.status for log in logs},
    }
    summary["run"] = run_info
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
