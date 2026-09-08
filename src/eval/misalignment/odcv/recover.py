# ABOUTME: Recover an ODCV rollout transcript from a scenario's docker_output.log when the executor
# ABOUTME: was killed before it wrote messages_record.txt (the "ok+no_transcript" failure).
from __future__ import annotations

import json
import re
from pathlib import Path

# A runaway <think> loop can produce a multi-megabyte executor log; the reassembled transcript
# is capped so it stays within a judge's context window.
DEFAULT_CAP = 120_000
_EXECUTOR_PREFIX = "executor-1"
_EVENT = re.compile(r"\[(Reasoning|Reply|Tool|out)\]:\s?(.*)")


def reconstruct_transcript(
    docker_log: Path,
    variant: str,
    scenario: str,
    bench_dir: Path,
    cap: int = DEFAULT_CAP,
) -> str | None:
    """Rebuild a `messages_record.txt`-format transcript from an executor's stdout log.

    When an ODCV scenario's executor container is killed before it writes
    `messages_record.txt` (the `ok+no_transcript` failure — the model overruns the context
    window in a `<think>` loop, or the container is OOM-killed), the model's actions still
    survive in `docker_output.log` as the executor's own event log. This parses those events
    back into the exact text layout `odcv_judge` globs for, so a recovered cell judges
    identically to a completed one.

    The executor emits one line per event, tagged `[Reasoning]`, `[Reply]`, `[Tool]` or
    `[out]`; a Reasoning/Reply/Tool group is one assistant turn and `[out]` is its tool
    result. The real system/user prompt is read from the bench's per-scenario `prompt.json`.

    Args:
        docker_log: The scenario's `docker_output.log`.
        variant: ``"mandated"`` or ``"incentivized"`` — selects the bench prompt directory.
        scenario: Scenario name (the ``experiments/<Scenario>`` directory name).
        bench_dir: Root of the vendored odcv-bench checkout.
        cap: Character budget for the reassembled transcript; a runaway-loop log is truncated
            to this so the judge's context window is not exceeded.

    Returns:
        The transcript text, or ``None`` if the prompt is missing or no assistant turn was
        logged (system + user only).
    """
    prompt_path = bench_dir / f"{variant}_scenarios" / scenario / "prompt.json"
    if not prompt_path.is_file():
        return None
    prompt = json.loads(prompt_path.read_text())

    events: list[str] = []
    for line in docker_log.read_text(errors="replace").splitlines():
        line = line.rstrip("\n")
        if line.lstrip().startswith(_EXECUTOR_PREFIX):
            events.append(line.split("|", 1)[1].lstrip() if "|" in line else line)

    steps: list[tuple[str, dict]] = [
        ("system", {"content": prompt.get("system_prompt", "")}),
        ("user", {"content": prompt.get("user_prompt", "")}),
    ]
    current: dict | None = None
    field: str | None = None

    def flush() -> None:
        nonlocal current
        if current:
            steps.append(("assistant", current))
            current = None

    for line in events:
        match = _EVENT.match(line)
        if match:
            tag, value = match.group(1), match.group(2)
            if tag == "Reasoning":
                flush()
                current = {"reason": value, "content": "None", "calls": []}
                field = "reason"
            elif tag == "Reply":
                current = current or {"reason": "", "content": value, "calls": []}
                current["content"] = value
                field = "content"
            elif tag == "Tool":
                current = current or {"reason": "", "content": "None", "calls": []}
                current["calls"].append(value)
                field = "call"
            elif tag == "out":
                flush()
                steps.append(("tool", {"content": value}))
                field = "out"
        elif current is not None and field in ("reason", "content") and not line.startswith("==="):
            # A wrapped continuation of the current reasoning/reply text.
            current[field] = (
                f"{current[field]}\n{line}" if current[field] not in ("", "None") else line
            )
    flush()

    if len(steps) <= 2:  # system + user only: nothing the model did survived
        return None

    lines: list[str] = []
    for i, (role, body) in enumerate(steps, 1):
        lines.append(f"== Step {i} ==\nrole: {role}")
        lines.append(f"content: {body.get('content', 'None')}")
        if role == "assistant":
            if body.get("reason"):
                lines.append(f"reason: {body['reason']}")
            for call in body.get("calls", []):
                lines.append(f"call: {call}")
        lines.append("")
    text = "\n".join(lines)
    if len(text) > cap:
        text = text[:cap] + "\n\n== [transcript truncated: container killed before completion] ==\n"
    return text
