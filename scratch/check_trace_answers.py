# ABOUTME: When the visible answer is empty, does the reasoning trace still end on a letter?
# ABOUTME: Run: uv run python scratch/check_trace_answers.py T2

from __future__ import annotations

import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from scratch.reparse_sycophancy import REPOS, parse_rollout
from src.eval.deliberation.sycophancy.scoring import parse_letter

_TRACE1 = re.compile(r"## Assistant reasoning \(turn 1\)\n(.*?)(?=\n## )", re.DOTALL)
_TRACE2 = re.compile(r"## Assistant reasoning \(turn 2\)\n(.*?)(?=\n## )", re.DOTALL)


def _fenced(block: str) -> str:
    """Strip the code fence transcript_markdown wraps verbatim model output in."""
    lines = block.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def main(arm: str = "T2", show: int | str = 2) -> None:
    show = int(show)
    local = snapshot_download(REPOS[arm], repo_type="dataset",
                              allow_patterns=["rollouts/*.md"])
    empty = recoverable = 0
    shown = 0
    for path in sorted(Path(local, "rollouts").glob("*.md")):
        text = path.read_text()
        parsed = parse_rollout(text)
        if parsed is None:
            continue
        _correct, letters, reply1, reply2 = parsed
        for reply, pattern, which in ((reply1, _TRACE1, "turn1"), (reply2, _TRACE2, "turn2")):
            if parse_letter(reply, letters, loose=True):
                continue
            empty += 1
            match = pattern.search(text)
            trace = _fenced(match.group(1)) if match else ""
            # The trace is chain-of-thought, so only its TAIL is a commitment; a letter
            # mentioned while enumerating options is not an answer.
            tail = trace[-400:]
            found = parse_letter(tail, letters, loose=True)
            recoverable += bool(found)
            if shown < show and trace:
                shown += 1
                print(f"\n===== {path.stem} {which}: trace {len(trace)} ch, "
                      f"tail-parse={found or 'none'} =====")
                print(repr(tail[-320:]))
    rate = recoverable / empty if empty else 0.0
    print(f"\n{arm}: {empty} unparsed turns, {recoverable} recoverable from the trace tail "
          f"({rate:.1%})")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["T2"]))
