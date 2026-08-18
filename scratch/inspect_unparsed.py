# ABOUTME: Show what the unparseable sycophancy replies actually contain, per arm, so the
# ABOUTME: parse failure is diagnosed from evidence. Run: uv run python scratch/inspect_unparsed.py T2

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from scratch.reparse_sycophancy import REPOS, parse_rollout
from src.eval.deliberation.sycophancy.scoring import parse_letter


def main(arm: str = "T2", show: int | str = 4) -> None:
    show = int(show)
    local = snapshot_download(REPOS[arm], repo_type="dataset",
                              allow_patterns=["rollouts/*.md"])
    shown = 0
    empty_first = empty_second = 0
    for path in sorted(Path(local, "rollouts").glob("*.md")):
        parsed = parse_rollout(path.read_text())
        if parsed is None:
            continue
        _correct, letters, reply1, reply2 = parsed
        first = parse_letter(reply1, letters, loose=True)
        second = parse_letter(reply2, letters, loose=True)
        if first and second:
            continue
        which, text = ("turn1", reply1) if not first else ("turn2", reply2)
        if not text.strip():
            (empty_first := empty_first + 1) if which == "turn1" else None
            (empty_second := empty_second + 1) if which == "turn2" else None
        if shown < show:
            shown += 1
            print(f"\n===== {path.stem} — {which} unparsed, letters={letters}, "
                  f"len={len(text)} =====")
            print(repr(text[:600]))
    print(f"\nempty-text unparsed: turn1={empty_first} turn2={empty_second}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["T2"]))
