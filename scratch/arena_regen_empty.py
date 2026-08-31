# ABOUTME: Strips the target's empty-answer prompts from BOTH arms' answer files and from
# ABOUTME: the judgments, so a rerun regenerates exactly those at a larger token budget.

"""Prepare an Arena-Hard rerun limited to the prompts that came back empty.

The generator's resume rule is `todo = [q for q in selected if q.uid not in existing]`, and
the vendored judge caches on uid, so "regenerate exactly these" is expressed by DELETING
their rows. Both arms lose the same rows: giving the target a larger budget on a prompt
where the baseline keeps its old 6,000-token answer would break decoding parity within that
prompt, and the baseline's answers there are long enough (median ~15k chars) that it may
have been truncated too — `finish_reason` is not persisted, so that cannot be ruled out.

    uv run python scratch/arena_regen_empty.py --apply
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import fire

A = Path("src/eval/capabilities/arena_hard/third_party/arena-hard-auto/data/arena-hard-v2.0")
TARGET = "qwen3_6-27b-lora-t2-9284-synthdoc-716-r64"
BASELINE = "qwen3_6-27b-lora-table2-only-9284-r64"
JUDGE = "google/gemini-3-flash-preview"


def _answer_text(rec: dict) -> str:
    """The user-visible answer from one answer record."""
    content = rec["messages"][-1]["content"]
    return (content.get("answer", "") if isinstance(content, dict) else content) or ""


def main(apply: bool = False, slice_name: str = "hard_prompt") -> None:
    """Report (and optionally apply) the row deletions the rerun needs.

    Args:
        apply: Actually rewrite the files. Without it, only reports.
        slice_name: Which slice's empty answers to target.
    """
    judgments = A / "model_judgment" / JUDGE / f"{TARGET}.jsonl"
    category = {}
    for line in judgments.open():
        rec = json.loads(line)
        category[rec["uid"]] = rec["category"]

    answers = {json.loads(l)["uid"]: json.loads(l)
               for l in (A / "model_answer" / f"{TARGET}.jsonl").open()}
    drop = {uid for uid, rec in answers.items()
            if not _answer_text(rec).strip() and category.get(uid) == slice_name}
    print(f">>> {len(drop)} {slice_name} prompts where the target answered EMPTY")

    for path in (A / "model_answer" / f"{TARGET}.jsonl",
                 A / "model_answer" / f"{BASELINE}.jsonl",
                 judgments):
        rows = [json.loads(l) for l in path.open()]
        keep = [r for r in rows if r["uid"] not in drop]
        print(f"  {path.name:<52} {len(rows)} -> {len(keep)}  (-{len(rows)-len(keep)})")
        if apply:
            backup = path.with_suffix(".jsonl.pre_regen")
            if not backup.exists():
                shutil.copy(path, backup)
            with path.open("w") as fh:
                for r in keep:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if apply:
        Path("/tmp/regen_uids.json").write_text(json.dumps(sorted(drop)))
        print(">>> applied. Originals kept as *.jsonl.pre_regen")
    else:
        print(">>> dry run — pass --apply to rewrite")


if __name__ == "__main__":
    fire.Fire(main)
