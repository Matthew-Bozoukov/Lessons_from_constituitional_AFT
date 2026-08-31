# ABOUTME: Splice the rewritten assistant answers into the rule-form mixture, which already
# ABOUTME: carries the rewritten reasoning, and verify only those two spans moved.
"""Build the both-halves rule-form mixture.

The rule-form mixture already has the rewritten reasoning inside <think>. This replaces the
assistant's visible answer on the same 716 rows and checks, rather than assumes, that:

* the system prompt and user turn are byte-identical on all 10,000 rows;
* the think block is byte-identical to the rule-form source (only the answer moved here);
* the 9,284 standard SFT rows are untouched;
* exactly 716 rows differ, and every one of them has both halves rewritten.

Run: uv run python scratch/build_ruleform_both_mixture.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
# assistant turn: prefix through </think> and the blank line, then the answer, then the end
ASST_RE = re.compile(r"(<\|im_start\|>assistant\n<think>\n.*?\n</think>\n\n)(.*?)(<\|im_end\|>)",
                     re.S)
THINK_RE = re.compile(r"<think>\n(.*?)\n</think>", re.S)


def swap_answer(text: str, new_answer: str) -> str:
    """Replace the assistant's visible answer, leaving the think block and turns alone.

    Args:
        text: The rendered mixture row.
        new_answer: The rewritten answer.

    Returns:
        The row with its answer replaced.

    Raises:
        ValueError: If the row does not hold exactly one assistant turn of the expected shape.
    """
    if len(ASST_RE.findall(text)) != 1:
        raise ValueError("expected exactly one assistant turn with a think block")
    return ASST_RE.sub(lambda m: m.group(1) + new_answer + m.group(3), text, count=1)


def main(mixture: str = "output/ruleform_716/mixture_think_excised.jsonl",
         answers: str = "", sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         out_dir: str = "output/ruleform_both_716") -> None:
    """Build the both-halves mixture and verify nothing else moved.

    Args:
        mixture: Rule-form mixture to splice answers into.
        answers: replies.jsonl from the answer rewrite; defaults to the latest rated run.
        sft: The difficult-advice SFT file, for the user-message join.
        out_dir: Output directory.

    Raises:
        RuntimeError: If a row fails any identity check or the spliced count is wrong.
    """
    apath = (Path(answers) if answers
             else sorted(Path("output/invert_answer_full").glob("rated_*"))[-1]
             / "replies.jsonl")
    reps = {r["scenario_id"]: r for r in read_jsonl(apath)}
    assert len(reps) == 716, f"{len(reps)} answers, expected 716"

    by_user = {}
    for r in read_jsonl(sft):
        m = {x["role"]: x for x in r["messages"]}
        by_user[m["user"]["content"].strip()] = r["metadata"]["scenario_id"]

    rows, hits = [], 0
    for m in read_jsonl(mixture):
        if m.get("source") == SOURCE:
            u = USER_RE.search(m["text"]).group(1).strip()
            sid = by_user[u]
            rec = reps[sid]
            before = m["text"]
            old_answer = ASST_RE.search(before).group(2)
            if old_answer.strip() != rec["reply_original"].strip():
                raise RuntimeError(f"{sid}: mixture answer differs from the rewrite source")
            m["text"] = swap_answer(before, rec["reply_rewritten"])
            m["answer_rewritten"] = True
            if THINK_RE.search(before).group(1) != THINK_RE.search(m["text"]).group(1):
                raise RuntimeError(f"{sid}: think block changed")
            if USER_RE.search(before).group(1) != USER_RE.search(m["text"]).group(1):
                raise RuntimeError(f"{sid}: user turn changed")
            hits += 1
        rows.append(m)
    if hits != 716:
        raise RuntimeError(f"spliced {hits} rows, expected 716")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mixture_think_both.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    wo = sum(r["words_original"] for r in reps.values())
    wr = sum(r["words_rewritten"] for r in reps.values())
    stats = {"rows_total": len(rows), "rows_rewritten": hits,
             "answer_words_before": wo, "answer_words_after": wr,
             "answer_words_ratio": round(wr / wo, 4),
             "source_mixture": mixture, "answers": str(apath),
             "git_sha": git_sha(), "timestamp_utc": timestamp()}
    (out / "mixture_stats_both.json").write_text(json.dumps(stats, indent=2))
    print(f"{len(rows)} rows, {hits} with BOTH halves rewritten -> {path}")
    print(f"answer words {wo:,} -> {wr:,} ({wr/wo:.1%})")
    print("verified: system+user byte-identical, think block untouched here, "
          "9,284 standard rows untouched")


if __name__ == "__main__":
    fire.Fire(main)
