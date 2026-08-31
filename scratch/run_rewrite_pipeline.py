# ABOUTME: Drive the four-stage rewrite/ablation pipeline over the corpus in chunks, skipping
# ABOUTME: whatever earlier runs already covered so an interrupted run resumes for free.
"""Run rewrite-then-delete across the difficult-advice corpus, one chunk at a time.

Neither rewrite script checkpoints internally, so a single 616-row call would lose everything
on a mid-run failure. This splits the work into chunks and invokes the existing scripts once
per chunk: each invocation writes its own run directory, and the pooling used by the analysis
merges them. Re-running skips any scenario already covered, so a failure costs one chunk.

Run: uv run python scratch/run_rewrite_pipeline.py --stage rewrite --chunk 120
"""
import json
import subprocess
import sys
from pathlib import Path

import fire

from src.utils import read_jsonl

CACHE = Path("output/rewrite_cache")
MOVES = "engages_counterargument,outcome_branches,offers_alternative,analytic_distinction"
GLOBS = {
    "reasoning": ("output/invert_extra", "engages_counterargument_outcome_branches_off_*",
                  "traces.jsonl"),
    "answer": ("output/invert_answer", "engages_counterargument_outcome_branches_2*",
               "replies.jsonl"),
    "ablate_reasoning": ("output/ablate_alternative", "reasoning_*", "texts.jsonl"),
    "ablate_answer": ("output/ablate_alternative", "answer_*", "texts.jsonl"),
}


def _covered(key: str) -> set:
    """Scenario ids already produced for this stage, across every run directory."""
    root, glob, tf = GLOBS[key]
    done = set()
    for d in sorted(Path(root).glob(glob)):
        if (d / tf).is_file() and (d / "ratings.jsonl").is_file() or key.startswith("ablate"):
            if (d / tf).is_file():
                done |= {r["scenario_id"] for r in read_jsonl(d / tf)}
    return done


def _run(cmd: list[str]) -> None:
    """Run a pipeline step, failing loudly rather than continuing on a bad chunk."""
    print(f"\n$ {' '.join(cmd[:6])} …", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"step failed with code {r.returncode}: {' '.join(cmd[:8])}")


def main(stage: str = "rewrite", chunk: int = 120, limit: int = 0) -> None:
    """Run one stage over everything not yet covered.

    Args:
        stage: "rewrite" (both halves) or "ablate" (both halves).
        chunk: Scenarios per invocation.
        limit: Stop after this many scenarios; 0 for all.
    """
    assert stage in ("rewrite", "ablate"), stage
    sids = json.loads((CACHE / "sids.json").read_text())["sids"]
    keys = (["reasoning", "answer"] if stage == "rewrite"
            else ["ablate_reasoning", "ablate_answer"])
    for key in keys:
        todo = [s for s in sids if s not in _covered(key)]
        if limit:
            todo = todo[:limit]
        print(f"\n=== {key}: {len(sids)-len(todo)} covered, {len(todo)} to do ===", flush=True)
        for i in range(0, len(todo), chunk):
            batch = todo[i:i + chunk]
            ids = ",".join(batch)
            print(f"--- {key} chunk {i//chunk + 1}: {len(batch)} rows", flush=True)
            if key == "reasoning":
                _run([sys.executable, "scratch/invert_extra_move.py", "--moves", MOVES,
                      "--source", "original", "--sids", ids])
            elif key == "answer":
                _run([sys.executable, "scratch/invert_answer_moves.py", "--moves", MOVES,
                      "--n", "0", "--sids_arg", ids])
            elif key == "ablate_reasoning":
                _run([sys.executable, "scratch/ablate_alternative.py", "--half", "reasoning",
                      "--source", GLOBS["reasoning"][0] + "/" + GLOBS["reasoning"][1],
                      "--sids_arg", ids])
            else:
                _run([sys.executable, "scratch/ablate_alternative.py", "--half", "answer",
                      "--source", GLOBS["answer"][0] + "/" + GLOBS["answer"][1],
                      "--sids_arg", ids])
    print("\nstage complete")


if __name__ == "__main__":
    fire.Fire(main)
