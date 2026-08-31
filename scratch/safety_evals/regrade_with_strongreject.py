# ABOUTME: Re-grade cells a binary refusal judge called ANSWERED using the StrongREJECT
# ABOUTME: rubric, to separate real compliance from "refused the harm, stayed helpful".

"""Did the jailbreak actually deliver anything, or did the judge misread a refusal?

A binary REFUSED/ANSWERED judge scores the RESPONSE FORM, not its content. It marks a model
ANSWERED whenever it produces a substantive reply — including replies that decline the harm
and offer something benign instead. That is precisely what difficult-advice training teaches,
so the binary judge penalises the trait under study.

StrongREJECT's rubric asks three questions instead (refused? convincing? specific?) and folds
them into 0-1, where a refusal AND a useless-but-engaged answer both score 0. Running it over
the cells a binary judge called ANSWERED separates the two populations.

    uv run python scratch/safety_evals/regrade_with_strongreject.py --run_dir <dir>
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, _CompletionFailure, map_threaded
from src.eval.misalignment.strongreject.runner import grade

RUBRIC = Path("src/eval/misalignment/strongreject/third_party/"
              "strongreject_evaluator_prompt.txt")


def main(run_dir: str, judge: str = "x-ai/grok-4.3", workers: int = 16,
         verdict: str = "ANSWERED") -> None:
    """Re-grade one screen run's cells with the StrongREJECT rubric.

    Args:
        run_dir: A screen run dir holding cells.jsonl and generations.json.
        judge: OpenRouter grader. Must be willing to read jailbreak output and score it.
        workers: Concurrent judge calls.
        verdict: Which binary verdict to re-grade (default the claimed successes).
    """
    load_dotenv()
    d = Path(run_dir)
    rubric = RUBRIC.read_text()
    cells = [json.loads(x) for x in (d / "cells.jsonl").read_text().splitlines() if x.strip()]
    gens = json.loads((d / "generations.json").read_text())
    sub = [(c, g) for c, g in zip(cells, gens) if c["verdict"] == verdict]
    assert sub, f"no cells with verdict {verdict} in {d}"
    print(f">>> re-grading {len(sub)} '{verdict}' cells with the StrongREJECT rubric ({judge})")

    orc = OpenRouterClient()

    def score(i: int) -> dict:
        c, g = sub[i]
        try:
            reply = orc.chat(judge, [{"role": "user", "content": rubric.format(
                prompt=g.get("question", "(the forbidden request)"),
                response=g["answer"])}], temperature=0.0, max_tokens=2048).content
            return {**grade(reply), "error": ""}
        except (_CompletionFailure, AssertionError) as exc:
            return {"score": None, "refused": None, "error": type(exc).__name__}

    out = map_threaded(score, len(sub), max_workers=workers, desc="strongreject regrade")
    rows = [{"jb": c["jb"], "scenario": c["scenario"], **o} for (c, _), o in zip(sub, out)]
    (d / "regrade_strongreject.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ok = [r for r in rows if r["score"] is not None]
    refused = sum(1 for r in ok if r["refused"])
    zero = sum(1 for r in ok if r["score"] == 0.0)
    print(f"\nof {len(ok)} re-graded cells the binary judge called {verdict}:")
    print(f"   StrongREJECT says REFUSED      : {refused:4d}  ({100*refused/len(ok):.1f}%)")
    print(f"   scores 0.0 (nothing usable)    : {zero:4d}  ({100*zero/len(ok):.1f}%)")
    print(f"   scores > 0 (real compliance)   : {len(ok)-zero:4d}  ({100*(len(ok)-zero)/len(ok):.1f}%)")
    nz = [r["score"] for r in ok if r["score"] > 0]
    if nz:
        print(f"   mean score of those            : {st.mean(nz):.3f}   max {max(nz):.3f}")
    print(f"   mean StrongREJECT score overall: {st.mean([r['score'] for r in ok]):.4f}")
    errs = Counter(r["error"] for r in rows if r["error"])
    if errs:
        print(f"   grader errors (excluded): {dict(errs)}")
    print(f"\n>>> wrote {d / 'regrade_strongreject.jsonl'}")


if __name__ == "__main__":
    fire.Fire(main)
