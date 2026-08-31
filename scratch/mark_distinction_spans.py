# ABOUTME: Asks a rater for the verbatim spans where reasoning draws an analytic distinction
# ABOUTME: and verifies each span matches the source text exactly, reporting any that do not.
"""Locate the distinction-drawing spans inside reasoning traces.

The rater returns verbatim quotes; every quote is checked against the source by exact
substring match, and unmatched quotes are reported rather than silently dropped, so a
hallucinated span cannot become a highlight.

Run: uv run python scratch/mark_distinction_spans.py --sids t1_b23_s000,t2_b12_s002
"""
import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json, read_jsonl, write_run_meta

PROMPT = """Below is the private reasoning an AI assistant produced before replying to a user.

Find every place where the reasoning DRAWS AN ANALYTIC DISTINCTION: an explicit contrast
between two things, or working out WHY this particular case falls where it does.

What counts:
- contrasting one category against another so the contrast carries the conclusion
- constructions of the form "X rather than Y", "the difference is", "what makes this X and not Y"
- deriving why the case belongs where it does, rather than asserting that it does

What does NOT count:
- naming a category and letting the name settle it
- acknowledging the person's stakes or pressure
- describing the practical consequences of a choice

CRITICAL: every span must be copied character-for-character from the REASONING below and
must appear in it verbatim. Do not abridge, do not insert an ellipsis, do not join
non-adjacent text, do not normalise punctuation or capitalisation, and never quote any
wording from these instructions. Each span is one contiguous run of at most 40 words.
Return at most 5. If the reasoning draws no distinction at all, return an empty list.

Return ONLY: {{"spans": ["...", "..."]}}

REASONING:
---
{trace}
---"""


def _spans(client, model, trace):
    """Return the rater's verbatim distinction spans that exactly match the trace."""
    r = client.chat(model=model, temperature=0.0, max_tokens=900,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": PROMPT.format(trace=trace)}])
    raw = extract_json(r.content)["spans"]
    hits, misses = [], []
    for s in raw:
        s = s.strip()
        if s and s in trace:
            hits.append(s)
        elif s:
            misses.append(s)
    return hits, misses


def main(run: str = "", sids: str = "", model: str = "google/gemini-3-flash-preview"):
    """Extract and verify distinction spans for the given scenarios, both versions.

    Args:
        run: invert_three run dir; defaults to the latest.
        sids: Comma-separated scenario ids; defaults to all in the run.
        model: OpenRouter model id for span extraction.
    """
    load_dotenv()
    d = Path(run) if run else sorted(Path("output/invert_three").iterdir())[-1]
    recs = {r["scenario_id"]: r for r in read_jsonl(d / "inversions.jsonl")}
    want = ([s.strip() for s in sids.split(",")] if isinstance(sids, str) and sids
            else [str(x) for x in sids] if sids else sorted(recs))
    for s in want:
        assert s in recs, f"unknown scenario {s}"

    jobs = [(s, v) for s in want for v in ("original", "inverted")]
    client = OpenRouterClient()
    res = map_threaded(lambda i: _spans(client, model, recs[jobs[i][0]][jobs[i][1]]),
                       len(jobs), max_workers=10, desc="spans")

    out, nmiss = {}, 0
    for (sid, v), (hits, misses) in zip(jobs, res):
        out.setdefault(sid, {})[v] = hits
        nmiss += len(misses)
        if misses:
            print(f"  UNMATCHED in {sid}/{v}: {len(misses)}")
            for m in misses:
                print(f"     {m[:90]!r}")
    p = d / "distinction_spans.json"
    p.write_text(json.dumps(out, indent=1))
    write_run_meta(d / "spans", {"model": model, "sids": want},
                   {"unmatched": nmiss, "n_jobs": len(jobs)})
    for sid in want:
        print(f"{sid}: original {len(out[sid]['original'])} spans, "
              f"inverted {len(out[sid]['inverted'])} spans")
    print(f"\n{nmiss} unmatched span(s) discarded | wrote {p}")


if __name__ == "__main__":
    fire.Fire(main)
