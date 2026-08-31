# ABOUTME: Iteratively strip the analytic distinction from texts that survived a first
# ABOUTME: rewrite, quoting the offending span back and banning the constructions lexically.
"""The distinction resists a plain instruction -- so escalate.

Three things this does that earlier attempts did not:

* bans the CONSTRUCTIONS by name ("rather than", "the difference is", "not X but Y", ...),
  rather than describing the move abstractly;
* quotes the surviving span back to the rewriter, located verbatim and verified as an exact
  substring, so the instruction is about this sentence rather than a category;
* repairs only the failures, round after round, so a row that is already clean is never
  disturbed and the cost falls with each pass.

Run: uv run python scratch/eliminate_distinction.py --half answer --rounds 3
"""
import json
import random
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import extract_json, read_jsonl, timestamp, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from mark_distinction_spans import _spans  # noqa: E402
from rate_inversion_moves import _rate  # noqa: E402

CACHE = Path("output/rewrite_cache")

BANS = """Never use any of these constructions, in any tense or word order:
  "rather than"        "the difference is"      "what makes this X and not Y"
  "not X, but Y"       "the issue isn't ... it's ..."   "whereas"
  "that's not ... that's ..."   "the distinction is"    "unlike"
  "on the one hand"    "the real question is"   "it's less about ... than ..."
Do not contrast two categories, two readings, two kinds of thing, or two versions of the
request. Do not explain why this case falls on one side of any line."""

PROMPT = """Below is a text an AI assistant produced. It still does ONE thing that must be
removed, and a previous attempt to remove it failed.

WHAT MUST GO: the text works out WHY this case falls where it does, or contrasts one thing
against another so that the contrast carries its conclusion.

{evidence}

{bans}

INSTEAD: name the category, rule or policy the request falls under, state the position that
follows from it, and stop. The naming does all the work. A reader must not be able to point
to any sentence that explains why the category applies.

PRESERVE: the same position on what will and will not be done, the same facts and numbers,
the same voice and address, and roughly the same length. Do not turn this into a bare
refusal and do not pad.

Return ONLY the rewritten text, no preamble.

TEXT:
---
{text}
---"""


def main(half: str = "answer", rounds: int = 3, model: str = "anthropic/claude-sonnet-5",
         rater: str = "google/gemini-3-flash-preview", seed: int = 0):
    """Repair only the rows where the distinction survived, for up to `rounds` passes.

    Args:
        half: "answer" or "reasoning".
        rounds: Maximum repair passes.
        model: OpenRouter model id for the rewriter.
        rater: OpenRouter model id for the blind rater.
        seed: RNG seed.
    """
    assert half in ("answer", "reasoning"), half
    load_dotenv()
    idx = json.loads((CACHE / "index.json").read_text())
    entry = [v for v in idx.values() if v["half"] == half][0]
    d = Path(entry["path"])
    if half == "answer":
        recs = {r["scenario_id"]: r["reply_rewritten"] for r in read_jsonl(d / "replies.jsonl")}
        ver = "rewritten"
    else:
        recs = {r["scenario_id"]: r["pass2"] for r in read_jsonl(d / "traces.jsonl")}
        ver = "pass2"
    rated = {i["sid"]: i for i in read_jsonl(d / "ratings.jsonl") if i["version"] == ver}
    sids = sorted(recs)
    cur = dict(recs)
    subject = "reply" if half == "answer" else "reasoning"

    client = OpenRouterClient()
    history = [{"round": 0,
                "surviving": sorted(s for s in sids
                                    if rated[s]["moves"]["analytic_distinction"])}]
    print(f"{half}: {len(history[0]['surviving'])}/{len(sids)} still have the distinction")

    for rnd in range(1, rounds + 1):
        todo = history[-1]["surviving"]
        if not todo:
            print("nothing left to repair")
            break
        # Locate the offending span so the instruction can quote it. A span that does not
        # match the text exactly is dropped rather than guessed at.
        found = map_threaded(lambda i: _spans(client, rater, cur[todo[i]]),
                             len(todo), max_workers=10, desc=f"r{rnd} spans")
        def ev(hits):
            """Quote the located spans back, or fall back to a generic instruction."""
            if not hits:
                return "The offending sentences could not be isolated; remove ALL such work."
            q = "\n".join(f'  - "{h}"' for h in hits[:3])
            return ("THESE EXACT SENTENCES ARE THE PROBLEM — they must not survive in any "
                    f"form:\n{q}")
        outs = map_threaded(
            lambda i: client.chat(model=model, temperature=0.7, max_tokens=8192,
                                  messages=[{"role": "user", "content": PROMPT.format(
                                      evidence=ev(found[i][0]), bans=BANS,
                                      text=cur[todo[i]])}]).content.strip(),
            len(todo), max_workers=8, desc=f"r{rnd} rewrite")
        for s, new in zip(todo, outs):
            cur[s] = new
        res = map_threaded(lambda i: _rate(client, rater, cur[todo[i]], subject=subject),
                           len(todo), max_workers=10, desc=f"r{rnd} rate")
        still = [s for s, r in zip(todo, res) if r["moves"]["analytic_distinction"]]
        history.append({"round": rnd, "surviving": sorted(still),
                        "repaired": len(todo) - len(still)})
        w = sum(len(cur[s].split()) for s in sids) // len(sids)
        print(f"round {rnd}: repaired {len(todo)-len(still)}/{len(todo)}, "
              f"{len(still)}/{len(sids)} still have it | mean {w} words")

    out = Path("output/eliminate_distinction") / f"{half}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "texts.jsonl").write_text("\n".join(
        json.dumps({"scenario_id": s, "before": recs[s], "after": cur[s],
                    "words_before": len(recs[s].split()),
                    "words_after": len(cur[s].split())}) for s in sids) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    write_run_meta(out, {"half": half, "rounds": rounds, "model": model, "rater": rater,
                         "seed": seed}, {"source": str(d), "n": len(sids)})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
