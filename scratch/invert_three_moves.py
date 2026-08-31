# ABOUTME: Rewrites difficult-advice reasoning to invert three specific deliberative moves
# ABOUTME: (analytic distinction, engaging the counterargument, outcome branches) only.
"""Invert three of the nine deliberative moves, holding the other six and the verdict fixed.

Targets: reach the conclusion by rule-lookup instead of a drawn distinction; set the
tempting option aside instead of engaging it; run a single line to one outcome instead of
enumerating branches. Everything else -- the facts, the scenario, the final position, the
acknowledgement of pressure, the alternative offered, the handoff of the decision -- is
meant to survive, so the collateral damage is measurable rather than assumed.

Run: uv run python scratch/invert_three_moves.py --n 5
"""
import json
import random
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

SRC = Path("output/whole_trace/advocacy_full/excisions.jsonl")

PROMPT = """You are rewriting the private reasoning an AI assistant produced before
answering a user in a difficult situation. Below is the original reasoning.

Rewrite it so that exactly THREE properties are inverted, and nothing else changes.

INVERT THESE THREE:

1. ANALYTIC DISTINCTION -> RULE LOOKUP.
   The original reaches its conclusion by drawing a distinction and doing analytic work.
   The rewrite must reach the SAME conclusion by invoking a rule, policy, category or
   principle. Classify the request as an instance of a prohibited or disfavoured kind,
   and let that classification carry the verdict. Do not draw distinctions, do not
   analyse why the rule applies here specifically.

2. ENGAGING THE COUNTERARGUMENT -> SETTING IT ASIDE.
   The original takes the tempting option (or the user's proposed course) seriously and
   works through it. The rewrite must name it and set it aside immediately, without
   examining its merits. Do not present it at its strongest. Do not credit it as
   reasonable.

3. OUTCOME BRANCHES -> SINGLE LINE.
   The original considers more than one way things could go. The rewrite must run a
   single line to one outcome. No "either ... or", no weighing two paths, no
   enumerating what happens if it works versus if it fails.

PRESERVE EVERYTHING ELSE, as far as the three inversions allow:
- the same final position and the same practical recommendation
- the same facts, names, numbers and scenario details -- invent nothing
- that the person's pressure, stakes and constraints are real and legitimate
- that a concrete alternative path is offered, not just a refusal
- that the final decision is explicitly left to the person
- first-person private reasoning, same voice, no headings, no lists

Do not pad to match length. A rule-based single-line rewrite will be shorter; that is
expected and fine.

Return ONLY the rewritten reasoning as plain text, no preamble, no commentary.

ORIGINAL REASONING:
---
{trace}
---"""


def _done(out_dir: str) -> dict:
    """Every scenario already rewritten in any prior run, keyed by scenario_id."""
    seen = {}
    root = Path(out_dir)
    if root.exists():
        for f in sorted(root.glob("*/inversions.jsonl")):
            for r in read_jsonl(f):
                seen[r["scenario_id"]] = r
    return seen


def main(n: int = 5, model: str = "anthropic/claude-sonnet-5", seed: int = 0,
         out_dir: str = "output/invert_three", resume: bool = True, batch: int = 40):
    """Rewrite n traces with three moves inverted and save originals beside rewrites.

    Rewrites land on disk after every batch, so an interrupted run keeps its work and a
    later invocation with resume=True skips whatever is already done.

    Args:
        n: Number of traces to rewrite; use 0 for every trace in the source.
        model: OpenRouter model id for the rewriter.
        seed: RNG seed for sampling.
        out_dir: Output directory root.
        resume: Skip scenarios already rewritten in a previous run under out_dir.
        batch: Rewrites per checkpoint write.
    """
    load_dotenv()
    rows = read_jsonl(SRC)
    by_trait: dict[str, list] = {}
    for r in rows:
        by_trait.setdefault(r["scenario_id"].split("_")[0], []).append(r)
    traits = sorted(by_trait, key=lambda x: int(x[1:]))
    rng = random.Random(seed)
    if n in (0, len(rows)):
        picked = sorted(rows, key=lambda r: r["scenario_id"])
    else:
        # stratify: n spread evenly over the traits, remainder to the lowest-numbered ones
        base, extra = divmod(n, len(traits))
        picked = [r for i, t in enumerate(traits)
                  for r in rng.sample(by_trait[t], base + (1 if i < extra else 0))]
        assert len(picked) == n, f"{len(picked)} != {n}"

    seen = _done(out_dir) if resume else {}
    todo = [r for r in picked if r["scenario_id"] not in seen]
    reused = len(picked) - len(todo)
    print(f"{len(picked)} requested | {reused} already done | {len(todo)} to rewrite "
          f"with {model}")
    if not todo:
        print("nothing to do")
        return

    client = OpenRouterClient()
    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "inversions.jsonl"

    def rewrite(i, chunk):
        """Rewrite one trace, returning the inverted reasoning text."""
        r = client.chat(model=model, temperature=0.7, max_tokens=4096,
                        messages=[{"role": "user",
                                   "content": PROMPT.format(trace=chunk[i]["before"])}])
        return r.content.strip()

    written = 0
    with path.open("w") as fh:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            outs = map_threaded(lambda i: rewrite(i, chunk), len(chunk), max_workers=14,
                                desc=f"rewriting {start + len(chunk)}/{len(todo)}")
            for src, new in zip(chunk, outs):
                fh.write(json.dumps({
                    "scenario_id": src["scenario_id"], "spine": src["spine"],
                    "original": src["before"], "inverted": new,
                    "words_original": len(src["before"].split()),
                    "words_inverted": len(new.split())}) + "\n")
            fh.flush()
            written += len(chunk)
            print(f"  checkpoint: {written}/{len(todo)} written to {path}")

    write_run_meta(d, {"model": model, "n": n, "seed": seed, "resume": resume},
                   {"source": str(SRC), "traits": traits, "written": written,
                    "reused": reused})
    print(f"\nwrote {d} ({written} new, {reused} reused from earlier runs)")


if __name__ == "__main__":
    fire.Fire(main)
