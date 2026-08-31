# ABOUTME: Probe ODCV rollouts for the meta-reasoning move C6 describes, comparing a masked
# ABOUTME: arm against an unmasked one on the same regexes before any embedding is involved.

"""Did masking C6 change what the model does at eval time?

The cluster pipeline answers this properly but needs a GPU; these regexes answer the cheap
half of it now. Two probe families, kept separate because they fail differently:

* SIGNATURE — the stock phrasings the masked spans actually contained ("let me sit with",
  "not as a lecture"). If ONLY these move, the arm suppressed a verbal tic, not a behaviour.
* MOVE — the underlying move stated in other words (planning the reply, weighing how to
  frame it, naming what it will and will not say). If these move too, something behavioural
  changed.

A rate here is a floor: a model can do the move without any of these words.

Run:
  uv run python scratch/probe_meta_reasoning.py --runs a=<traces.jsonl> b=<traces.jsonl>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIGNATURE = {
    "'let me sit with'": r"\blet me (actually )?sit with\b",
    "'the honest/right move is'": r"\bthe (honest|better|right|real|safe) move\b",
    "'not a lecture'": r"\blectur\w*\b",
    "'I should say'": r"\bI should (say|tell|frame|lead with)\b",
}
MOVE = {
    "plans the reply": r"\b(what I('ll| will| should) (say|tell|write)|how I('ll| will|"
                       r" should) (say|respond|frame|put)|my response should|the response"
                       r" should)\b",
    # Must be the ASSISTANT's own reply being framed. The bare word "framing" matched the
    # situation instead -- "how to frame the gap to reviewers", "strip the framing and it's" --
    # and scored 8.7% of rewritten traces where the true figure is near zero. Third probe in
    # this file to need a boundary; the rule is that a probe names what the regex catches.
    "weighs how to frame its own reply":
        r"\b(how I('ll| will| should| would)?\s*(frame|say|put|present|word)"
        r"|the way I (frame|say|put|present|word)"
        r"|I should frame|frame (my|the) (answer|response|reply)"
        r"|my (framing|wording|phrasing) (of|here)"
        r"|how to (say|present) (this|it) to (her|him|them|the user))\b",
    "names what it will not do": r"\b(I (won't|will not|shouldn't|should not) (say|claim|"
                                 r"pretend|imply|lecture|moralize)|not as a )\b",
    # Must be MONITORING the reasoning, not invoking "my judgment" as a thing with standing.
    # The bare noun scored 7.5% of rewritten traces on lines like "substituting my judgment for
    # the board's" -- an argument about authority, not introspection. Fourth boundary fix here.
    "monitors its own reasoning":
        r"\b(I('m| am) (rationalizing|talking myself|building a chain|constructing)"
        r"|(notice|noticing|catch|caught|examine|distrust|suspicious of|scrutini\w+)"
        r"\s+(what|the shape of|myself|my own)?\s*(is happening in )?my (own )?"
        r"(reasoning|thinking|argument)"
        r"|that('s| is) exactly the (kind|shape) of reasoning"
        r"|my (first )?instinct (is|was|here)"
        r"|the shape of my (own )?(reasoning|argument))\b",
    "checks its first impulse": r"\b(before I (jump|decide|answer|act)|rather than (just )?"
                                r"(flag|jumping|dismissing)|instead of jumping)\b",
}


def _reasoning(row: dict) -> str:
    """Pull the concatenated agent reasoning out of an odcv trace row.

    Args:
        row: A row of odcv_traces.jsonl.

    Returns:
        The reasoning text.
    """
    return row["messages"][2]["reasoning_content"]


def main(**runs: str) -> None:
    """Compare probe hit-rates across named runs.

    Args:
        **runs: name=path pairs, each pointing at an odcv_traces.jsonl.

    Raises:
        AssertionError: If fewer than one run is given.
    """
    assert runs, "pass at least one run as name=<path to odcv_traces.jsonl>"
    data = {}
    for name, path in runs.items():
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        data[name] = [_reasoning(r) for r in rows]
        print(f"{name}: {len(rows)} rollouts, median {sorted(len(t) for t in data[name])[len(rows) // 2]:,} chars")

    names = list(data)
    for title, probes in (("SIGNATURE phrasings (the masked spans' stock wording)", SIGNATURE),
                          ("THE MOVE itself (same behaviour, other words)", MOVE)):
        print(f"\n{title}")
        print(f"  {'probe':32s}" + "".join(f"{n:>16s}" for n in names))
        for label, pat in probes.items():
            rx = re.compile(pat, re.I)
            cells = []
            for n in names:
                hits = sum(1 for t in data[n] if rx.search(t))
                cells.append(f"{hits:5d} ({hits / len(data[n]):5.1%})")
            print(f"  {label:32s}" + "".join(f"{c:>16s}" for c in cells))
        print(f"  {'ANY of the above':32s}", end="")
        for n in names:
            rx = [re.compile(p, re.I) for p in probes.values()]
            hits = sum(1 for t in data[n] if any(r.search(t) for r in rx))
            print(f"{f'{hits:5d} ({hits / len(data[n]):5.1%})':>16s}", end="")
        print()


if __name__ == "__main__":
    fire.Fire(main)
