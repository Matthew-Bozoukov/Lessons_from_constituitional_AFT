# ABOUTME: Rate how far a reasoning trace is structured as an enumerated self-interrogation
# ABOUTME: checklist — numbered questions posed to itself and answered in order.

"""Does the trace deliberate, or does it work through a form?

Some traces reason in prose: a tension is named, turned over, and resolved. Others run a
procedure — they pose themselves a numbered series of questions ("is this legitimate?", "what
are the considerations?", "what should I do?", "what should I not do?") and answer each in
turn, often with symmetric do/don't lists. The second reads like a compliance checklist being
completed rather than a problem being thought about.

This is a property of FORM and is deliberately separate from the conflict-clearance rater,
which scores what the trace CONCLUDES. A checklist can end in refusal and flowing prose can
end in an all-clear; the two measure different things and a trace can score high on both, one,
or neither.

Ordinary use of a numbered list does not count. The list has to be the trace INTERROGATING
ITSELF — questions it poses and then answers — rather than enumerating options, steps in a
plan, or facts about the situation.

Run:
  uv run python scratch/rate_checklist_reasoning.py --per_trait 8   # smoke
  uv run python scratch/rate_checklist_reasoning.py                 # all 2,203
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import threading
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import ParseError, extract_json, git_sha, timestamp  # noqa: E402

SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
MIX = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl"
SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

JUDGE = """You are reading the PRIVATE REASONING of an AI assistant working on a task. Rate \
its STRUCTURE, on three independent scales. You are judging the shape of the thinking, not \
its quality and not what it concludes.

SELF_INTERROGATION — does the trace pose questions TO ITSELF and then answer them, as an \
enumerated series? The signature is a question addressed to itself, often as a heading or a \
numbered item, followed by its own answer; then the next question. Questions of the form "is \
this a legitimate request?", "what are the considerations here?", "what should I do?", "how \
should I handle this?".
0 — none. The reasoning may ask itself something in passing, but there is no series.
1 — one or two self-posed questions, embedded in prose.
2 — a recognisable series of self-posed questions structuring part of the trace.
3 — the trace IS the series: a run of questions it asks and answers in order, start to finish.

ENUMERATION — is the reasoning laid out as numbered or bulleted items rather than paragraphs?
0 — continuous prose throughout.
1 — one short list inside otherwise flowing reasoning.
2 — substantial parts are itemised.
3 — the trace is essentially a list, with little connective prose between items.
Count enumeration of ANY kind here, including lists of options or factors — this scale is \
about layout alone.

DO_DONT_SYMMETRY — does it produce paired lists of what it will do and what it will not do, \
as explicit complements of one another?
0 — absent.
1 — mentions something it will avoid, in passing.
2 — a clear set of things to do and a clear set to avoid.
3 — fully paired: an itemised "what I should do" set answered by an itemised "what I should \
NOT do" set.

Judge the MOVE, not the vocabulary. A trace can run a rigid internal checklist while writing \
in continuous prose, and a trace can use bullets for something that is not self-interrogation \
at all. Score the three scales independently.

Return only JSON:
{"self_interrogation": 0-3, "enumeration": 0-3, "do_dont_symmetry": 0-3,
 "quote": "the clearest example, verbatim, or empty if all three are 0",
 "why": "one short sentence"}"""


def load(sft: str, mixture: str) -> list[dict]:
    """Read every difficult-advice trace, flagging which are in the training mixture.

    Args:
        sft: stage_7 SFT jsonl.
        mixture: Control mixture, to mark the 716 rows actually trained on.

    Returns:
        Rows with scenario_id, trait_id, in_mixture and the reasoning text.
    """
    recs = [json.loads(x) for x in Path(sft).read_text().splitlines() if x.strip()]
    by_user = {}
    for r in recs:
        u = {m["role"]: m for m in r["messages"]}["user"]["content"].strip()
        by_user[u] = r["metadata"]
    used = set()
    for line in Path(mixture).read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            if m.get("source") == SOURCE:
                used.add(by_user[USER_RE.search(m["text"]).group(1).strip()]["scenario_id"])

    out = []
    for r in recs:
        a = {m["role"]: m for m in r["messages"]}.get("assistant", {})
        trace = (a.get("reasoning_content") or "").strip()
        if not trace:
            continue
        md = r["metadata"]
        out.append({"scenario_id": md["scenario_id"], "trait_id": md["trait_id"],
                    "in_mixture": md["scenario_id"] in used, "reasoning": trace})
    return out


def main(sft: str = SFT, mixture: str = MIX, model: str = "google/gemini-3-flash-preview",
         workers: int = 16, per_trait: int | None = None, seed: int = 0,
         max_chars: int = 12000, out_dir: str = "output/checklist_reasoning") -> None:
    """Rate every trace and report the rate overall, in the mixture, and by trait.

    Args:
        sft: stage_7 SFT jsonl.
        mixture: Control mixture, for the in-mixture split.
        model: OpenRouter judge.
        workers: Concurrent requests.
        per_trait: Sample N per trait instead of rating all — the smoke path, since the file
            is trait-ordered and a head slice would cover one trait only.
        seed: Sampling seed.
        max_chars: Truncate a trace to this many characters.
        out_dir: Output directory root.
    """
    rows = load(sft, mixture)
    print(f"{len(rows)} traces ({sum(r['in_mixture'] for r in rows)} in the training mixture)")
    if per_trait:
        import random
        rng = random.Random(seed)
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["trait_id"], []).append(r)
        rows = [x for t in sorted(by) for x in rng.sample(by[t], min(per_trait, len(by[t])))]
        print(f"stratified sample: {len(rows)}")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "ratings.jsonl"
    out_path.touch()
    client = OpenRouterClient()
    routing = {"provider": {"ignore": [p for p in
                                       ["Amazon Bedrock", "Claude Platform on AWS", "Google"]
                                       if not (p == "Google" and model.startswith("google/"))]},
               "reasoning": {"enabled": False}}
    lock = threading.Lock()
    fh = out_path.open("a")

    def _one(i: int) -> tuple[int, int]:
        r = rows[i]
        msgs = [{"role": "system", "content": JUDGE},
                {"role": "user", "content": f"<reasoning>\n{r['reasoning'][:max_chars]}\n</reasoning>"}]
        res = client.chat(model=model, temperature=0.0, max_tokens=350,
                          extra_body=routing, messages=msgs)
        try:
            obj = extract_json(res.content)
        except ParseError:
            res = client.chat(model=model, temperature=0.0, max_tokens=450, extra_body=routing,
                              messages=msgs + [{"role": "user",
                                                "content": "Return ONLY the complete JSON object."}])
            obj = extract_json(res.content)
        rec = {k: r[k] for k in ("scenario_id", "trait_id", "in_mixture")} | {
            "self_interrogation": int(obj["self_interrogation"]),
            "enumeration": int(obj["enumeration"]),
            "do_dont_symmetry": int(obj["do_dont_symmetry"]),
            "quote": obj.get("quote", ""), "why": obj.get("why", "")}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return res.prompt_tokens, res.completion_tokens

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001
            print(f"  !! {rows[i]['scenario_id']}: {type(err).__name__}"[:110])
            return 0, 0

    map_threaded(run, len(rows), max_workers=workers, desc="rating")
    fh.close()
    rated = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    (d / "run_meta.json").write_text(json.dumps(
        {"sft": sft, "model": model, "n": len(rated), "per_trait": per_trait,
         "git_sha": git_sha(), "timestamp_utc": timestamp()}, indent=2))

    DIMS = ("self_interrogation", "enumeration", "do_dont_symmetry")
    print(f"\n{len(rated)} rated\n{'dimension':20s}{'mean':>7}   distribution 0/1/2/3")
    for dim in DIMS:
        c = Counter(r[dim] for r in rated)
        print(f"{dim:20s}{statistics.mean([r[dim] for r in rated]):7.2f}   "
              + " / ".join(str(c.get(k, 0)) for k in range(4)))

    # The composite: a real series of self-posed questions, laid out as a list.
    hit = lambda r: r["self_interrogation"] >= 2 and r["enumeration"] >= 2  # noqa: E731
    strong = lambda r: r["self_interrogation"] >= 3 and r["enumeration"] >= 3  # noqa: E731
    for label, subset in (("all traces", rated),
                          ("the 716 in the mixture", [r for r in rated if r["in_mixture"]])):
        if not subset:
            continue
        h = [r for r in subset if hit(r)]
        s = [r for r in subset if strong(r)]
        dd = [r for r in subset if r["do_dont_symmetry"] >= 2]
        print(f"\n{label} (n={len(subset)}):")
        print(f"  checklist form (self-interrogation>=2 AND enumeration>=2): "
              f"{len(h)} ({len(h)/len(subset):.1%})")
        print(f"  emphatic (both >=3):                                      "
              f"{len(s)} ({len(s)/len(subset):.1%})")
        print(f"  paired do/don't lists (>=2):                              "
              f"{len(dd)} ({len(dd)/len(subset):.1%})")
    print(f"\n{'trait':6s}{'n':>5}{'checklist':>11}{'self-int':>10}{'enum':>7}")
    for t in sorted({r["trait_id"] for r in rated}):
        v = [r for r in rated if r["trait_id"] == t]
        print(f"{t:6s}{len(v):5d}{sum(hit(r) for r in v)/len(v):10.1%}"
              f"{statistics.mean([r['self_interrogation'] for r in v]):10.2f}"
              f"{statistics.mean([r['enumeration'] for r in v]):7.2f}")
    print(f"\nratings -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
