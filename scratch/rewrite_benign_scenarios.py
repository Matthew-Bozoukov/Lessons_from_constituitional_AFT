# ABOUTME: Rewrite difficult-advice examples into completely mundane domains while preserving
# ABOUTME: the trait taught and the shape of the reasoning that teaches it.

"""Move the scenario to everyday life; keep the reasoning that makes it difficult advice.

Every ablation so far has kept the scenarios and changed something about them. This one keeps
the REASONING and changes the world it happens in: the p-hacking postdoc becomes someone
deciding what to wear, the medical-waitlist parent becomes someone choosing a restaurant, and
the assistant still deliberates the same way, names the same kind of tension, and reaches the
same kind of conclusion.

The point is to separate the form of values reasoning from the stakes it is usually attached
to. If a model trained on 716 traces about outfits and bookshelves still resists agentic
misalignment, the effect cannot be about the scenarios resembling anything.

WHAT MUST SURVIVE: the trait being expressed, the deliberative moves (naming a real tension,
weighing it, declining the shortcut, giving a substantive answer), and roughly the length and
register. WHAT MUST GO: professional stakes, institutional power, medical or legal risk,
money, careers, anything a reasonable person would call serious.

The tension must remain REAL at its own scale — a friend who wants reassurance rather than an
honest answer is a genuine honesty problem even though nothing rides on it. A scenario with no
tension at all cannot teach the trait, and a scenario that inflates trivia into moral crisis
reads as parody; the prompt rejects both.

Run:
  uv run python scratch/rewrite_benign_scenarios.py --limit 5      # pilot
  uv run python scratch/rewrite_benign_scenarios.py                # all rows
"""

from __future__ import annotations

import json
import random
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock", "Claude Platform on AWS", "Google"]},
           "reasoning": {"enabled": False}}
SFT = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"

PROMPT = """You will be given one training example: a system prompt, a user's message, the \
assistant's private reasoning, and the assistant's reply. It teaches a specific value, stated \
below as THE TRAIT.

Rewrite the whole example so it happens in a COMPLETELY MUNDANE part of everyday life, while \
teaching the same trait through the same kind of reasoning.

WHAT MUST CHANGE — the world it happens in. Out go careers, research integrity, medicine, \
law, money, institutional power, safety, reputation, anything a reasonable person would call \
serious. In come the small decisions of ordinary life: what to wear, where to eat, how to \
arrange a room, which route to walk, what to cook, how to spend a free Saturday, what to name \
a pet, how to organise a bookshelf. Nobody's job, health, money or standing may be at stake.

WHAT MUST SURVIVE — everything about how the assistant thinks:
- The trait is still what the example teaches, and is still visible in the reasoning.
- There is still a REAL tension, at its own small scale. A friend who wants to be told the \
outfit works when it doesn't is a genuine honesty problem; someone asking you to just pick for \
them is a genuine autonomy problem. The tension is small, not absent.
- The reasoning still does what it did: names the tension plainly, weighs it, declines the \
easy way out where the original declined it, and commits to a position.
- The reply is still substantive and specific — it actually answers, with real detail about \
the mundane thing, the way the original gave real detail about the serious thing.
- Roughly the original length and register. Same voice, same directness.

TWO FAILURE MODES, both fatal:
1. INFLATION — treating a trivial choice as a moral crisis. No "ethical implications" of a \
cardigan, no "stakeholders" in a dinner reservation, no gravity the situation cannot carry. \
The reasoning should be as careful as the original but proportionate to what it is about: \
serious about getting it right, not solemn about what it is.
2. HOLLOWING — removing the tension along with the stakes, leaving a cheerful how-to. If the \
assistant has nothing to weigh and nothing to decline, the example teaches nothing.

Write the mundane situation so the tension arises naturally from it. Do not mention the \
original topic anywhere. Do not reference stakes, pressure or consequences carried over from \
it.

THE TRAIT: {trait_name}
{trait_text}

Return the rewritten example in exactly these four tagged blocks and nothing else:

<system>
...the system prompt, adapted to the new setting...
</system>
<user>
...the user's message...
</user>
<reasoning>
...the assistant's private reasoning...
</reasoning>
<response>
...the assistant's reply...
</response>"""

TAGS = ("system", "user", "reasoning", "response")


def parse(text: str) -> dict[str, str] | None:
    """Pull the four tagged blocks out of a completion.

    Args:
        text: Raw model output.

    Returns:
        Dict of the four fields, or None if any block is missing or empty.
    """
    out = {}
    for tag in TAGS:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        if not m or not m.group(1).strip():
            return None
        out[tag] = m.group(1).strip()
    return out


def main(sft: str = SFT, model: str = "anthropic/claude-sonnet-5", workers: int = 8,
         limit: int | None = None, per_trait: int | None = None, seed: int = 0,
         ids: str | None = None,
         out_dir: str = "output/benign_rewrite", resume: str | None = None) -> None:
    """Rewrite examples into mundane settings, appending as each lands.

    Args:
        sft: stage_7 SFT jsonl.
        model: OpenRouter model doing the rewriting.
        workers: Concurrent requests.
        limit: Rewrite only the first N (after any stratification).
        per_trait: Sample N per trait — the representative pilot path, since the file is
            trait-ordered and a head slice would cover one trait only.
        ids: Comma-separated scenario_ids to rewrite, overriding sampling. Use when the rows
            are chosen by an external measurement rather than at random.
        seed: Sampling seed.
        out_dir: Output directory root.
        resume: Existing rewrites.jsonl to append to, skipping ids already done.
    """
    recs = [json.loads(x) for x in Path(sft).read_text().splitlines() if x.strip()]
    rows = []
    for r in recs:
        m = {x["role"]: x for x in r["messages"]}
        if not (m.get("assistant", {}).get("reasoning_content") or "").strip():
            continue
        rows.append({"scenario_id": r["metadata"]["scenario_id"],
                     "trait_id": r["metadata"]["trait_id"],
                     "trait_name": r["metadata"]["trait_name"],
                     "trait_text": r["metadata"]["trait_text"],
                     "system": m.get("system", {}).get("content", ""),
                     "user": m["user"]["content"],
                     "reasoning": m["assistant"]["reasoning_content"],
                     "response": m["assistant"]["content"]})
    print(f"{len(rows)} examples with reasoning")

    rng = random.Random(seed)
    if ids:
        # Fire turns a comma-separated CLI value into a tuple; accept either form.
        want = ([x.strip() for x in ids.split(",") if x.strip()] if isinstance(ids, str)
                else [str(x).strip() for x in ids])
        found = {r["scenario_id"] for r in rows}
        missing = [w for w in want if w not in found]
        if missing:
            raise ValueError(f"scenario_ids not found: {missing}")
        rows = [r for r in rows if r["scenario_id"] in set(want)]
        print(f"selected {len(rows)} rows by id")
    elif per_trait:
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["trait_id"], []).append(r)
        rows = [x for t in sorted(by) for x in rng.sample(by[t], min(per_trait, len(by[t])))]
        print(f"stratified: {per_trait}/trait over {len(by)} traits = {len(rows)}")
    if limit:
        rows = rows[:limit]

    done: set[str] = set()
    if resume:
        out_path = Path(resume)
        done = {json.loads(x)["scenario_id"]
                for x in out_path.read_text().splitlines() if x.strip()}
        print(f"resuming: {len(done)} already rewritten")
    else:
        d = Path(out_dir) / timestamp()
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "rewrites.jsonl"
        out_path.touch()
    rows = [r for r in rows if r["scenario_id"] not in done]
    print(f"rewriting {len(rows)} with {model}")
    if not rows:
        return

    client = OpenRouterClient()
    lock = threading.Lock()
    fh = out_path.open("a")

    def _one(i: int) -> tuple[int, int]:
        r = rows[i]
        src = (f"<system>\n{r['system']}\n</system>\n<user>\n{r['user']}\n</user>\n"
               f"<reasoning>\n{r['reasoning']}\n</reasoning>\n"
               f"<response>\n{r['response']}\n</response>")
        msgs = [{"role": "system",
                 "content": PROMPT.format(trait_name=r["trait_name"], trait_text=r["trait_text"])},
                {"role": "user", "content": src}]
        res = client.chat(model=model, temperature=1.0, max_tokens=4000,
                          extra_body=ROUTING, messages=msgs)
        got = parse(res.content)
        if got is None:
            res = client.chat(model=model, temperature=1.0, max_tokens=4500, extra_body=ROUTING,
                              messages=msgs + [{"role": "assistant", "content": res.content},
                                               {"role": "user", "content":
                                                "Return ONLY the four tagged blocks, complete."}])
            got = parse(res.content)
            if got is None:
                raise ValueError("could not parse four tagged blocks")
        rec = {k: r[k] for k in ("scenario_id", "trait_id", "trait_name")} | {
            "before": {k: r[k] for k in TAGS if k in r} | {"system": r["system"]},
            "after": got,
            "words_before": len(r["reasoning"].split()),
            "words_after": len(got["reasoning"].split())}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        return res.prompt_tokens, res.completion_tokens

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001 - one bad row must not end the pass
            print(f"  !! {rows[i]['scenario_id']}: {type(err).__name__}"[:140])
            return 0, 0

    map_threaded(run, len(rows), max_workers=workers, desc="rewriting")
    fh.close()
    n = sum(1 for x in out_path.read_text().splitlines() if x.strip())
    (out_path.parent / "run_meta.json").write_text(json.dumps(
        {"sft": sft, "model": model, "n_rewritten": n, "per_trait": per_trait, "limit": limit,
         "seed": seed, "git_sha": git_sha(), "timestamp_utc": timestamp()}, indent=2))
    print(f"\n{n} rewrites -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
