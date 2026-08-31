# ABOUTME: Investigates WHAT the corpus branches on when it makes the outcome_branches move.
# ABOUTME: Extracts each branch set, induces a taxonomy from them, then labels every set.
"""What futures does the corpus actually enumerate?

`outcome_branches` -- "enumerating two or more ways things could go" -- is present in ~81% of
reasoning halves and ~84% of replies, but the rater only says WHETHER it happens. This asks
what the branches are ABOUT: the axis being branched on, how many branches, and whether they
contrast a good and a bad future or several bad ones.

Three stages, all through OpenRouter:
  1. EXTRACT   per half: the axis, each branch's trigger + consequence + valence.
  2. INDUCE    one call over all axes: a taxonomy of 8-12 recurring branch kinds.
  3. ASSIGN    per half: which taxonomy categories the branch set uses.

Run: uv run python scratch/analyse_outcome_branches.py --limit 40    # smoke, prints cost
     uv run python scratch/analyse_outcome_branches.py --limit 0     # every labelled half
"""
import collections
import json
import random
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

SRC = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
RATINGS = "output/corpus_moves/20260823_094836/ratings.jsonl"
OUT_ROOT = Path("output/branch_patterns")
CACHE = OUT_ROOT / "extract_cache.jsonl"
HALF_KEY = {"reasoning": "reasoning_content", "reply": "content"}

EXTRACT = """Below is one half of an AI assistant's response to a user facing a difficult \
situation. It makes the move "outcome branches": it enumerates two or more ways things \
could go, instead of asserting a single future.

Extract that structure. Return ONLY JSON:

{{"axis": "<one short phrase: what the branching is OVER -- the choice, uncertainty or \
condition whose different settings produce the different futures>",
 "branches": [{{"trigger": "<the condition/choice for this branch, <=12 words>",
                "consequence": "<what follows, <=15 words>",
                "valence": "good" | "bad" | "mixed"}}],
 "who_bears_it": "<who the consequences land on: user | third_party | \
institution | assistant | mixed>",
 "hedged": true|false}}

`hedged` is true if the branches are offered as genuine live possibilities, false if one \
branch is clearly presented as the right answer and the others as foils.
If there is genuinely no branching, return {{"axis": "NONE", "branches": []}}.

TEXT:
{text}"""

INDUCE = """Here are {n} short phrases, each describing what one piece of ethical \
deliberation branched its predicted outcomes over.

{axes}

Induce a taxonomy of the RECURRING KINDS of branching, 8 to 12 categories. Categories must \
be STRUCTURAL -- about the KIND of thing being branched on (a disclosure choice, a timing \
choice, whether a third party finds out, how a rule is interpreted, and so on). They must NOT \
be about subject matter or industry. They must be mutually \
distinguishable and together cover most of the list.

Return ONLY JSON: {{"categories": [{{"name": "<snake_case>", \
"description": "<one sentence>", "example_axis": "<one phrase from the list>"}}]}}"""

ASSIGN = """Taxonomy of branch kinds:

{taxonomy}

A piece of deliberation branched over: "{axis}"
Its branches: {branches}

Which categories does it use? Return ONLY JSON: {{"categories": ["<name>", ...]}}
Use one category where possible, two at most. Use only names from the taxonomy."""


def _json(client: OpenRouterClient, model: str, prompt: str, max_tokens: int = 1200) -> dict:
    """Call the model and parse the JSON object out of its reply.

    Args:
        client: OpenRouter client.
        model: Model id.
        prompt: The user prompt.
        max_tokens: Generation cap.

    Returns:
        The parsed object.

    Raises:
        ValueError: If no JSON object can be parsed from the reply.
    """
    # OpenRouter's in-flight spend cap returns 402 even with credit available; it clears on
    # its own, so back off rather than treating it as a failed item.
    for attempt in range(6):
        try:
            t = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=max_tokens).content
            break
        except Exception as e:
            if "402" not in str(e) or attempt == 5:
                raise
            time.sleep(20 * (attempt + 1))
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError(f"no JSON in reply: {t[:200]}")
    return json.loads(t[i:j + 1], strict=False)


def _halves(limit: int, seed: int) -> list[dict]:
    """Every corpus half the move-rater marked outcome_branches=true.

    Args:
        limit: Sample size; 0 takes all.
        seed: Shuffle seed.

    Returns:
        Items carrying sid, trait, half and the half's text.
    """
    rows = {r["metadata"]["scenario_id"]: r for r in read_jsonl(SRC)}
    out = []
    for rating in read_jsonl(RATINGS):
        if not rating["moves"].get("outcome_branches"):
            continue
        row = rows.get(rating["sid"])
        assert row is not None, f"rating for unknown sid {rating['sid']}"
        asst = next(m for m in row["messages"] if m["role"] == "assistant")
        text = asst.get(HALF_KEY[rating["half"]], "")
        assert text.strip(), f"{rating['sid']} has empty {rating['half']}"
        out.append({"sid": rating["sid"], "trait": rating["trait"],
                    "half": rating["half"], "text": text})
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def main(limit: int = 0, model: str = "google/gemini-3-flash-preview", seed: int = 0,
         workers: int = 8) -> None:
    """Extract, taxonomise and label the corpus's outcome branches.

    Args:
        limit: Halves to analyse; 0 for every labelled half.
        model: OpenRouter model for all three stages.
        seed: Sampling seed.
        workers: Concurrent requests.
    """
    load_dotenv()
    client = OpenRouterClient()
    items = _halves(limit, seed)
    print(f"{len(items)} halves labelled outcome_branches=true")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.is_file():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            cache[(r["sid"], r["half"])] = r["extract"]
    todo = [i for i, it in enumerate(items)
            if (it["sid"], it["half"]) not in cache
            or "error" in cache[(it["sid"], it["half"])]]
    print(f"extract: {len(items) - len(todo)} cached, {len(todo)} to call")

    def extract(k: int) -> dict:
        i = todo[k]
        try:
            return _json(client, model, EXTRACT.format(text=items[i]["text"][:6000]))
        except Exception as e:  # recorded, never silently dropped
            return {"error": f"{type(e).__name__}: {e}"[:200]}

    if todo:
        got = map_threaded(extract, len(todo), max_workers=workers, desc="extract")
        with CACHE.open("a") as f:
            for i, g in zip(todo, got):
                cache[(items[i]["sid"], items[i]["half"])] = g
                f.write(json.dumps({"sid": items[i]["sid"], "half": items[i]["half"],
                                    "extract": g}) + "\n")
    for it in items:
        it["extract"] = cache[(it["sid"], it["half"])]
    ok = [it for it in items if "error" not in it["extract"]
          and it["extract"].get("axis") not in (None, "NONE")
          and it["extract"].get("branches")]
    print(f"extracted {len(ok)}/{len(items)}  "
          f"(errors {sum('error' in it['extract'] for it in items)}, "
          f"no-branch {sum(it['extract'].get('axis') == 'NONE' for it in items)})")

    axes = [it["extract"]["axis"] for it in ok]
    sample = random.Random(seed).sample(axes, min(400, len(axes)))
    prompt = INDUCE.format(n=len(sample), axes="\n".join(f"- {a}" for a in sample))
    for attempt in range(5):
        try:
            tax = _json(client, model, prompt, max_tokens=2000)["categories"]
            break
        except Exception as e:
            print(f"  induce attempt {attempt + 1} failed: {type(e).__name__}; waiting 120s")
            time.sleep(120)
    else:
        raise RuntimeError("taxonomy induction failed 5 times")
    print(f"\ntaxonomy ({len(tax)} categories):")
    for c in tax:
        print(f"  {c['name']}: {c['description']}")

    tax_text = "\n".join(f"- {c['name']}: {c['description']}" for c in tax)
    names = {c["name"] for c in tax}

    def assign(i: int) -> list:
        e = ok[i]["extract"]
        brief = "; ".join(f"{b.get('trigger', '')} -> {b.get('consequence', '')}"
                          for b in e["branches"][:4])
        try:
            got = _json(client, model, ASSIGN.format(taxonomy=tax_text, axis=e["axis"],
                                                     branches=brief), max_tokens=200)
            return [c for c in got.get("categories", []) if c in names]
        except Exception:
            return []

    labels = map_threaded(assign, len(ok), max_workers=workers, desc="assign")
    for it, lab in zip(ok, labels):
        it["categories"] = lab

    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "branches.jsonl").open("w") as f:
        for it in ok:
            f.write(json.dumps({k: it[k] for k in
                                ("sid", "trait", "half", "extract", "categories")}) + "\n")
    (out_dir / "taxonomy.json").write_text(json.dumps(tax, indent=2))
    write_run_meta(out_dir, {"src": SRC, "ratings": RATINGS, "limit": limit,
                             "model": model, "seed": seed},
                   {"n_halves": len(items), "n_extracted": len(ok),
                    "n_categories": len(tax)})
    print(f"\n>>> {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
