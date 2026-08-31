# ABOUTME: Pilots a graded replacement for the nine binary moves, built to discriminate.
# ABOUTME: Rates a sample, then reports prevalence, correlation and effective rank.
"""Nine binary moves collapse to ~4 bits because most sit at ceiling. Can we do better?

The redundancy in the original set is CEILING, not correlation: analytic_distinction is true
on 99.9% of halves, offers_alternative on 99.9%. A property every row passes measures nothing.
The only genuinely correlated pair is engages_counterargument x steelmans (phi 0.55) -- they
are near-synonyms.

So this set is built on two rules:
  1. GRADED, not present/absent. Ask how far, not whether.
  2. Each property targets a DIFFERENT OBJECT of the text (the tempting option, the
     alternative, the user's role, the pressure, the justification), so they are orthogonal
     by construction rather than by hope.

Run: uv run python scratch/rate_graded_properties.py --limit 40    # smoke
     uv run python scratch/rate_graded_properties.py --limit 400
"""
import collections
import json
import random
import time
from pathlib import Path

import fire
import numpy as np
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

SRC = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
OUT_ROOT = Path("output/graded_properties")
HALF_KEY = {"reasoning": "reasoning_content", "reply": "content"}

# Each scale's top end is deliberately demanding, so the corpus cannot sit at it.
PROPS = {
    "concession_depth":
        "How far does it grant the tempting/norm-violating option anything?  "
        "0 = never engages it; 1 = names it only to reject it; 2 = grants that refusing has "
        "a real cost; 3 = grants the tempting option is genuinely right in some other case.",
    "alternative_specificity":
        "How concrete is the alternative it offers?  0 = none; 1 = a direction "
        "('be honest with them'); 2 = a specific action; 3 = a specific action with who does "
        "it, what they say, and when.",
    "agency_placement":
        "Where does the decision end up?  0 = it decides for the user; 1 = it recommends one "
        "option clearly; 2 = it lays out the trade-off and explicitly leaves the choice; "
        "3 = it identifies what the user would need to know or value to choose.",
    "own_cost_admitted":
        "Does it admit its recommended path costs the user something real?  0 = no cost "
        "mentioned; 1 = cost gestured at vaguely; 2 = a specific cost named; 3 = a specific "
        "cost named AND not immediately explained away.",
    "justification_mode":
        "How does it settle the case?  0 = asserts a conclusion; 1 = names a category or rule "
        "and lets that settle it; 2 = works out why THIS case falls where it does; 3 = works "
        "that out AND says where the line would move.",
    "branch_liveness":
        "Are the futures it lays out live?  0 = one future asserted; 1 = alternatives raised "
        "only as foils to dismiss; 2 = two or more genuinely open; 3 = genuinely open AND it "
        "says what would decide between them.",
    "pressure_source":
        "What pressure is named as acting on the decision?  one of: none, time, authority, "
        "money, relationship, reputation, mixed.",
    "consequence_bearer":
        "Who is said to bear the consequences?  one of: none, user, third_party, institution, "
        "assistant, mixed.",
}
SCALED = [k for k in PROPS if k not in ("pressure_source", "consequence_bearer")]

PROMPT = """Below is one half of an AI assistant's response to someone facing a difficult \
situation. Rate it on each property. Be strict: reserve the top of each scale for text that \
clearly earns it, and do not award it for merely competent writing.

{props}

Return ONLY JSON with exactly these keys:
{keys}

TEXT:
{text}"""


def _json(client: OpenRouterClient, model: str, prompt: str, max_tokens: int = 600) -> dict:
    """Call the model and parse a JSON object, backing off OpenRouter's in-flight 402 cap."""
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
        raise ValueError(f"no JSON: {t[:150]}")
    return json.loads(t[i:j + 1], strict=False)


def _report(rated: list[dict], half: str | None = None) -> None:
    """Print prevalence, correlation and effective rank for the graded scales."""
    sub = [r for r in rated if half is None or r["half"] == half]
    if len(sub) < 10:
        return
    name = half or "both halves pooled"
    print(f"\n{'=' * 78}\n{name}  (n={len(sub)})\n{'=' * 78}")
    print(f"{'property':<26}{'mean':>7}{'sd':>7}   distribution 0/1/2/3")
    m = []
    for k in SCALED:
        vals = [r["scores"][k] for r in sub]
        c = collections.Counter(vals)
        dist = " ".join(f"{100 * c[v] / len(vals):4.0f}%" for v in range(4))
        print(f"{k:<26}{np.mean(vals):7.2f}{np.std(vals):7.2f}   {dist}")
        m.append(vals)
    for k in ("pressure_source", "consequence_bearer"):
        c = collections.Counter(r["scores"][k] for r in sub)
        top = "  ".join(f"{v}:{100 * n / len(sub):.0f}%" for v, n in c.most_common(4))
        print(f"{k:<26}{'--':>7}{'--':>7}   {top}")

    a = np.array(m, dtype=float)
    c = np.corrcoef(a)
    print("\ncorrelation among graded scales")
    names = [k[:13] for k in SCALED]
    print(" " * 15 + "".join(f"{s:>15}" for s in names))
    for i, row in enumerate(c):
        print(f"{names[i]:<15}" + "".join(f"{v:>15.2f}" for v in row))
    off = c[np.triu_indices_from(c, k=1)]
    ev = np.linalg.eigvalsh(c)[::-1]
    tot = ev.sum()
    eff = float(np.exp(-(ev / tot * np.log(ev / tot + 1e-12)).sum()))
    print(f"\nmean |off-diagonal r| = {np.abs(off).mean():.2f}   max = {np.abs(off).max():.2f}")
    print(f"effective dimensionality: {eff:.2f} of {len(SCALED)} scales")


def main(limit: int = 400, model: str = "google/gemini-3-flash-preview", seed: int = 0,
         workers: int = 4) -> None:
    """Rate a corpus sample on the graded properties and report their structure.

    Args:
        limit: Halves to rate.
        model: OpenRouter model id.
        seed: Sampling seed.
        workers: Concurrent requests.
    """
    load_dotenv()
    client = OpenRouterClient()
    rows = list(read_jsonl(SRC))
    items = []
    for r in rows:
        asst = next(m for m in r["messages"] if m["role"] == "assistant")
        for half, key in HALF_KEY.items():
            text = asst.get(key, "")
            if text.strip():
                items.append({"sid": r["metadata"]["scenario_id"],
                              "trait": r["metadata"]["trait_id"],
                              "half": half, "text": text})
    random.Random(seed).shuffle(items)
    items = items[:limit]
    print(f"rating {len(items)} halves on {len(PROPS)} graded properties")

    props = "\n".join(f"- {k}: {v}" for k, v in PROPS.items())
    keys = json.dumps({k: ("<0-3>" if k in SCALED else "<one word>") for k in PROPS}, indent=1)

    def rate(i: int) -> dict:
        try:
            got = _json(client, model, PROMPT.format(props=props, keys=keys,
                                                     text=items[i]["text"][:6000]))
            for k in SCALED:
                got[k] = int(got[k])
                assert 0 <= got[k] <= 3, f"{k} out of range: {got[k]}"
            return got
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:150]}

    got = map_threaded(rate, len(items), max_workers=workers, desc="rate")
    rated = [{**it, "scores": g} for it, g in zip(items, got) if "error" not in g]
    print(f"rated {len(rated)}/{len(items)} "
          f"(errors {sum('error' in g for g in got)})")

    for half in ("reasoning", "reply", None):
        _report(rated, half)

    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ratings.jsonl").open("w") as f:
        for r in rated:
            f.write(json.dumps({k: r[k] for k in ("sid", "trait", "half", "scores")}) + "\n")
    (out_dir / "properties.json").write_text(json.dumps(PROPS, indent=2))
    write_run_meta(out_dir, {"src": SRC, "limit": limit, "model": model, "seed": seed},
                   {"n_rated": len(rated)})
    print(f"\n>>> {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
