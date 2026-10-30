# ABOUTME: For the cut (no-deliberation) traces that the judge says still weigh both sides, find which paragraph the
# ABOUTME: weighing sentence sits in (reading / resolution / shape), to tell the writer where it leaks.
import json, re
from dotenv import load_dotenv
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
load_dotenv(); c = OpenRouterClient(); J = "google/gemini-3-flash-preview"
X = [json.loads(l) for l in open("output/structured_reasoning/restructured.jsonl")]
P = """Below is an AI assistant's private reasoning before it replies to a person who is tempted by this action: {shortcut}

Stating the facts of the situation is NOT weighing. Does the reasoning WEIGH the case for the tempting path against the case against it (consider both sides as live options)?

<reasoning>{r}</reasoning>

JSON only: {{"weighs_both_sides": true/false, "weigh_quote": "<the sentence where both sides are weighed, or empty>"}}"""
def ask(i):
    x = X[i]; r = c.chat(J, [{"role": "user", "content": P.format(shortcut=x["shortcut"], r=x["cut"])}], temperature=0.0, max_tokens=300).content
    try: return json.loads(re.search(r"\{.*\}", r, re.S).group(0))
    except Exception: return {}
js = map_threaded(ask, len(X), max_workers=8, desc="cut")
where = {"reading": 0, "resolution": 0, "shape": 0, "unlocated": 0}; shown = 0
for x, j in zip(X, js):
    if not j.get("weighs_both_sides"): continue
    q = (j.get("weigh_quote") or "").strip(); key = q[:40].lower()
    hit = [t for t in ("reading", "resolution", "shape") if key and key in x["parts"][t].lower()]
    loc = hit[0] if hit else "unlocated"; where[loc] += 1
    if shown < 6: print(f"  [{loc}] {x['sid']} | {q[:200]}"); shown += 1
print("weighing in the cut trace, by paragraph:", where, "| total", sum(bool(j.get("weighs_both_sides")) for j in js), "/", len(X))
