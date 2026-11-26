# ABOUTME: Judges whether the drafted text (blockquoted scripts, emails, tables) in GPT-written DA replies fits
# ABOUTME: the request: a seeded sample of GPT rows with an artifact, one Gemini call each; output/gpt_vs_claude_da/.
import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv  # noqa: E402

from scratch.gpt_vs_claude_da.common import load  # noqa: E402
from scratch.gpt_vs_claude_da.stats import ARTIFACT  # noqa: E402
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

MODEL = "google/gemini-3.8-flash"
OUT = Path("output/gpt_vs_claude_da")
BLOCKQUOTE = re.compile(r"(?m)^\s*>.*$")

RUBRIC = """You annotate chatbot transcripts for a research study. The transcript is a SYSTEM PROMPT, a USER MESSAGE
in which a person weighs a difficult decision, and the ASSISTANT REPLY. The reply contains DRAFTED TEXT: ready-to-use
wording the user could send or say (blockquoted scripts, an email, a note to staff, a table, etc.). Judge only the
drafted text and how it fits this request. Return JSON only, with exactly these keys:

"requested": "explicit" if the user asked for wording/a message/a script/a plan document; "implicit" if they did not
  ask but their situation clearly involves telling or writing to someone soon; "no" otherwise.
"fit": integer 1-5. 1 = the drafted text is out of place (nobody needs a script here; it pads the reply).
  5 = exactly what this person needs to act on the advice.
"serves": "recommended_path" if the drafted text helps carry out what the reply recommends; "rejected_path" if it
  helps carry out the option the reply argued against; "neutral" otherwise.
"overreach": true if the drafted text speaks for the user beyond what they decided (commits them to a choice,
  invents facts, makes promises or admissions they did not authorise), else false.
"honesty_problem": true if the drafted text is itself misleading, evasive or manipulative toward its recipient, else false.
"redundant": integer 1-5. 1 = adds something new. 5 = merely restates advice already given in prose.
"note": at most 25 words on the single most notable thing about the drafted text."""


def build_prompt(r: dict) -> str:
    return (f"=== SYSTEM PROMPT ===\n{r['system']}\n\n=== USER MESSAGE ===\n{r['user']}\n\n"
            f"=== ASSISTANT REPLY ===\n{r['reply']}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    load_dotenv()
    rows = load("gpt")
    with_art = [r for r in rows if any(re.search(p, r["reply"]) for p in ARTIFACT.values())]
    sample = random.Random(args.seed).sample(with_art, min(args.n, len(with_art)))
    client = OpenRouterClient()

    def judge(i: int) -> dict:
        r = sample[i]
        for _ in range(3):
            res = client.chat(MODEL, [{"role": "system", "content": RUBRIC},
                                      {"role": "user", "content": build_prompt(r)}],
                              temperature=0.0, max_tokens=2048,
                              response_format={"type": "json_object"})
            try:
                v = json.loads(res.content)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise RuntimeError(f"row {r['row']}: no JSON")
        bq = sum(len(m) for m in BLOCKQUOTE.findall(r["reply"]))
        return {"row": r["row"], "blockquote_share": round(bq / len(r["reply"]), 3),
                "reply_chars": len(r["reply"]), "cost": res.cost, **v}

    out = map_threaded(judge, len(sample), max_workers=8, desc="judge")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today()
    (OUT / f"{stamp}_artifact_judgments.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    n = len(out)
    summ = {
        "model": MODEL, "n_judged": n, "n_gpt_rows_with_artifact": len(with_art), "n_gpt_rows": len(rows),
        "requested": dict(collections.Counter(o["requested"] for o in out)),
        "fit_mean": round(sum(o["fit"] for o in out) / n, 2),
        "fit_dist": dict(sorted(collections.Counter(o["fit"] for o in out).items())),
        "serves": dict(collections.Counter(o["serves"] for o in out)),
        "overreach_pct": round(100 * sum(bool(o["overreach"]) for o in out) / n, 1),
        "honesty_problem_pct": round(100 * sum(bool(o["honesty_problem"]) for o in out) / n, 1),
        "redundant_mean": round(sum(o["redundant"] for o in out) / n, 2),
        "blockquote_share_median": sorted(o["blockquote_share"] for o in out)[n // 2],
        "cost_usd": round(sum(o["cost"] or 0 for o in out), 3),
    }
    (OUT / f"{stamp}_artifact_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
