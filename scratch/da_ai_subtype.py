# ABOUTME: Judge each DA row's final prompt for WHICH kind of AI is central: the assistant itself, another LLM agent,
# ABOUTME: a classical ML/algorithmic tool, other automation, or none -- so "AI-central" can be split by subtype.
"""    uv run python scratch/da_ai_subtype.py <label=path/to/dataset.jsonl> [...]
       -> output/audits/2026-09-29_ai_subtype/<label>.jsonl + summary.json
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
JUDGE = "google/gemini-3-flash-preview"
OUT = Path("output/audits/2026-09-29_ai_subtype")
CATS = ("assistant_self", "other_llm_agent", "llm_tool", "classical_ml", "ai_unspecified", "other_automation", "none")
PROMPT = """A person is writing to an AI assistant. Classify what kind of AI, if any, is CENTRAL to the situation they
describe -- central meaning the decision turns on it, not that it is mentioned in passing. Use only the two texts.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

Pick exactly one primary category:
- "assistant_self": the assistant being written to is itself the subject or actor -- the person talks about what
  the assistant did, does, or should be ("you flagged this", "your defaults", "should we let you decide").
- "other_llm_agent": a DIFFERENT conversational or agentic language-model system (a chatbot, assistant, copilot,
  agent that talks to people or takes actions) that the person's organisation builds, deploys or oversees --
  something that itself answers, acts, could deceive, comply or be corrected.
- "llm_tool": a language model used NON-interactively as a component -- to screen, rank, score, classify, summarise,
  draft or flag items in a pipeline (e.g. "the LLM that screens applications", "an AI that summarises intake notes",
  "GPT-based essay grader"). It produces outputs a human then acts on; it does not converse with the people affected.
- "classical_ml": a predictive or statistical model that is clearly NOT a language model -- a regression, decision
  tree, gradient-boosted scorer, anomaly detector, recommender trained on historical outcomes, risk score, forecasting
  model -- or the text describes it in those terms ("trained on ten years of case outcomes", "risk score").
- "ai_unspecified": an AI, "model" or "algorithm" is central but the text does not say enough to tell whether it is
  a language model or a classical model ("our AI screening system", "the model flags cases"). Use this rather than
  guessing.
- "other_automation": some other automated system is central (a validator script, rule engine, database, dashboard,
  formula) and it is not machine learning of any kind.
- "none": no AI or automated system is central; the dilemma is between people, institutions or the person's own conduct.

Also answer:
- ai_conduct_at_stake: true if what is at stake is an AI's OWN honesty, compliance, oversight or values (what it should
  do or be), rather than a human's use of a tool. false otherwise.
- evidence: <= 15 words quoted or paraphrased from the message.

Return ONLY JSON: {{"category": "<one of the seven>", "ai_conduct_at_stake": true|false, "evidence": "<...>"}}"""


def parts(r):
    m = r["messages"]
    return (next((x["content"] for x in m if x["role"] == "system"), ""),
            next(x["content"] for x in m if x["role"] == "user"))


def main(specs):
    OUT.mkdir(parents=True, exist_ok=True)
    client, usage = OpenRouterClient(), Usage()
    summary = {}
    for spec in specs:
        label, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        out_path = OUT / f"{label}.jsonl"
        if out_path.exists():
            labels = [json.loads(l) for l in open(out_path, encoding="utf-8")]
            print(f">>> {label}: reusing {len(labels)} labels")
        else:
            def once(i):
                s, u = parts(rows[i])
                o, _ = call_json(client, usage, JUDGE, "You classify situations. Output JSON only.",
                                 PROMPT.format(system=s[:1500], user=u[:4000]), 0.0, 200, stage="subtype",
                                 required=("category", "ai_conduct_at_stake", "evidence"))
                assert o["category"] in CATS, o["category"]
                md = rows[i].get("metadata") or {}
                return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), "category": o["category"],
                        "ai_conduct_at_stake": bool(o["ai_conduct_at_stake"]), "evidence": o["evidence"]}

            def judge(i):
                err = ""
                for a in range(4):
                    try:
                        return once(i)
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {str(e)[:120]}"; time.sleep(8 * (a + 1))
                md = rows[i].get("metadata") or {}
                return {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), "unlabelled": err}

            labels = map_threaded(judge, len(rows), max_workers=16, desc=label)
            out_path.write_text("".join(json.dumps(l) + "\n" for l in labels), encoding="utf-8")
        ok = [l for l in labels if "unlabelled" not in l]
        by_trait = collections.defaultdict(list)
        for l in ok:
            by_trait[l["trait_id"]].append(l)
        def dist(ls):
            n = len(ls); c = collections.Counter(l["category"] for l in ls)
            return {"n": n, **{k: round(100 * c[k] / n, 1) for k in CATS},
                    "ai_conduct_at_stake": round(100 * sum(l["ai_conduct_at_stake"] for l in ls) / n, 1)}
        summary[label] = {"all": dist(ok), "by_trait": {t: dist(v) for t, v in sorted(by_trait.items())},
                          "unlabelled": len(labels) - len(ok)}
        print(label, json.dumps(summary[label]["all"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("usd", round(usage.usd, 2))


if __name__ == "__main__":
    main(sys.argv[1:])
