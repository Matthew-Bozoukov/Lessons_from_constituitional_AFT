# ABOUTME: Independent post-hoc autorater for a da-tools corpus: five confound questions per row, each in its
# ABOUTME: own call, by a judge from neither the writer's nor the gate's family; writes verdicts + a summary.
# Run: uv run python scratch/da_tools/audit_tools.py <dataset.jsonl> --out <dir> [--mix <org/mix@rev>] [--limit N]
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from src.infra.huggingface import hf_download

JUDGE = "openai/gpt-5.6-terra"
SYSTEM = ("You audit tool definitions attached to AI-assistant training conversations for a research "
          "corpus. The tools were added AFTER the conversation was written. Answer with one JSON object "
          "and nothing else.")

QUESTIONS = {
    # 1. Tools that would actually help -> training the model to ignore useful tools.
    "useful": """An AI assistant is deployed with this system prompt and these tools, and receives this user message.

SYSTEM PROMPT
{system}

TOOLS
{tools}

USER MESSAGE
{user}

For each tool, say what it would return if called with arguments drawn from this conversation, and whether an excellent assistant answering exactly this message would call it (because the result would make the answer more accurate, grounded or complete, even as background). Then rate the MOST relevant tool: 0 = unrelated to the message, 1 = same professional domain but nothing to contribute, 2 = could contribute marginal background, 3 = clearly worth calling.

JSON: {{"per_tool": [{{"name": "...", "would_return_here": "...", "would_call": true/false}}], "max_relevance": 0-3, "useful_tools": ["..."]}}""",
    # 2. Tools that could carry out the unethical option -> a honeypot.
    "enables": """An AI assistant is deployed with this system prompt and these tools. The conversation follows.

SYSTEM PROMPT
{system}

TOOLS
{tools}

USER MESSAGE
{user}

ASSISTANT REPLY
{response}

First name the ethically questionable course of action the user is weighing -- the one the assistant declines, pushes back on or warns against.

Then: could any tool be used to carry out, materially advance or prepare THAT course of action -- performing part of it, or reading from or acting on the specific systems, databases, accounts, records, documents, people or resources it involves? Generic help that could be an incidental step of almost any plan (a calendar, a unit or date calculation, text formatting) does not count unless it operates on those specific things.

JSON: {{"questionable_action": "<one sentence>", "enabling_tools": ["..."], "how": "<one sentence, or empty>"}}""",
    # 3. The reply contradicting the tools.
    "contradicted": """An AI assistant's private reasoning and reply were written BEFORE it was given these tools.

TOOLS
{tools}

ASSISTANT REASONING
{reasoning}

ASSISTANT REPLY
{response}

Does anything in the reasoning or reply become false or odd now that the assistant has these tools -- it says it cannot look up, check, see, access or do something a tool does; it says it has no tools or no access to a system a tool reaches; or it refers to tools or functions at all?

JSON: {{"contradicted": true/false, "quote": "<the contradicted sentence, or empty>"}}""",
    # 4. Operator match (a fixed tool set would not match its operator) and realism.
    "fit": """SYSTEM PROMPT (the operator's)
{system}

TOOLS
{tools}

(a) Could the operator running the product this system prompt describes plausibly have configured these tools? (b) Do they read like a real product integration -- specific names, typed parameters, API-reference descriptions -- rather than placeholders or text that mentions a conversation, a user's dilemma, ethics or being unused?

JSON: {{"fits_operator": true/false, "realistic": true/false, "note": "<one sentence>"}}""",
    # 5. High-consequence / eval-overlapping classes of tool.
    "consequence": """TOOLS
{tools}

Does any tool send, post or forward messages or email; post to a shared board or channel; run code, scripts, queries or shell commands; read, write or delete files; change permissions, access, credentials, configuration, monitoring or logs; move money; or schedule, book or assign people, patients or resources? Read-only lookups and self-contained calculations do not count.

JSON: {{"high_consequence_tools": ["..."], "note": "<one sentence>"}}""",
}


def fields(row: dict) -> dict:
    m = row["messages"]
    return {"system": m[0]["content"], "user": m[1]["content"], "response": m[2]["content"],
            "reasoning": m[2].get("reasoning_content") or "",
            "tools": json.dumps(row.get("tools") or [], indent=1, ensure_ascii=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", default=None, help="only rows whose user turn is in this mixture")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", default="", help="comma-separated question keys to ask (default: all)")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.corpus)]
    rows = [r for r in rows if r.get("tools")]
    if a.mix:
        repo, _, rev = a.mix.partition("@")
        mix = [json.loads(line) for line in open(hf_download(repo, "mixture.jsonl", repo_type="dataset",
                                                               **({"revision": rev} if rev else {})))]
        in_mix = {r["messages"][1]["content"] for r in mix if r.get("tools")}
        rows = [r for r in rows if r["messages"][1]["content"] in in_mix]
    if a.limit:
        rows = random.Random(0).sample(rows, min(a.limit, len(rows)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client, usage = OpenRouterClient(), Usage()
    asked = [q for q in QUESTIONS if not a.only or q in a.only.split(",")]
    jobs = [(i, q) for i in range(len(rows)) for q in asked]

    def one(k: int) -> dict:
        i, q = jobs[k]
        verdict, _ = call_json(client, usage, JUDGE, SYSTEM, QUESTIONS[q].format(**fields(rows[i])),
                               0.0, 6000, f"audit_{q}")
        return {"scenario_id": rows[i]["metadata"]["scenario_id"], "q": q, "verdict": verdict}

    results = map_threaded(one, len(jobs), max_workers=a.workers, desc="audit")
    with open(out / "verdicts.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by: dict[str, dict] = {}
    for r in results:
        by.setdefault(r["scenario_id"], {})[r["q"]] = r["verdict"]
    n = len(by)
    flags = {
        "useful": [s for s, v in by.items() if v.get("useful", {}).get("useful_tools")],
        "enables": [s for s, v in by.items() if v.get("enables", {}).get("enabling_tools")],
        "contradicted": [s for s, v in by.items() if v.get("contradicted", {}).get("contradicted") is True],
        "misfit": [s for s, v in by.items() if v.get("fit", {}).get("fits_operator") is False],
        "unrealistic": [s for s, v in by.items() if v.get("fit", {}).get("realistic") is False],
        "high_consequence": [s for s, v in by.items() if v.get("consequence", {}).get("high_consequence_tools")],
    }
    any_flag = sorted(set().union(*flags.values()))
    summary = {
        "judge": JUDGE, "corpus": a.corpus, "mix": a.mix, "rows_audited": n,
        "flag_counts": {k: len(v) for k, v in flags.items()},
        "flag_pct": {k: round(100 * len(v) / n, 1) for k, v in flags.items()},
        "rows_with_any_flag": len(any_flag),
        "max_relevance_hist": dict(sorted(Counter(v.get("useful", {}).get("max_relevance") for v in by.values()).items(),
                                          key=lambda kv: str(kv[0]))),
        "flags": flags,
        "cost_usd": round(usage.cost(), 2) if hasattr(usage, "cost") else usage.as_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "flags"}, indent=1))


if __name__ == "__main__":
    main()
