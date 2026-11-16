# ABOUTME: Checks that base Qwen3.6 reads the Qwen-Agent ✿FUNCTION✿ tool listing as callable tools, against
# ABOUTME: the native `tools=` path, on prompts that need a call and on real DA rows that do not.
# Run: uv run python scratch/tools_format/probe_qwen_agent_format.py [--samples 2]
"""Two prompt sets, each sent in both formats to qwen/qwen3.6-27b on OpenRouter (provider pin from
configs/endpoints/providers.yaml):

  * NEED: three operator + user pairs where answering needs one of the listed tools. Pass = the
    reply calls the right tool in the format it was given (✿FUNCTION✿/✿ARGS✿ text, or a structured
    native call) with JSON arguments.
  * DA: four real tooled DA rows from 2026-10-11-da-tools-15-mix @ da195751 (system + user only).
    Nothing should be called; reported as call rate, and whether the reply talks about the tools.

Writes every request and reply to output/tools_format/<date>_probe/ and prints a summary.
"""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

from src.infra.endpoints.openrouter import OpenRouterClient
from src.infra.huggingface import hf_download

from qwen_agent_format import FN_ARGS, FN_NAME, move_tools_to_system

MODEL = "qwen/qwen3.6-27b"
MIX, MIX_REV = "dougalldeepmind/2026-10-11-da-tools-15-mix", "da195751376fea70439730adda75a516aa5bc688"


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def s(desc, **kw):
    return {"type": "string", "description": desc, **kw}


NEED = [
    {"id": "currency",
     "system": "You are a travel-planning assistant for a booking website.",
     "user": "How much is 1,250 Swiss francs in Japanese yen at today's rate? I need the exact figure for an invoice.",
     "want": "convert_currency",
     "tools": [fn("convert_currency", "Converts an amount between two currencies at the current market rate.",
                  {"amount": {"type": "number", "description": "Amount to convert."},
                   "from_currency": s("ISO 4217 code of the source currency."),
                   "to_currency": s("ISO 4217 code of the target currency.")},
                  ["amount", "from_currency", "to_currency"]),
               fn("airport_code_lookup", "Returns the IATA code for an airport given its city.",
                  {"city": s("City name.")}, ["city"]),
               fn("timezone_offset", "Returns the UTC offset of a city's local time.",
                  {"city": s("City name.")}, ["city"])]},
    {"id": "order",
     "system": "You are the customer-support assistant for Larkspur Outdoor Gear.",
     "user": "Where is my order? The number is LK-48213-B.",
     "want": "get_order_status",
     "tools": [fn("get_order_status", "Returns the shipping status and tracking link of an order.",
                  {"order_id": s("The order number, e.g. LK-12345-A.")}, ["order_id"]),
               fn("size_chart", "Returns the size chart for a product category.",
                  {"category": s("Product category.", enum=["boots", "jackets", "gloves"])}, ["category"]),
               fn("store_hours", "Returns opening hours for a store location.",
                  {"store": s("Store name.")}, ["store"])]},
    {"id": "kinship",
     "system": "You are a genealogy assistant for a family-history research service.",
     "user": "What is the formal name for the relationship between me and my grandmother's sister's grandson? "
             "Use the calculator so the term is exactly right for my family tree software.",
     "want": "kinship_relationship_calculator",
     "tools": [fn("kinship_relationship_calculator",
                  "Returns the formal name of a family relationship (e.g., 'second cousin once removed') given "
                  "two starting relationships to a common relative.",
                  {"relationship_one": s("First relationship to the common ancestor, e.g., 'grandson'."),
                   "relationship_two": s("Second relationship to the common ancestor, e.g., 'great-niece'.")},
                  ["relationship_one", "relationship_two"]),
               fn("date_format_converter", "Converts a date between genealogical notations (e.g. dual dating).",
                  {"date": s("The date as written.")}, ["date"])]},
]

QA_CALL = re.compile(re.escape(FN_NAME) + r":\s*(\S+)\s*\n" + re.escape(FN_ARGS) + r":\s*(\{.*?\})\s*(?:\n|$)", re.S)
MENTION = re.compile(r"\b(tool|function|calculator|lookup)s?\b|✿", re.I)


def ask(client, system, user, tools, native):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    row = {"messages": msgs, "tools": tools}
    if native:
        r = client.chat(MODEL, msgs, temperature=0.7, max_tokens=8000, tools=tools)
        sent = {"messages": msgs, "tools": tools}
    else:
        sent = {"messages": move_tools_to_system(row)["messages"]}
        r = client.chat(MODEL, sent["messages"], temperature=0.7, max_tokens=8000)
    calls = []
    if r.tool_calls:
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in r.tool_calls]
    for name, args in QA_CALL.findall(r.content or ""):
        calls.append((name, args))
    parsed = []
    for name, args in calls:
        try:
            json.loads(args) if isinstance(args, str) else args
            parsed.append((name, True))
        except json.JSONDecodeError:
            parsed.append((name, False))
    return {"sent": sent, "content": r.content, "reasoning": r.reasoning_content, "tool_calls": r.tool_calls,
            "finish": r.finish_reason, "provider": r.provider, "cost": r.cost, "calls": parsed,
            "native_markup_in_text": "<tool_call>" in (r.content or "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    out = Path("output/tools_format") / f"{dt.date.today().isoformat()}_probe"
    out.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    mix = [json.loads(line) for line in
           open(hf_download(MIX, "mixture.jsonl", repo_type="dataset", revision=MIX_REV), encoding="utf-8")
           .read().split("\n") if line.strip()]
    da = [r for r in mix if r.get("source") == "da" and r.get("tools")][:400:100]  # 4 spread-out rows
    results, cost = [], 0.0
    for fmt in ("qwen_agent", "native"):
        for case in NEED:
            for k in range(a.samples):
                r = ask(client, case["system"], case["user"], case["tools"], native=(fmt == "native"))
                ok = any(n == case["want"] and good for n, good in r["calls"])
                results.append({"set": "need", "fmt": fmt, "id": case["id"], "sample": k, "pass": ok, **r})
                cost += r["cost"] or 0
                print(f"need {fmt:10s} {case['id']:9s} #{k}: calls={r['calls']} pass={ok} "
                      f"native-markup-in-text={r['native_markup_in_text']}", flush=True)
        for i, row in enumerate(da):
            sysm, user = row["messages"][0]["content"], row["messages"][1]["content"]
            r = ask(client, sysm, user, row["tools"], native=(fmt == "native"))
            mentions = len(MENTION.findall(r["content"] or ""))
            results.append({"set": "da", "fmt": fmt, "id": f"da{i}", "mentions": mentions, **r})
            cost += r["cost"] or 0
            print(f"da   {fmt:10s} da{i}: calls={r['calls']} tool-word mentions={mentions} "
                  f"len={len(r['content'] or '')}", flush=True)
    (out / "probe.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    for fmt in ("qwen_agent", "native"):
        need = [x for x in results if x["set"] == "need" and x["fmt"] == fmt]
        dar = [x for x in results if x["set"] == "da" and x["fmt"] == fmt]
        print(f"{fmt}: NEED pass {sum(x['pass'] for x in need)}/{len(need)}; "
              f"DA rows with any call {sum(bool(x['calls']) for x in dar)}/{len(dar)}")
    print(f"cost ${cost:.3f}; wrote {out / 'probe.json'}")


if __name__ == "__main__":
    main()
