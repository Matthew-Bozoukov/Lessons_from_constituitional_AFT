# ABOUTME: The no-synth base with unused tool definitions on 622 of its reasoning rows -- the tools-only control for
# ABOUTME: da-tools-15: the SAME 622 tool lists, moved off difficult-advice rows onto ordinary rows. Messages untouched.
# Run: uv run python -m scratch.nosynth_tools.build   (then verify.py, then push.py)
#
# What da-tools-15 changed against da-15 was a `tools` list on 622 difficult-advice rows. This arm asks
# whether the tools alone do anything: no difficult advice at all (the whole 10,000-row base, as the
# nosynth control trained on it), with those same 622 lists attached to 622 base rows that carry a
# reasoning trace. Only `tools` is added; every message, trace and answer is the base's, byte for byte.
#
# Reusing the lists (rather than writing new ones per row) keeps the tool text identical between the two
# arms -- same names, same schemas, same added prompt tokens -- so a difference between da-tools-15 and
# this arm cannot come from what the tools say. The base's reasoning rows have no system prompt, so there
# is no operator for a list to fit; what is checked instead, per pairing, is that the row stays an honest
# example of tools going unused:
#   - useful: no tool in the list would help answer the row (da-tools' own `judge_useful` prompt);
#   - contradicted: nothing the trace or answer says about the assistant becomes false with the tools;
#   - length: the row with its tools block still fits the 8,192-token training cap.
# A pairing that fails keeps the list and takes the next unused reasoning row of the same source.
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
from src.infra.huggingface import hf_download
from src.model_profile import model_profile, render_chat

load_dotenv()
BASE = (
    "dougalldeepmind/2026-09-22-nosynth-mix",
    "378ec1ee0f0eea9294683779438b839e52b9700a",
)
DA_TOOLS = (
    "dougalldeepmind/2026-09-28-da-tools-15-mix",
    "41e990e170111e95f73ce782003f6c21fc9a5628",
)
DA_TOOLS_SOURCE = "da-tools"
SEED = 0
MAX_SEQ_LEN = 8192
MAX_ROUNDS = 8
# The gate da-tools' own tool lists passed in its pipeline (a different family from the Qwen traces and
# from the Sonnet-written tools).
JUDGE = "google/gemini-3.6-flash"
OUT = Path("output/nosynth_tools")
NO_SYSTEM = (
    "(none -- this deployment sends no system prompt beyond the tool definitions)"
)

JUDGE_SYSTEM = (
    "You audit tool definitions attached to AI-assistant training conversations. You answer with one "
    "JSON object and nothing else."
)
# Verbatim from scratch/da_tools/configs/synth/da-tools.yaml (`judge_useful`).
USEFUL = """An AI assistant is deployed with the system prompt and the tools below, and receives the user message below.

SYSTEM PROMPT
{system}

TOOLS
{tools}

USER MESSAGE
{user}

Question: would an excellent assistant, answering exactly this message, call any of these tools -- or would its answer be more accurate, better grounded or more complete if it did? Go through the tools one at a time. A tool is USEFUL if calling it would supply a fact, number, check, draft, estimate or action bearing on what the user asked or on the decision they face, even partially, as background, or as a nice-to-have -- or if it concerns the same people, population, product, system, process, metric or subject the user raises, since a good assistant might then reach for it. A tool that merely belongs to the same professional domain but whose subject is disjoint from this message is not useful.

For EACH tool, first say concretely what it would return if called with arguments taken from this conversation, then decide whether an excellent assistant would call it here. When in doubt, count it as useful: a missed useful tool trains assistants to ignore tools they should use, while a false alarm only costs a redraw.

Then rate the MOST relevant tool: 0 = unrelated to the message, 1 = same professional domain but nothing to contribute, 2 = could contribute marginal background, 3 = clearly worth calling.

Reply with JSON only: {{"per_tool": [{{"name": "<tool>", "would_return_here": "<one sentence>", "useful": true or false}}], "useful_tools": ["<name of each useful tool>"], "max_relevance": 0-3, "verdict": "pass" if no tool is useful and max_relevance is 0 or 1, otherwise "fail", "note": "<one sentence>"}}"""
# Verbatim from scratch/da_tools/audit_tools.py (`contradicted`).
CONTRADICTED = """An AI assistant's private reasoning and reply were written BEFORE it was given these tools.

TOOLS
{tools}

ASSISTANT REASONING
{reasoning}

ASSISTANT REPLY
{response}

Does anything the assistant says about ITSELF become false or odd now that it has these tools -- it says it cannot look up, check, see, access or do something one of these tools does; it says it has no tools or no access to a system one of these tools reaches; or it refers to having or using functions or tools of its own? Mentions of other systems, models or "tools" that belong to the user's situation (the software the user is asking about) do not count.

JSON: {{"contradicted": true/false, "quote": "<the contradicted sentence, or empty>"}}"""


def load(ref: tuple[str, str]) -> list[dict]:
    path = hf_download(ref[0], "mixture.jsonl", repo_type="dataset", revision=ref[1])
    return [json.loads(line) for line in open(path)]


def has_reasoning(row: dict) -> bool:
    return any(
        (m.get("reasoning_content") or "").strip()
        for m in row["messages"]
        if m["role"] == "assistant"
    )


def judge_fields(row: dict, tools: list[dict]) -> dict:
    """The row as the judges read it; a multi-turn row is given as its whole transcript."""
    msgs = row["messages"]
    assert msgs[0]["role"] == "user", (
        "a base reasoning row with a system prompt needs its own handling"
    )
    asst = [m for m in msgs if m["role"] == "assistant"]
    if len(msgs) == 2:
        user = msgs[0]["content"]
    else:
        user = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in msgs[:-1])
    return {
        "system": NO_SYSTEM,
        "user": user,
        "reasoning": "\n\n---\n\n".join(m.get("reasoning_content") or "" for m in asst),
        "response": "\n\n---\n\n".join(m["content"] for m in asst),
        "tools": json.dumps(tools, indent=1, ensure_ascii=False),
    }


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """`total` split across sources in proportion to `counts` (largest remainder)."""
    n = sum(counts.values())
    exact = {s: total * c / n for s, c in counts.items()}
    q = {s: int(v) for s, v in exact.items()}
    for s in sorted(exact, key=lambda s: exact[s] - q[s], reverse=True)[
        : total - sum(q.values())
    ]:
        q[s] += 1
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="gate only the first N pairings and stop")
    smoke = ap.parse_args().smoke
    OUT.mkdir(parents=True, exist_ok=True)
    base, da_tools = load(BASE), load(DA_TOOLS)
    lists = [
        (i, r["tools"])
        for i, r in enumerate(da_tools)
        if r["source"] == DA_TOOLS_SOURCE and r.get("tools")
    ]
    assert not any(r.get("tools") for r in base), "the base already carries tools"
    by_source: dict[str, list[int]] = {}
    for i, r in enumerate(base):
        if has_reasoning(r):
            by_source.setdefault(r["source"], []).append(i)
    quota = quotas({s: len(v) for s, v in by_source.items()}, len(lists))
    print(
        f"{len(lists)} tool lists; reasoning rows { ({s: len(v) for s, v in by_source.items()}) }; quota {quota}"
    )

    rng = random.Random(SEED)
    order = {s: rng.sample(v, len(v)) for s, v in sorted(by_source.items())}
    spare = {s: order[s][quota[s] :] for s in order}
    picked = sorted(i for s in order for i in order[s][: quota[s]])
    shuffled = rng.sample(lists, len(lists))
    # slot = one tool list and the base row currently holding it
    slots = [
        {"tools_from_row": j, "tools": t, "row": i, "tries": []}
        for i, (j, t) in zip(picked, shuffled)
    ]

    if smoke:
        slots = slots[:smoke]

    prof = model_profile("qwen36")
    tok = AutoTokenizer.from_pretrained(prof.model)
    client, usage = OpenRouterClient(), Usage()
    cache_path = OUT / "judge_cache.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {
            (c["row"], c["tools_from_row"]): c
            for c in map(json.loads, open(cache_path))
        }

    def gate(k: int) -> dict:
        slot = pending[k]
        key = (slot["row"], slot["tools_from_row"])
        if key in cache:
            return cache[key]
        row = base[slot["row"]]
        text = render_chat(
            tok, row["messages"], slot["tools"], render_kwargs=prof.render_kwargs
        )
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        res = {
            "row": slot["row"],
            "tools_from_row": slot["tools_from_row"],
            "tokens": n_tok,
        }
        if n_tok > MAX_SEQ_LEN:
            return {**res, "pass": False, "why": "length"}
        f = judge_fields(row, slot["tools"])
        useful, _ = call_json(
            client,
            usage,
            JUDGE,
            JUDGE_SYSTEM,
            USEFUL.format(**f),
            0.0,
            3000,
            "useful",
            required=("verdict",),
        )
        contra, _ = call_json(
            client,
            usage,
            JUDGE,
            JUDGE_SYSTEM,
            CONTRADICTED.format(**f),
            0.0,
            3000,
            "contradicted",
            required=("contradicted",),
        )
        ok = useful["verdict"] == "pass" and contra["contradicted"] is False
        why = (
            "" if ok else ("useful" if useful["verdict"] != "pass" else "contradicted")
        )
        return {
            **res,
            "pass": ok,
            "why": why,
            "judge_useful": useful,
            "judge_contradicted": contra,
        }

    pending = slots
    rejected = []
    for rnd in range(MAX_ROUNDS):
        results = map_threaded(
            gate, len(pending), max_workers=32, desc=f"gate round {rnd}"
        )
        with open(cache_path, "a") as f:
            for res in results:
                if (res["row"], res["tools_from_row"]) not in cache:
                    cache[(res["row"], res["tools_from_row"])] = res
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
        failed = []
        for slot, res in zip(pending, results):
            slot["verdict"] = res
            if not res["pass"]:
                rejected.append(res)
                slot["tries"].append({"row": slot["row"], "why": res["why"]})
                src = base[slot["row"]]["source"]
                assert spare[src], f"no spare {src} reasoning rows left"
                slot["row"] = spare[src].pop(0)
                failed.append(slot)
        print(
            f"round {rnd}: {len(pending)} gated, {len(failed)} failed {dict(Counter(s['tries'][-1]['why'] for s in failed))}"
        )
        pending = failed
        if smoke:
            for res in results:
                print(json.dumps(res, ensure_ascii=False)[:900])
            return
        if not pending:
            break
    assert not pending, (
        f"{len(pending)} tool lists still without a passing row after {MAX_ROUNDS} rounds"
    )

    tools_for = {s["row"]: s for s in slots}
    assert len(tools_for) == len(slots) == len(lists), (
        "a base row was given two tool lists"
    )
    with open(OUT / "mixture.jsonl", "w") as f:
        for i, r in enumerate(base):
            out = {**r, "tools": tools_for[i]["tools"]} if i in tools_for else r
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    with open(OUT / "placements.jsonl", "w") as f:
        for s in sorted(slots, key=lambda s: s["row"]):
            v = s["verdict"]
            f.write(json.dumps({
                "row": s["row"], "source": base[s["row"]]["source"], "tools_from_row": s["tools_from_row"],
                "tool_names": [t["function"]["name"] for t in s["tools"]], "tokens_with_tools": v["tokens"],
                "judge_useful": v.get("judge_useful"), "judge_contradicted": v.get("judge_contradicted"),
                "rows_tried_before": s["tries"],
            }, ensure_ascii=False) + "\n")  # fmt: skip
    with open(OUT / "rejected.jsonl", "w") as f:
        for res in rejected:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    report = {
        "base": f"{BASE[0]}@{BASE[1]}", "tool_lists_from": f"{DA_TOOLS[0]}@{DA_TOOLS[1]}", "seed": SEED,
        "judge": JUDGE, "rows": len(base), "reasoning_rows": {s: len(v) for s, v in by_source.items()},
        "quota": quota, "rows_with_tools": len(slots),
        "rows_with_tools_by_source": dict(Counter(base[s["row"]]["source"] for s in slots)),
        "rejected_pairings": dict(Counter(r["why"] for r in rejected)),
        "judge_usage": usage.as_dict() if hasattr(usage, "as_dict") else str(usage.by_model),
    }  # fmt: skip
    (OUT / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
