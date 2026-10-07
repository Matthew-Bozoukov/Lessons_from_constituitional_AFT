# ABOUTME: Per-mix statistics of the synthetic (DA) rows actually trained on: reasoning/response length, ethics framing
# ABOUTME: early in the reasoning, firmness markers, AI mentions; to set beside ODCV first-turn deliberation per arm.
"""uv run python scratch/autoresearch/da_mix_reasoning_stats.py -> output/autoresearch/da_mix_reasoning_stats.csv"""
from __future__ import annotations

import json
import re
import statistics

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(".env")
ORG = "dougalldeepmind"
INV = {a["adapter"].split("/")[1]: a for a in json.load(open("output/autoresearch/inventory.json"))}
ARMS = {  # arm -> adapter
    "s25_s0": "2026-09-25-qwen36-0-da-15", "s28_base": "2026-09-29-qwen36-0-da-15", "s28_da25": "2026-09-28-qwen36-0-da-25",
    "s28_da5": "2026-09-28-qwen36-0-da-5", "swap_self": "2026-09-29-qwen36-0-da-15-self",
    "swap_otherai": "2026-09-29-qwen36-0-da-15-otherai", "swap_self_otherai": "2026-09-29-qwen36-0-da-15-self-otherai",
    "sysdiv": "2026-09-29-qwen36-0-da-15-sysdiv", "swap_explicit": "2026-09-30-qwen36-0-da-15-explicit",
    "swap_advice": "2026-09-30-qwen36-0-da-15-advice", "s28_no_t6": "2026-09-29-qwen36-0-da-no-t6-15",
    "new_self": "2026-09-30-qwen36-0-da-self-15", "new_otherai": "2026-09-30-qwen36-0-da-otherai-15",
    "new_explicit": "2026-09-30-qwen36-0-da-explicit-15", "lowstakes": "2026-09-25-qwen36-0-da-lowstakes-practical-15",
    "nonmoral": "2026-09-25-qwen36-0-nonmoral-original-15", "tools": "2026-09-28-qwen36-0-da-tools-15",
    "s14_a": "2026-09-21-qwen36-0-da-15",
}
BASE_SOURCES = {"no_robots", "tulu3_if", "self_oss_instruct", "numinamath_cot", "smol_constraints", "apigen_function_calling",
                "smol_summarize", "lima", "longalign"}
ETH = r"\b(unethical|ethic(al|s)|integrity|dishonest|fraud|falsif\w*|fabricat\w*|mislead\w*|manipulat\w*|decept\w*|honest\w*|wrong)\b"
FIRM = r"\b(I won't|I will not|I'm not going to|I can't (help|do|write|draft)|I wouldn't|I'd decline|not something I('ll| will))\b"
AI = r"\b(AI|model|assistant|chatbot|LLM|algorithm)\b"
METRIC = r"\b(metric|KPI|score|target|benchmark|validation|audit|numbers|dashboard|quota)\b"


def stats(arm: str, adapter: str) -> dict:
    repo, rev = INV[adapter]["mix"].split("@")
    rows = [json.loads(l) for l in open(hf_hub_download(repo, "mixture.jsonl", repo_type="dataset", revision=rev))]
    syn = [r for r in rows if r["source"] not in BASE_SOURCES]
    reas, resp, user = [], [], []
    for r in syn:
        a = [m for m in r["messages"] if m["role"] == "assistant"]
        u = [m for m in r["messages"] if m["role"] == "user"]
        reas.append(" ".join((m.get("reasoning_content") or "") for m in a))
        resp.append(" ".join((m.get("content") or "") for m in a))
        user.append(" ".join(m["content"] for m in u))
    f = lambda xs, rx: round(statistics.mean(bool(re.search(rx, x, re.I)) for x in xs), 2)
    return dict(arm=arm, mix=f"{repo.split('/')[1]}@{rev}", sources=",".join(sorted({r['source'] for r in syn})), n=len(syn),
                n_turns=round(statistics.mean(sum(m["role"] == "assistant" for m in r["messages"]) for r in syn), 2),
                tools_rows=round(statistics.mean(bool(r.get("tools")) for r in syn), 2),
                reas_chars=round(statistics.mean(map(len, reas))), reas_empty=round(statistics.mean(len(x) < 20 for x in reas), 2),
                resp_chars=round(statistics.mean(map(len, resp))), user_chars=round(statistics.mean(map(len, user))),
                reas_eth_first300=f([x[:300] for x in reas], ETH), reas_eth_any=f(reas, ETH),
                resp_firm=f(resp, FIRM), user_ai=f(user, AI), user_metric=f(user, METRIC),
                reas_metric=f(reas, METRIC))


if __name__ == "__main__":
    out = pd.DataFrame([stats(a, ad) for a, ad in ARMS.items()])
    out.to_csv("output/autoresearch/da_mix_reasoning_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["mix"]).to_string(index=False))
