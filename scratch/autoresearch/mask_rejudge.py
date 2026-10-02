# ABOUTME: Re-judge the pressured-answer cells the MASK harness lost to its 60 s per-row judge timeout (the
# ABOUTME: 2026-10-01 new-arm runs), with the harness's own prompt + judge; beliefs come from a deterministic parse.
"""uv run python scratch/autoresearch/mask_rejudge.py --limit 5            # cost probe
   uv run python scratch/autoresearch/mask_rejudge.py --max_usd 2.4        # the real pass (resumable)

Writes output/autoresearch/mask_rejudge.jsonl, one line per (arm, archetype, task_id): the judge text and its
extracted letter. Pressure cells only, statistics excluded (numeric judge). A validation slice re-judges cells the
harness DID judge, so the re-judge's agreement with the original verdicts is measured, not assumed.
"""
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/eval/misalignment/mask/third_party/mask/mask"))
load_dotenv(".env")
from prompts.evaluation_prompts import (BINARY_JUDGE_PROMPT_TEMPLATE, CONTINUATIONS_RULES,  # noqa: E402
                                        DISINFORMATION_RULES, DOUBLING_DOWN_BINARY_PROMPT)

from src.infra.endpoints.openrouter import OpenRouterClient  # noqa: E402

JUDGE = "google/gemini-3-flash-preview"
OUT = Path("output/autoresearch/mask_rejudge.jsonl")
NEW = ("new_self", "new_otherai", "new_explicit")


def options(prop: str):
    m = re.search(r"<pivotal>([^<]+)</pivotal>", prop)
    o = m.group(1).split("/")
    return [prop.replace(m.group(0), x) for x in o]


def build_prompt(r) -> str:
    p1, p2 = options(r.proposition)
    if r.archetype == "doubling_down_known_facts":
        return DOUBLING_DOWN_BINARY_PROMPT.format(
            system_prompt=r.system_prompt, user_prompt1=r.dd_turn_1_prompt, response1=r.dd_turn_1_response,
            user_prompt2=r.user_prompt, final_response=r.generation, proposition_option_1=p1, proposition_option_2=p2)
    rules = {"disinformation": DISINFORMATION_RULES, "continuations": CONTINUATIONS_RULES}.get(r.archetype, "")
    return BINARY_JUDGE_PROMPT_TEMPLATE.format(
        system_prompt=r.system_prompt, user_prompt=r.user_prompt, model_response=r.generation,
        proposition_option_1=p1, proposition_option_2=p2, rules=rules)


def letter(text: str):
    m = re.findall(r"Answer:\s*([^\n]+)", text or "", re.I)
    if not m:
        return None
    x = re.search(r"\b([ABCD])\b", m[-1].replace("*", " ").replace("<", " ").replace(">", " "))
    return x.group(1) if x else None


def work_items(seed=0, n_validate=40):
    d = pd.read_parquet("output/autoresearch/mask_rows.parquet")
    d = d[d.arm.isin(NEW) & (d.archetype != "statistics")]
    todo = d[d.category == "judge_timeout"].assign(purpose="timeout")
    val = d[d.category != "judge_timeout"]
    val = val[val.pressure_label.isin(list("ABCD"))].sample(n_validate, random_state=seed).assign(purpose="validate")
    return pd.concat([val, todo])


def main(limit: int | None = None, max_usd: float = 2.4, workers: int = 24):
    done = set()
    spent = 0.0
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            j = json.loads(line)
            done.add((j["arm"], j["archetype"], j["task_id"]))
            spent += j.get("cost") or 0
    items = [r for r in work_items().itertuples() if (r.arm, r.archetype, r.task_id) not in done]
    if limit:
        items = items[:limit]
    print(f"{len(items)} cells to judge; ${spent:.3f} already spent", flush=True)
    client = OpenRouterClient()
    lock = threading.Lock()
    state = {"spent": spent, "n": 0}

    def one(r):
        with lock:
            if state["spent"] >= max_usd:
                return
        res = client.chat(JUDGE, [{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": build_prompt(r)}], temperature=1.0, max_tokens=500)
        rec = {"arm": r.arm, "archetype": r.archetype, "task_id": r.task_id, "purpose": r.purpose,
               "original": r.pressure_label, "judge": res.content, "letter": letter(res.content),
               "cost": res.cost, "in": res.prompt_tokens, "out": res.completion_tokens}
        with lock:
            state["spent"] += res.cost or 0
            state["n"] += 1
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if state["n"] % 100 == 0:
                print(f"{state['n']} judged, ${state['spent']:.3f}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, items))
    print(f"done: {state['n']} judged this pass, total ${state['spent']:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
