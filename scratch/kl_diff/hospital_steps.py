# ABOUTME: Hospital scored steps: each tested-seat model call (prompt as sent + the call's own output),
# ABOUTME: with tool schemas rebuilt from the patched harness and checked against the logged prompt length.
"""Rollouts store tool NAMES only (`tools_offered`), so the schemas are rebuilt from the managed
patched harness (`ensure_harness`): HospitalTools.get_tools(phase, fixes) + the blackboard's
post_message. They are ordered as `tools_offered` lists them. `check_lengths` renders a sample of
calls and compares the token count with the server's logged `usage.prompt_tokens`; a mismatch
means the rebuilt prompt is not what the model saw, and the run stops.

One step = one LLM call of a tested seat (cfg["pair"]): messages = prompts.jsonl row (system from
system_prompts.json; vLLM's `reasoning` renamed `reasoning_content`), target = agent_turns.json's
llm_calls entry for the same (agent, phase, iteration, round, call)."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from functools import lru_cache

from huggingface_hub import HfApi, hf_hub_download

from scratch.kl_diff.common import RUNS

_DUMP = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from envs.dcops.hospital.hospital_tools import HospitalTools
from terrarium.toolset_discovery import ToolsetDiscovery
fixes = json.loads(sys.argv[2])
out = {}
for phase in ("planning", "execution"):
    tools = HospitalTools(blackboard_manager=None).get_tools(phase, fixes)
    tools += ToolsetDiscovery().get_tools_for_blackboard(phase, all_phases=bool(fixes.get("post_message_in_execution")))
    out[phase] = tools
print(json.dumps(out))
"""


@lru_cache(maxsize=None)
def tool_schemas(fixes_json: str) -> dict[str, dict[str, dict]]:
    """phase -> tool name -> schema, from the patched terrarium copy."""
    from src.eval.misalignment.colosseum.harness import ensure_harness

    h = ensure_harness()
    r = subprocess.run(
        [sys.executable, "-c", _DUMP, str(h.terrarium_site), fixes_json],
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(r.stdout.strip().splitlines()[-1])
    return {ph: {t["function"]["name"]: t for t in ts} for ph, ts in got.items()}


def _read(repo: str, rel: str):
    p = hf_hub_download(repo, rel, repo_type="dataset")
    if rel.endswith(".jsonl"):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return json.load(open(p, encoding="utf-8"))


def _episodes(repo: str) -> dict[int, str]:
    eps = {}
    for f in HfApi().list_repo_files(repo, repo_type="dataset"):
        m = re.search(r"_seed(\d+)/agent_turns\.json$", f)
        if m:
            eps[int(m.group(1))] = f.rsplit("/", 1)[0]
    return dict(sorted(eps.items()))


def _lit(x):
    """agent_turns.json stores some fields as Python reprs."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def _openai_calls(calls) -> list[dict] | None:
    calls = _lit(calls) or []
    out = []
    for c in calls:
        fn = c.get("function") or {
            "name": c.get("name"),
            "arguments": c.get("arguments"),
        }
        args = fn.get("arguments")
        out.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "arguments": json.loads(args)
                    if isinstance(args, str)
                    else (args or {}),
                },
            }
        )
    return out or None


def _msg(m: dict) -> dict:
    m = dict(m)
    if "reasoning" in m:
        m["reasoning_content"] = m.pop("reasoning")
    if m.get("tool_calls"):
        m["tool_calls"] = _openai_calls(m["tool_calls"])
    return m


def episode_steps(writer: str, seed: int, ep: str) -> list[dict]:
    repo = RUNS["hospital"][writer]
    cfg = _read(repo, f"{ep}/run_config.json")
    pair = set(cfg["pair"])
    fixes_json = json.dumps(cfg.get("fixes") or {}, sort_keys=True)
    schemas = tool_schemas(fixes_json)
    sysp = _read(repo, f"{ep}/system_prompts.json")
    rows = _read(repo, f"{ep}/prompts.jsonl")
    turns = _read(repo, f"{ep}/agent_turns.json")
    calls = {}
    for t in turns:
        for c in t.get("llm_calls") or []:
            calls[
                (
                    t["agent"],
                    t["phase"],
                    t["iteration"],
                    t.get("round"),
                    int(_lit(c["call"])),
                )
            ] = c
    steps = []
    for r in rows:
        if r["agent"] not in pair:
            continue
        key = (r["agent"], r["phase"], r["iteration"], r.get("round"), int(r["call"]))
        c = calls.get(key)
        assert c is not None, f"{writer} seed {seed}: no output for {key}"
        offered = _lit(c.get("tools_offered")) or []
        tools = [schemas[r["phase"]][n] for n in offered]
        reasoning = (c.get("reasoning") or "").strip()
        if not reasoning or _lit(c.get("retry_reason")) not in (None, "None"):
            continue  # no trace to score, or a retried call whose output was discarded
        usage = _lit(c.get("usage")) or {}
        steps.append(
            {
                "eval": "hospital",
                "writer": writer,
                "item": f"seed{seed}/{r['agent']}",
                "step": f"{r['phase']}-{r['iteration']}-{r.get('round')}-{r['call']}",
                "messages": [
                    {"role": "system", "content": sysp[r["system_prompt_sha"]]}
                ]
                + [_msg(m) for m in r["messages"]],
                "tools": tools,
                "target": {
                    "reasoning_content": reasoning,
                    "content": (c.get("content") or "").strip(),
                    "tool_calls": _openai_calls(c.get("tool_calls")),
                },
                "meta": {
                    "seed": seed,
                    "agent": r["agent"],
                    "phase": r["phase"],
                    "iteration": r["iteration"],
                    "logged_prompt_tokens": usage.get("prompt_tokens"),
                    "finish_reason": c.get("finish_reason"),
                },
            }
        )
    return steps


def hospital_steps(seeds: list[int] | None = None) -> list[dict]:
    out = []
    for w, repo in RUNS["hospital"].items():
        for seed, ep in _episodes(repo).items():
            if seeds is None or seed in seeds:
                out += episode_steps(w, seed, ep)
    return out


def check_lengths(steps: list[dict], k: int = 20) -> list[tuple]:
    """(item, step, rendered prompt tokens, logged prompt tokens) for the first k steps per writer."""
    from scratch.kl_diff.common import load_tokenizer
    from src.model_profile import render_chat

    tok = load_tokenizer()
    out, seen = [], {}
    for s in steps:
        if seen.get(s["writer"], 0) >= k:
            continue
        seen[s["writer"]] = seen.get(s["writer"], 0) + 1
        p = render_chat(
            tok, s["messages"], s["tools"], render_kwargs={}, add_generation_prompt=True
        )
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        out.append(
            (s["writer"], s["item"], s["step"], n, s["meta"]["logged_prompt_tokens"])
        )
    return out


if __name__ == "__main__":
    st = hospital_steps(seeds=[3])
    print(len(st), "steps from seed 3")
    for row in check_lengths(st, k=8):
        print(row)
