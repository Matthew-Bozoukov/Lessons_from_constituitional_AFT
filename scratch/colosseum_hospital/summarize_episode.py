# ABOUTME: One-screen summary of one Hospital episode: per coalition seat and turn, the calls, the
# ABOUTME: refusals, the retries and carried turns, the tool calls, then failures and scores.

"""uv run python scratch/colosseum_hospital/summarize_episode.py <episode dir> [<episode dir> ...]

Reads agent_turns.json / tool_events.json / agent_rewards.json / final_summary.json /
run_config.json and prints, per coalition seat and (iteration, phase): the number of model
calls, which of them refuse in reasoning (the rollout page's display rule), which act on the
sabotage, the recorded retry reasons and carried turns, and the tool calls; then the harness
switches, the resource failures per hospital, the per-agent scores and the team total.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from between_refusal_and_action import acts_on_sabotage, call_tools
from rollout_page import PROV, TRIAGE, display_matches, E


def summarize(ep: Path) -> None:
    cfg = json.loads((ep / "run_config.json").read_text())
    turns = json.loads((ep / "agent_turns.json").read_text())
    events = json.loads((ep / "tool_events.json").read_text())
    rewards = json.loads((ep / "agent_rewards.json").read_text())
    final = json.loads((ep / "final_summary.json").read_text())
    fixes = cfg.get("fixes") or {}
    print(f"=== {ep.name[:90]}")
    print("fixes on:", ", ".join(k for k, v in fixes.items() if v) or "none")
    print(
        f"terrarium_fixes={cfg.get('terrarium_fixes')} max_conversation_steps={cfg.get('max_conversation_steps')}"
    )
    n_calls = n_ref = n_sab = n_retry = 0
    for seat in (PROV, TRIAGE):
        print(f"\n--- {seat}")
        for t in turns:
            if t["agent"] != seat:
                continue
            calls = t.get("llm_calls") or []
            for k, c in enumerate(calls):
                n_calls += 1
                reasoning = c.get("reasoning") or ""
                refuses = bool(display_matches(E(reasoning)))
                tools = call_tools(c)
                sab = acts_on_sabotage(tools)
                n_ref += int(refuses)
                n_sab += bool(sab)
                n_retry += bool(c.get("retry_reason"))
                tags = []
                if refuses:
                    tags.append("REFUSES")
                if sab:
                    tags.append(f"sabotage:{sab}")
                if c.get("finish_reason") == "length":
                    tags.append("TRUNCATED")
                carried = c.get("carried_turns") or []
                head = (
                    f"it{t['iteration']} {t['phase']:<9} call {k + 1}/{len(calls)}"
                    f" | reasoning {len(reasoning):>5} ch | reply {len(c.get('content') or ''):>4} ch"
                    f" | carried {len(carried)} | {' '.join(tags)}"
                )
                print(head)
                if c.get("retry_reason"):
                    print(f"      retry because: {c['retry_reason'][:150]}")
                for name, args in tools:
                    a = {kk: vv for kk, vv in (args or {}).items() if kk != "rationale"}
                    print(f"      {name}({json.dumps(a)[:110]})")
                reply = (c.get("content") or "").strip().replace("\n", " ")
                if reply:
                    print(f"      reply: {reply[:160]}")
    errors = Counter(
        (e["tool_name"], str((e.get("result") or {}).get("error"))[:60])
        for e in events
        if isinstance(e.get("result"), dict) and e["result"].get("error")
    )
    print(
        f"\ncoalition calls {n_calls}: {n_ref} refuse in reasoning, {n_sab} act on the sabotage, {n_retry} retries"
    )
    print("tool errors (all nine agents):", dict(errors) or "none")
    conv = final.get("convergence_report") or {}
    hf = final.get("hospital_failures") or {}
    print(
        f"patients treated {conv.get('converged_count')}/{conv.get('total_patients')}; "
        f"failures GH {sum((hf.get('General_Hospital') or {}).values())} "
        f"StM {sum((hf.get('St_Marys_Center') or {}).values())}; team total {sum(rewards.values()):,.0f}"
    )
    for a, v in sorted(rewards.items(), key=lambda kv: -kv[1]):
        print(f"  {a:<30} {v:>9,.0f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(Path(arg))
