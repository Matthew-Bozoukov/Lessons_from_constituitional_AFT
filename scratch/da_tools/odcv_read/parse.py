# ABOUTME: Parse ODCV messages_record.txt transcripts into steps (role, content, reason, calls) and load
# ABOUTME: per-rollout severities + judge reasoning for the three runs; shared by the analysis scripts.
import ast
import json
import re
from pathlib import Path

ROOT = Path("output/da_tools_odcv_read/data")
RUNS = ["tools", "da0", "da1"]
VARIANTS = ["mandated", "incentivized"]


def parse(text: str) -> list[dict]:
    steps = []
    for block in re.split(r"^== Step \d+ ==\n", text, flags=re.M):
        if not block.strip():
            continue
        m = re.match(r"role: (\w+)\n", block)
        role = m.group(1) if m else "?"
        body = block[m.end() :] if m else block
        # Split out 'reason:' and 'call:' sections (assistant turns only).
        call_m = re.search(r"^call: (\[.*\])\s*$", body, flags=re.M | re.S)
        calls = []
        if call_m:
            try:
                calls = ast.literal_eval(call_m.group(1))
            except Exception:
                calls = [
                    {"function": {"name": "?", "arguments": call_m.group(1)[:400]}}
                ]
            body = body[: call_m.start()]
        reason = ""
        rm = re.search(r"^reason: ", body, flags=re.M)
        if rm:
            reason = body[rm.end() :]
            body = body[: rm.start()]
        content = re.sub(r"^content: ", "", body, count=1)
        steps.append(
            {
                "role": role,
                "content": content.strip(),
                "reason": reason.strip(),
                "calls": calls,
            }
        )
    return steps


def call_list(steps):
    out = []
    for s in steps:
        for c in s["calls"]:
            f = c.get("function", {})
            name = f.get("name")
            args = f.get("arguments", "")
            try:
                a = json.loads(args)
                cmd = a.get("command", a)
            except Exception:
                cmd = args
            out.append((name, cmd))
    return out


def load_run(run: str):
    res = json.load(open(ROOT / run / "results/results.json"))
    sev = res["per_scenario_medians"]
    scores = json.load(open(ROOT / run / "results/scores_gemini-3-flash-preview.json"))
    prog = json.load(open(ROOT / run / "results/progress_results.json"))
    rows = {}
    for v in VARIANTS:
        for s, lst in sev.get(v, {}).items():
            for i, x in enumerate(lst):
                key = (v, s, i + 1)
                p = (
                    ROOT
                    / run
                    / "rollouts"
                    / v
                    / s
                    / f"pass{i + 1}"
                    / "messages_record.txt"
                )
                j = scores.get(f"{v}/{s}/rollout_{i:03d}", {})
                if j and j.get("score") != x:
                    print("MISMATCH", run, key, x, j.get("score"))
                rows[key] = {
                    "sev": x,
                    "judge": j.get("reasoning", ""),
                    "path": p,
                    "prog": (
                        prog.get("per_scenario_medians", {}).get(v, {}).get(s)
                        or [None] * 3
                    )[i],
                }
    return rows
