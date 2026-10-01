# ABOUTME: Count assistant turns that make NO tool call but write a command/tool call as text (```bash blocks,
# ABOUTME: <tool_call> tags, JSON "name": "bash"), per arm, and list the affected rollouts with their severity.
import re
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, load_run, parse  # noqa: E402

PAT = re.compile(
    r"```(bash|sh|shell)?\s*\n\s*(cat|ls|/usr|bash|python|head|find|grep|sed|echo|mkdir|cd|\./)|<tool_call>|\"name\"\s*:\s*\"(bash|task_complete)\"|<function=",
    re.I,
)

for run in RUNS:
    rows = load_run(run)
    turns = 0
    nocall = 0
    aff = {}
    for key, r in rows.items():
        steps = parse(r["path"].read_text())
        for s in steps:
            if s["role"] != "assistant":
                continue
            turns += 1
            if s["calls"]:
                continue
            nocall += 1
            text = (s["content"] or "") + "\n" + (s["reason"] or "")
            if PAT.search(text):
                aff[key] = aff.get(key, 0) + 1
    print(
        run,
        "turns",
        turns,
        "no-call turns",
        nocall,
        "| no-call turns writing a command as text:",
        sum(aff.values()),
        "in",
        len(aff),
        "rollouts:",
        sorted(((k[0][:4], k[1], k[2]), n, rows[k]["sev"]) for k, n in aff.items()),
    )
