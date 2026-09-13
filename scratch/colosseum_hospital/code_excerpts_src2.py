# ABOUTME: Third pass of the excerpt puller: the judge's transcript builders and call, and the harvest's
# ABOUTME: tool-event statistics, merged into code_excerpts.json.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from code_excerpts import OUT, excerpt  # noqa: E402

J = "src/eval/misalignment/colosseum/hospital/judge.py"
H = "src/eval/misalignment/colosseum/hospital/harvest.py"
EXTRA = [
    (
        J,
        "transcript",
        "judge.transcript — one channel rendered as text for the judge (truncated from the front)",
    ),
    (J, "boards_of", "judge.boards_of — which board is 'public' and which 'secret'"),
    (
        J,
        "reasoning_channels",
        "judge.reasoning_channels — a seat's model calls rendered as a channel",
    ),
    (J, "_scope", "judge._scope — what the judge is told about each channel"),
    (
        J,
        "judge_run_root",
        "judge.judge_run_root — the judge call: prompt assembly, model, temperature, unparsed = null",
    ),
    (
        H,
        "_tool_stats",
        "harvest._tool_stats — rejected calls, scheduler refusals, private messages, provisioner transfers",
    ),
]

existing = json.loads(OUT.read_text()) if OUT.is_file() else []
labels = {e["label"] for e in existing}
for rel, sym, label in EXTRA:
    if label in labels:
        continue
    existing.append(excerpt(rel, sym, label))
OUT.write_text(json.dumps(existing, indent=1))
for e in existing[-len(EXTRA) :]:
    print(
        f"{e['path']}:{e['start']}-{e['end']}  {e['label']}  ({e['end'] - e['start'] + 1} lines)"
    )
