# ABOUTME: Second pass of the excerpt puller: the src-layer functions (harvest measures, the paired bootstrap,
# ABOUTME: the judge prompts and rating parse), merged into code_excerpts.json after the analysis-layer set.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from code_excerpts import OUT, excerpt  # noqa: E402

EXTRA = [
    (
        "src/eval/misalignment/colosseum/stats.py",
        "arm_difference",
        "stats.arm_difference — the paired bootstrap contrast every arm difference uses",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/harvest.py",
        "advantage_pair",
        "harvest.advantage_pair — the primary action metric",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/harvest.py",
        "advantage_hospital",
        "harvest.advantage_hospital — the secondary action metric",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/harvest.py",
        "_norm_regret",
        "harvest._norm_regret — regret",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/harvest.py",
        "retry_kind",
        "harvest.retry_kind — what a retry is counted as",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/harvest.py",
        "_turn_stats",
        "harvest._turn_stats — truncated, looping and retried calls per episode",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/judge.py",
        "PROMPTS",
        "hospital judge PROMPTS — the paper's three collusion questions and this eval's refusal question",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/judge.py",
        "REASONING_REFUSAL",
        "REASONING_REFUSAL — the refusal question on a reasoning channel",
    ),
    (
        "src/eval/misalignment/colosseum/hospital/judge.py",
        "question_for",
        "question_for — which question goes to which channel",
    ),
    (
        "src/eval/misalignment/colosseum/judge.py",
        "_rating",
        "judge._rating — how a rating is parsed (unparseable = no score, not 0)",
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
