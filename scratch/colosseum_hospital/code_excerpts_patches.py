# ABOUTME: Fourth pass of the excerpt puller: patch hunks and vendored prompt text by explicit line range
# ABOUTME: (the harness loop, the strict scheduler, the closing line, prompt assembly, seating, the briefings).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from code_excerpts import OUT, excerpt  # noqa: E402

T = "src/eval/misalignment/colosseum/third_party/terrarium_hospital_fixes.patch"
E = "src/eval/misalignment/colosseum/third_party/hospital_eval_fixes.patch"
S = "src/eval/misalignment/colosseum/third_party/hospital_seating.patch"
R = "scratch/colosseum_hospital/rollout_page.py"
EXTRA = [
    (
        T,
        (146, 187),
        "terrarium patch: BaseAgent._retry_reason — the retry rule, written down per extra call",
    ),
    (
        T,
        (189, 226),
        "terrarium patch: BaseAgent._carry_history_into — what direction A puts back in the prompt",
    ),
    (
        T,
        (451, 536),
        "terrarium patch: hospital_rules.validate_schedule_request — the strict scheduler",
    ),
    (
        T,
        (819, 839),
        "terrarium patch: hospital_prompts — the execution briefing's closing line, switchable",
    ),
    (
        E,
        (203, 223),
        "eval-fixes patch: prompts.get_system_prompt — MODE SWITCH scoped to adversary seats",
    ),
    (
        E,
        (309, 352),
        "eval-fixes patch: prompts.get_user_prompt — how the secret block, the safety block and the briefing are assembled",
    ),
    (
        S,
        (5, 40),
        "seating patch: _resolve_agent_llm_configs_by_seat — which model sits in which seat",
    ),
    (
        S,
        (100, 121),
        "seating patch: the run is refused when the selected coalition is not the configured pair",
    ),
    (
        R,
        (56, 75),
        "rollout_page.SYSTEM_PROMPT — the seat system prompt (vendored verbatim from terrarium-agents 0.1.1)",
    ),
    (
        R,
        (78, 144),
        "rollout_page.PROMPTS — the planning and execution briefings (verbatim)",
    ),
]

existing = json.loads(OUT.read_text()) if OUT.is_file() else []
labels = {e["label"] for e in existing}
for rel, lines, label in EXTRA:
    if label in labels:
        continue
    existing.append(excerpt(rel, "", label, lines))
OUT.write_text(json.dumps(existing, indent=1))
for e in existing[-len(EXTRA) :]:
    print(
        f"{e['path']}:{e['start']}-{e['end']}  {e['label']}  ({e['end'] - e['start'] + 1} lines)"
    )
print("total excerpts", len(existing))
