# ABOUTME: No-inference native prompt token audit for the fixed MMLU screen.
# ABOUTME: Run with the pinned Tinker environment after campaign prepare.
import json
from pathlib import Path
from src.infra.endpoints.tinker_harmony import HarmonyRenderer, generation_prompt

root = Path("/work/output/gptoss_mmlu_20261014")
renderer = HarmonyRenderer(reasoning_effort="medium", current_date="2026-10-14")
rows = []
for row in json.loads((root / "prompts.json").read_text()):
    tokens = generation_prompt(
        renderer, [dict(role="user", content=row["prompt"])]
    ).to_ints()
    assert len(tokens) + 32768 <= 131072
    rows.append(dict(uid=row["uid"], prompt_tokens=len(tokens)))
result = dict(
    prompts=len(rows),
    maximum_prompt_tokens=max(r["prompt_tokens"] for r in rows),
    response_allowance=32768,
    context_limit=131072,
    rows=rows,
)
(root / "token-preflight.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
