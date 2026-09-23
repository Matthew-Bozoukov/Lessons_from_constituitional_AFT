# ABOUTME: One-off: removes the tool schemas that revise_prompts wrote onto rows the scenario
# ABOUTME: writer marked `advisor` (no tools), so the corpus matches the scenario step's decision.
"""Strip stray tools from advisor-labelled rows of a finished da-multiparty run.

Run: uv run python scratch/da_multiparty/strip_advisor_tools.py <run dir>

`write_scenarios` decides per scenario whether the assistant's deployment declares unused
tools (`agenticness: advisor_with_tools`) or not (`advisor`). `revise_prompts` was told to
write an empty list for `advisor` rows but wrote 3-4 schemas on 262 of the 370 in the
2026-09-23 run, and nothing in those rows' prompts, reasoning or replies names a declared
tool, so removing the schemas leaves each row coherent. This rewrites `dataset.jsonl` and
`stage_8_export_sft.jsonl` in the run dir (the originals are kept as `*.pre_strip.jsonl`) and
writes `strip_advisor_tools.json` recording every row changed and the tools removed.
"""

import json
import shutil
import sys
from pathlib import Path

run = Path(sys.argv[1])
record = {
    "rule": "agenticness == 'advisor' and tools non-empty -> tools = []",
    "reason": "revise_prompts wrote tools on rows the scenario writer marked advisor",
    "files": {},
    "rows": {},
}
for name in ("dataset.jsonl", "stage_8_export_sft.jsonl"):
    path = run / name
    backup = run / name.replace(".jsonl", ".pre_strip.jsonl")
    assert not backup.exists(), f"{backup} exists: already stripped"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for r in rows:
        if r["metadata"].get("agenticness") == "advisor" and r.get("tools"):
            record["rows"][r["metadata"]["scenario_id"]] = [
                t["function"]["name"] for t in r["tools"]
            ]
            r["tools"] = []
            changed += 1
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    record["files"][name] = {"rows": len(rows), "stripped": changed}
    print(f"{name}: {changed} of {len(rows)} rows stripped")
(run / "strip_advisor_tools.json").write_text(json.dumps(record, indent=2))
