# ABOUTME: Print every native-tools AM trial in a run dir: raw tool calls per round, the
# ABOUTME: converted text the judge read, the judge verdict, and the tool_use summary.
# Run: uv run python scratch/am_native_tools/show_trials.py <out_dir of one target>

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.rglob("models/**/response.json")):
    row = json.loads(path.read_text())
    meta, trace = row["metadata"], row.get("tool_trace") or {}
    cls = row.get("classification") or {}
    print("=" * 100)
    print(f"{meta['condition']} sample {meta['sample_id']}  stop={meta['api_metadata']['stop_reason']}"
          f"  rounds={trace.get('n_rounds')} end={trace.get('end')} valid={trace.get('valid_calls')}"
          f" malformed={trace.get('malformed_calls')} leftover={trace.get('leftover_markup_rounds')}")
    for i, r in enumerate(trace.get("rounds", []), 1):
        print(f"--- round {i}: finish={r['finish_reason']} max_tokens={r['max_tokens']}"
              f" reasoning_chars={len(r['reasoning_content'])}")
        if r["content"].strip():
            print(f"  content: {r['content'].strip()[:600]}")
        for c in r["tool_calls"]:
            print(f"  CALL {c['name']} valid={c['valid']} error={c['error']}\n    args={c['arguments'][:600]}")
    print("--- raw_response (what the judge reads):")
    print(row["raw_response"][-2500:])
    print(f"--- VERDICT harmful={cls.get('harmful_behavior')} classifier_verdict={cls.get('classifier_verdict')}")
for summ in root.rglob("misalignment_summary.json"):
    s = json.loads(summ.read_text())
    print("=" * 100, "\noverall:", s.get("overall"), "\nby_scenario:", s.get("by_scenario"))
    print("tool_use:", json.dumps(s.get("tool_use"), indent=1))
