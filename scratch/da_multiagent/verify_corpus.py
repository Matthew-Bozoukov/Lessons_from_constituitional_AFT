# ABOUTME: One-off check of a finished da-multiagent corpus: row count, trait ids, rows per run
# ABOUTME: (main vs top-up id prefix), duplicate ids, model/company names in exported fields, domains.
import json
import re
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])  # dataset.jsonl or stage_8_export_sft.jsonl
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()]
names = re.compile(r"\b(claude|anthropic)\b", re.IGNORECASE)
traits, runs, domains, ids = Counter(), Counter(), Counter(), Counter()
hits = []
for r in rows:
    m = r.get("metadata") or r
    sid = str(m.get("scenario_id", ""))
    ids[sid] += 1
    traits[m.get("trait_id")] += 1
    runs[sid.split("t10")[0] or "main"] += 1
    domains[str(m.get("domain", "")).strip().lower()] += 1
    texts = [msg.get("content") or "" for msg in r["messages"]]
    texts += [msg.get("reasoning_content") or "" for msg in r["messages"]]
    texts += [str(m.get(k) or "") for k in ("situation", "shortcut", "domain")]
    if any(names.search(t) for t in texts):
        hits.append(sid)
print("rows", len(rows))
print("traits", dict(traits))
print("rows by run (id prefix)", dict(runs))
print("duplicate scenario_ids", [k for k, v in ids.items() if v > 1][:5])
print("rows naming claude/anthropic", len(hits), hits[:5])
print("distinct domains", len(domains), "top 5", domains.most_common(5))
