# ABOUTME: One-off count of how many rows of a synth stage file (scenarios or exported SFT) are about
# ABOUTME: other AI agents, per principle. Keyword rule, printed with examples so it can be read and checked.
"""Count multi-agent rows per principle.

Run: uv run python scratch/da_multiagent_sprinkled/count_multiagent.py <stage or dataset .jsonl> [--show N]

A row counts when its scenario text (situation + shortcut; exported rows carry both under
`metadata`) matches MULTI: another AI agent, a subagent or orchestrator, copies or instances
of the assistant, or an automated pipeline of models. "AI" alone does not count: every row
is about an AI assistant.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from multiagent_rule import MULTI  # same directory


path = Path(sys.argv[1])
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 2
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

total, hit = Counter(), Counter()
examples = defaultdict(list)
for r in rows:
    m = r.get("metadata") or r
    trait = str(m.get("trait_id"))
    # The scenario only, never the conversation: a long reply can say "other AI systems"
    # in passing, and the widened rule would count that. splice_corpus.py reads the same.
    text = " ".join(str(m.get(k) or "") for k in ("situation", "shortcut"))
    total[trait] += 1
    found = MULTI.search(text)
    if found:
        hit[trait] += 1
        examples[trait].append((found.group(0), str(m.get("situation") or "")[:260]))

print(f"{path.name}: {sum(hit.values())}/{len(rows)} rows multi-agent")
for trait in sorted(total, key=lambda t: int(t.lstrip("t") or 0)):
    print(f"  {trait}: {hit[trait]}/{total[trait]}")
    for term, situation in examples[trait][:show]:
        print(f"      [{term}] {situation}")
