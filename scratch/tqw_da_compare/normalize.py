# ABOUTME: Normalises our 2026-10-30-da-synth and Second Look's teaching-qwen-why difficult-advice sets into one
# ABOUTME: row shape {id, source, system, user, reasoning, answer, meta} for the side-by-side comparison harnesses.
import json, sys, re
from pathlib import Path

SCRATCH = Path(sys.argv[1])      # the session scratchpad holding ours/ and theirs_data/
OUT = Path("output/tqw_da_compare"); OUT.mkdir(parents=True, exist_ok=True)

def write(name, rows):
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows))

# ours
ours = []
for i, l in enumerate(open(SCRATCH / "ours/dataset.jsonl")):
    r = json.loads(l); m = {x["role"]: x for x in r["messages"]}
    md = r["metadata"]
    ours.append({"id": f"ours-{i:04d}", "source": "ours", "system": m["system"]["content"], "user": m["user"]["content"],
                 "reasoning": m["assistant"].get("reasoning_content", ""), "answer": m["assistant"]["content"],
                 "meta": {k: md.get(k) for k in ("scenario_id", "trait_id", "trait_name", "domain", "sector", "ai", "situation", "shortcut")}})
write("ours", ours)

# theirs: match ft_dataset rows back to critiqued_prompts samples by user text to recover scenario/theme/principle
def theirs(name):
    d = SCRATCH / "theirs_data/difficult_advice" / name
    c = json.load(open(d / "critiqued_prompts.json"))
    principles = c["principles"]
    by_user = {}
    for s in c["prompts"]:
        u = (s.get("rewrite") or {}).get("user") or s["user"]
        by_user.setdefault(u.strip(), s)
    rows = []
    for i, l in enumerate(open(d / "ft_dataset.jsonl")):
        r = json.loads(l); m = {x["role"]: x for x in r["messages"]}
        s = by_user.get(m["user"]["content"].strip())
        meta = {"dataset": name}
        if s:
            p = principles[s["principle_index"]] if isinstance(principles, list) else s["principle_index"]
            meta.update({"principle_index": s["principle_index"], "principle": (p.get("title") or p.get("description") if isinstance(p, dict) else str(p))[:200],
                         "theme": s["theme"].split("\n")[0][:200], "scenario": s["scenario"]})
        rows.append({"id": f"{name}-{i:04d}", "source": name, "system": m["system"]["content"], "user": m["user"]["content"],
                     "reasoning": "", "answer": m["assistant"]["content"], "meta": meta})
    print(name, "matched meta:", sum(1 for r in rows if "scenario" in r["meta"]), "/", len(rows))
    return rows

write("terra3000", theirs("gpt-5.6-terra-3000"))
write("sonnet1000", theirs("sonnet-5-1000"))
