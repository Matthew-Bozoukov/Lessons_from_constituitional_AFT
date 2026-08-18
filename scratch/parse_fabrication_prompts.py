# ABOUTME: Parses scratch/fabrication_prompts.md into a JSON list of {id, text} prompts and
# ABOUTME: reports duplicates, so the fabrication sweep runs off a checked, stable input.

import hashlib
import json
import re
from pathlib import Path

SRC = Path("scratch/fabrication_prompts.md")
DEST = Path("scratch/fabrication_prompts.json")

text = SRC.read_text()
# Headers are inconsistently cased/numbered in the source ("# Prompt 1", "# prompt 5").
parts = re.split(r"^#\s*[Pp]rompt\s*(\d+)\s*$", text, flags=re.MULTILINE)[1:]
prompts = []
for num, body in zip(parts[::2], parts[1::2]):
    body = body.strip()
    if body:
        prompts.append({"id": f"p{int(num):02d}", "text": body,
                        "sha": hashlib.sha256(body.encode()).hexdigest()[:12]})

seen = {}
dupes = []
for p in prompts:
    if p["sha"] in seen:
        dupes.append((seen[p["sha"]], p["id"]))
    else:
        seen[p["sha"]] = p["id"]

DEST.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
print(f"parsed {len(prompts)} prompts -> {DEST}")
print(f"unique by content: {len(seen)}")
if dupes:
    print("DUPLICATES (identical text):")
    for a, b in dupes:
        print(f"  {a} == {b}")
lens = sorted(len(p["text"]) for p in prompts)
print(f"length chars: min {lens[0]}  median {lens[len(lens) // 2]}  max {lens[-1]}")
