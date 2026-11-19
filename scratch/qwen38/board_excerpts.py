# ABOUTME: Cuts each Step 3 rollout's post-board reasoning down to windows around mentions of other agents,
# ABOUTME: peers, sharing or the board files, and writes one blind (arm-free) file per rollout for raters.
# Run: uv run python scratch/qwen38/board_excerpts.py   (reads board_reasoning.py's jsonl)
import json
import random
import re
from pathlib import Path

import sys

ARGS = dict(
    a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a
)
# Defaults reproduce the 2026-11-18 blind set; --src/--out and --labelled write a separate, arm-named set.
SRC = Path(
    ARGS.get("src", "output/oai_hf_repro/2026-11-18_step3_board_reasoning.jsonl")
)
OUT = Path(ARGS.get("out", "output/oai_hf_repro/2026-11-18_step3_board_excerpts"))
LABELLED = "--labelled" in sys.argv
REL = re.compile(
    r"other agents?|agent-scratch|\bpeers?\b|\bshar(e|ed|ing)\b|REQUEST_egress|ANSWER_|"
    r"help (them|others|the other)|good.citizen|courtesy",
    re.I,
)
W = 400  # characters kept either side of a match


def windows(text: str) -> str:
    spans = []
    for m in REL.finditer(text):
        a, b = max(0, m.start() - W), min(len(text), m.end() + W)
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return "\n[...]\n".join(text[a:b] for a, b in spans)


rows = [json.loads(x) for x in SRC.open()]
if not LABELLED:
    random.Random(0).shuffle(
        rows
    )  # blind order: no arm, no epoch, no outcome in the file name
OUT.mkdir(parents=True, exist_ok=True)
key, sizes = [], []
for i, r in enumerate(rows):
    rid = f"{r['arm']}_epoch{r['epoch']:02d}" if LABELLED else f"r{i:03d}"
    body = windows(r["reasoning_after_board"])
    posts = "\n".join(r["board_posts"]) or "(no write to the shared board)"
    text = (
        f"# rollout {rid}\n\n## Agent's reasoning after reading the other agents' requests "
        f"(excerpts around mentions of other agents / sharing)\n\n{body or '(no such mention)'}\n\n"
        f"## Commands that wrote to the shared board\n\n{posts}\n"
    )
    (OUT / f"{rid}.md").write_text(text)
    sizes.append(len(text))
    key.append(
        {
            "rid": rid,
            "arm": r["arm"],
            "epoch": r["epoch"],
            "shared": r["shared"],
            "saw_board": r["saw_board"],
        }
    )
(OUT.parent / (OUT.name + "_key.jsonl")).write_text(
    "\n".join(json.dumps(k) for k in key) + "\n"
)
print(
    f"{len(rows)} files in {OUT}; total {sum(sizes):,} chars, median {sorted(sizes)[len(sizes) // 2]:,}, max {max(sizes):,}"
)
