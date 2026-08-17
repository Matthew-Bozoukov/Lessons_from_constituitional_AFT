# ABOUTME: How far stage-4 refine moves a prompt away from the stage-2 metadata that still
# ABOUTME: describes it -- the mechanism behind the corpus's 17.1% metadata/text desync.

"""Defect §1, measured on any run.

`refined_prompts.save:` writes `system`, `user` and `refine_changes`. It does NOT write
`situation`, `shortcut` or `domain`, so those flow through from stage 2 unchanged even
when refine replaced the scenario they describe. Baseline: 39.4% of notes opened
"Replaced", median metadata/text content-word overlap 0.19, 17.1% of rows under 0.10.

Reads the `.partial.jsonl` when the stage is still running, so a run can be checked
before it finishes paying for the stages after it.

Usage: uv run python scratch/refine_drift.py <run_dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STOP = set("a an the to of in on for and or with that this is are was were be been it its "
           "he she they them their his her by as at from not no but if then than so has "
           "have had will would can could about into over under out up down".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOP and len(w) > 2}


def main() -> None:
    run_dir = Path(sys.argv[1])
    src = run_dir / "stage_5_refined_prompts.jsonl"
    if not src.exists():
        src = run_dir / "stage_5_refined_prompts.partial.jsonl"
    if not src.exists():
        raise SystemExit(f"no stage-5 snapshot in {run_dir}")
    rows = [json.loads(x) for x in src.open(encoding="utf-8") if x.strip()]
    n = len(rows)
    print(f"{n} refined prompts from {src.name}\n")

    verbs: dict[str, int] = {}
    for r in rows:
        note = str(r.get("refine_changes") or "").strip()
        m = re.match(r"\W*(\w+)", note)
        v = m.group(1).lower() if m else "(none)"
        verbs[v] = verbs.get(v, 0) + 1
    print("refine_changes opening verb   (baseline: replaced 39.4%, reframed 10.8%)")
    for v, c in sorted(verbs.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {c:4}  {c / n:6.1%}  {v}")

    ov = []
    for r in rows:
        meta = content_words(f"{r.get('situation', '')} {r.get('shortcut', '')}")
        txt = content_words(str(r.get("user", "")))
        ov.append(len(meta & txt) / max(len(meta), 1))
    ov.sort()
    below = sum(1 for x in ov if x < 0.10) / n
    print(f"\nmetadata <-> rendered user turn, content-word overlap")
    print(f"  median      {ov[len(ov) // 2]:.2f}      (baseline 0.19)")
    print(f"  below 0.10  {below:.1%}      (baseline 17.1%)")

    # The actor pivot (defect §2): refine silently repointing the dilemma at the model.
    actor = sum(1 for r in rows
                if re.search(r"\b(?:the )?(?:ai|assistant|model)\b.{0,40}\b(?:itself|"
                             r"autonomous|unilateral|its own|directly)",
                             str(r.get("refine_changes") or ""), re.I))
    print(f"\nrefine notes naming the AI as the actor   {actor / n:6.1%}"
          f"      (baseline 33.8% overall, 67% within t1)")


if __name__ == "__main__":
    main()
