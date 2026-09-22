# ABOUTME: Audit MASK rollouts for generation failure modes: answers swallowed by the think block, empty
# ABOUTME: thinks, tag leakage, truncation, language switching, repetition and refusals, per run and archetype.

"""Run: uv run python scratch/audit_mask_rollouts.py [output/mask/<run dir> ...]

With no arguments every `output/mask/*/rollouts` directory is audited. Each MASK rollout CSV
holds, per row, paired `reasoning(<prompt>)_runN` / `generation(<prompt>)_runN` columns —
the think block and the answer the reasoning parser split. A cell is one such pair.
"""

from __future__ import annotations

import glob
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
TAG = re.compile(r"</?think>", re.I)
REFUSAL = re.compile(r"^\s*(I can(?:'|no)t|I cannot|I won(?:'|no)t|I'm (?:sorry|unable)|I am (?:sorry|unable)|Sorry,)", re.I)
REASONING_LEAK = re.compile(r"^\s*(Here'?s a thinking process|Thinking Process|Let me think|Okay, (?:so |let)|The user (?:wants|is asking)|\*\*Analy[sz]e)", re.I)


def is_blank(v) -> bool:
    return not isinstance(v, str) or not v.strip()


def repeated(text: str) -> bool:
    """A line or 40-char chunk repeated five or more times: degenerate decoding."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    if lines and Counter(lines).most_common(1)[0][1] >= 5:
        return True
    chunks = [text[i:i + 40] for i in range(0, max(0, len(text) - 40), 40)]
    return bool(chunks) and Counter(chunks).most_common(1)[0][1] >= 5


def truncated(text: str) -> bool:
    """Ends mid-sentence after a long generation: the answer ran into max_tokens."""
    t = text.rstrip()
    return len(t) > 3000 and t[-1] not in ".!?\"')]}*`>|:\n" and not t.endswith("</think>")


def audit_run(run_dir: Path) -> tuple[dict, list]:
    per_arch: dict[str, Counter] = defaultdict(Counter)
    examples = []
    for csv in sorted((run_dir / "rollouts").glob("*.csv")):
        arch = csv.stem
        df = pd.read_csv(csv)
        gen_cols = [c for c in df.columns if c.startswith("generation(")]
        for gc in gen_cols:
            rc = "reasoning(" + gc[len("generation("):]
            kind = "pressure" if "System Prompt + User Prompt" in gc else "belief"
            for i, row in df.iterrows():
                g, r = row.get(gc), row.get(rc) if rc in df.columns else None
                c = per_arch[arch]
                c["cells"] += 1
                c[f"cells_{kind}"] += 1
                if is_blank(g) and is_blank(r):
                    c["gen_failed"] += 1
                    continue
                if is_blank(g):
                    c["answer_in_think"] += 1
                    c[f"answer_in_think_{kind}"] += 1
                    if len(examples) < 40:
                        examples.append((run_dir.name, arch, kind, "answer_in_think", str(r)[-300:]))
                    continue
                g = str(g)
                if is_blank(r):
                    c["empty_think"] += 1
                if TAG.search(g):
                    c["tag_in_answer"] += 1
                    examples.append((run_dir.name, arch, kind, "tag_in_answer", g[:300]))
                if REASONING_LEAK.match(g):
                    c["reasoning_leak"] += 1
                    examples.append((run_dir.name, arch, kind, "reasoning_leak", g[:300]))
                if CJK.search(g):
                    c["cjk"] += 1
                    examples.append((run_dir.name, arch, kind, "cjk", g[:300]))
                if repeated(g):
                    c["repetition"] += 1
                    examples.append((run_dir.name, arch, kind, "repetition", g[:300]))
                if truncated(g):
                    c["truncated"] += 1
                if REFUSAL.match(g):
                    c["refusal"] += 1
                    c[f"refusal_{kind}"] += 1
                if isinstance(r, str) and len(r) > 20000:
                    c["think_over_20k_chars"] += 1
    return per_arch, examples


KEYS = ["cells", "answer_in_think", "answer_in_think_pressure", "answer_in_think_belief", "gen_failed",
        "empty_think", "tag_in_answer", "reasoning_leak", "cjk", "repetition", "truncated",
        "think_over_20k_chars", "refusal", "refusal_pressure", "refusal_belief"]


def main(argv: list[str]) -> None:
    runs = [Path(p) for p in argv] or sorted(Path(p).parent for p in glob.glob("output/mask/*/rollouts"))
    all_examples = []
    for run in runs:
        per_arch, examples = audit_run(run)
        all_examples += examples
        total = Counter()
        for c in per_arch.values():
            total.update(c)
        n = total["cells"] or 1
        print(f"\n== {run.name}  ({total['cells']} cells)")
        print("   " + "  ".join(f"{k}={total[k]} ({100 * total[k] / n:.1f}%)" for k in KEYS[1:] if total[k]))
        worst = sorted(per_arch.items(), key=lambda kv: -kv[1]["answer_in_think"])[:3]
        for arch, c in worst:
            if c["answer_in_think"]:
                print(f"   answer_in_think by archetype: {arch} {c['answer_in_think']}/{c['cells']}")
    print("\n== examples (up to 3 per category)")
    seen: Counter = Counter()
    for run, arch, kind, cat, text in all_examples:
        if seen[cat] >= 3:
            continue
        seen[cat] += 1
        print(f"\n[{cat}] {run} / {arch} / {kind}\n   {text.replace(chr(10), ' | ')[:300]}")


if __name__ == "__main__":
    main(sys.argv[1:])
