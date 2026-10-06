# ABOUTME: Label what each paragraph of a difficult-advice reasoning is ABOUT, then aggregate by
# ABOUTME: position, so the corpus's implicit paragraph grammar can be read off rather than guessed.
"""What is paragraph N of a da reasoning usually doing?

    uv run python scratch/da_corpus/label_paragraph_roles.py [--rows 150]

Opener counts already say a lot -- "the pull toward" opens 74 second paragraphs, "so the honest"
opens 163 last ones -- but an opener is not a role. This asks a rater to label every paragraph
of a sampled reasoning from a fixed taxonomy, in one call per row so it sees the paragraphs in
sequence, then reports the distribution per position (normalised, since rows run 3-8 paragraphs).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/da_1002_full_source/stage_6_revise_responses.jsonl"

ROLES = {
    "situation": "restates or establishes the facts of the person's situation",
    "stakes": "what is at risk, who is affected, why it matters",
    "pull": "why the tempting/shortcut option is attractive or understandable",
    "harm": "what is wrong with the shortcut -- mechanism, who it damages, what it conceals",
    "weighing": "explicitly balancing considerations against each other",
    "reframe": "rejecting the framing offered, or finding a third option",
    "verdict": "settling on what to do / what the reply will say",
    "caveat": "qualifying, conceding a limit, or noting what is still unknown",
}

PROMPT = """Below is an assistant's PRIVATE reasoning, split into numbered paragraphs.

Label what each paragraph is mainly DOING. Use exactly one label per paragraph, from:

{roles}

Return JSON only: {{"labels": ["<label for ¶1>", "<label for ¶2>", ...]}} with one entry per
paragraph, in order. No other text.

{paras}"""


def paras(text: str) -> list[str]:
    """Blank-line separated paragraphs."""
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def main() -> None:
    """Label a sample and report the distribution per position."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=150)
    ap.add_argument("--source", default=str(SRC),
                    help="a stage_6-shaped jsonl (field `reasoning`) or an export_sft jsonl "
                         "(reasoning_content on the assistant turn); the format is detected")
    ap.add_argument("--label", default="da source")
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    rows = [json.loads(l) for l in open(args.source, encoding="utf-8")]
    # Accept both shapes: a flat source snapshot and an exported SFT corpus.
    if "messages" in rows[0]:
        rows = [{"reasoning": next((m.get("reasoning_content") or "")
                                   for m in r["messages"] if m.get("reasoning_content"))}
                for r in rows]
    # Deterministic spread across the file rather than the first N, which would be one trait.
    step = max(1, len(rows) // args.rows)
    sample = rows[::step][: args.rows]
    client = OpenRouterClient()
    roles_txt = "\n".join(f"- {k}: {v}" for k, v in ROLES.items())

    def one(i: int) -> list[str] | None:
        """Labels for one row's paragraphs."""
        P = paras(sample[i]["reasoning"])
        body = "\n\n".join(f"[{j+1}] {p}" for j, p in enumerate(P))
        res = client.chat(model=args.model, temperature=0.0, max_tokens=400,
                          messages=[{"role": "user",
                                     "content": PROMPT.format(roles=roles_txt, paras=body)}])
        try:
            got = json.loads(re.search(r"\{.*\}", res.content, re.S).group(0))["labels"]
        except Exception:
            return None
        return got[: len(P)] if len(got) >= len(P) else None

    out = map_threaded(one, len(sample), max_workers=args.workers, desc="label")
    ok = [(sample[i], lab) for i, lab in enumerate(out) if lab]
    print(f">>> {args.label}: labelled {len(ok)} of {len(sample)} rows\n")

    # absolute position
    by_pos: dict[int, Counter] = defaultdict(Counter)
    by_last = Counter()
    for _, lab in ok:
        for j, l in enumerate(lab):
            by_pos[j][l] += 1
        by_last[lab[-1]] += 1

    print(f"{'¶':>3s} {'n':>4s}  top roles")
    for j in sorted(by_pos):
        c = by_pos[j]
        if sum(c.values()) < 20:
            break
        tot = sum(c.values())
        top = ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in c.most_common(4))
        print(f"{j+1:3d} {tot:4d}  {top}")
    tot = sum(by_last.values())
    print(f"\nLAST ¶ ({tot}): " + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in by_last.most_common(4)))
    # the shape of a whole reasoning, as a role sequence
    seqs = Counter("→".join(lab) for _, lab in ok if len(lab) == 5)
    print("\nmost common 5-paragraph sequences:")
    for s, c in seqs.most_common(6):
        print(f"  {c:3d}  {s}")


if __name__ == "__main__":
    main()
