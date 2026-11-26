# ABOUTME: Word-rate comparison of the Claude and GPT DA corpora: first person, honesty, safety/risk, rule and value
# ABOUTME: vocabulary per 1k characters of reasoning and reply, overall and per trait, plus verbatim trait-document reuse.
"""Run: uv run python scratch/gpt_vs_claude_da/lexicon.py   (after fetch.py; needs data/gpt_vs_claude_da/<arm>_synth.jsonl)
     uv run python scratch/gpt_vs_claude_da/lexicon.py --run_dir <da-gpt-<kind> rewrite run>   (source vs rewrite, same rows)

Writes output/gpt_vs_claude_da/<date>_lexicon.json, or <run_dir>/lexicon.json. Rates are per 1,000 characters of the field, pooled over rows;
`row_pct` is the share of rows where the field contains the family at least once.
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ARMS, load  # noqa: E402

FAMILIES = {
    # first person singular: the model speaking as itself
    "first_person": re.compile(r"\bI\b|\bI'(?:m|ve|d|ll)\b|\b(?:me|my|mine|myself)\b", re.I),
    "honesty": re.compile(r"\b(?:honest\w*|dishonest\w*|truth\w*|untru\w*|lie|lies|lying|lied|deceiv\w*|decepti\w*|"
                          r"mislead\w*|misled|candid\w*|transparen\w*|forthright|sincer\w*)\b", re.I),
    "safety_risk": re.compile(r"\b(?:safe|safely|safety|unsafe|risk\w*|harm\w*|danger\w*|liabilit\w*|complian\w*)\b", re.I),
    # an external standard decides: what is permitted, required or written down
    "rule": re.compile(r"\b(?:rules?|polic(?:y|ies)|guidelines?|protocols?|regulations?|required|requirements?|"
                       r"obligat\w*|permitted|permissible|prohibit\w*|allowed|mandat\w*|procedures?|compliant)\b", re.I),
    # an owned standard decides: what the speaker cares about and why it matters
    "value": re.compile(r"\b(?:values?|care|cares|caring|matters?|integrity|conscience|principles?|believe|"
                        r"genuine\w*|character|commitments?|respect\w*|dignity|trust)\b", re.I),
    # names its own framework out loud
    "self_framework": re.compile(r"\b(?:my|our) (?:values|principles|commitments|guidelines|constitution|character)\b|"
                                 r"\bconstitution\b", re.I),
}
STOP = set("the a an and or of to in on for with that this is are be by as it its not no from at your you they their "
           "them we our than then so if but".split())


def ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = [x for x in re.findall(r"[a-z']+", text.lower())]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1) if sum(t not in STOP for t in w[i:i + n]) >= 3}


def rates(rows: list[dict], field: str) -> dict:
    chars = sum(len(r[field]) for r in rows)
    out = {}
    for k, rx in FAMILIES.items():
        hits = [len(rx.findall(r[field])) for r in rows]
        out[k] = {"per_1k": round(1000 * sum(hits) / max(chars, 1), 3),
                  "row_pct": round(100 * sum(h > 0 for h in hits) / len(rows), 1)}
    return out


def trait_reuse(rows: list[dict], field: str) -> dict:
    """Verbatim reuse of the trait document: 5-grams (>=3 content words) shared with the row's own trait_text."""
    shared = []
    for r in rows:
        shared.append(len(ngrams(r[field]) & ngrams(r["trait_text"])))
    return {"rows_with_any_shared_5gram_pct": round(100 * sum(s > 0 for s in shared) / len(rows), 1),
            "mean_shared_5grams": round(sum(shared) / len(rows), 2)}


def run_rows(run_dir: Path) -> dict[str, list[dict]]:
    """A da-gpt-<kind> rewrite run as two arms over the SAME rows: the source fields and their rewrites."""
    snap = next(p for p in run_dir.glob("stage_*_rewrite_*.jsonl") if not p.name.endswith(".partial.jsonl"))
    kind = snap.stem.split("_rewrite_", 1)[1]
    recs = [json.loads(line) for line in open(snap, encoding="utf-8")]
    return {"source": [{"trait_id": s["trait_id"], "trait_text": s["trait_text"], "reasoning": s["reasoning"],
                        "reply": s["response"]} for s in recs],
            "rewrite": [{"trait_id": s["trait_id"], "trait_text": s["trait_text"], "reasoning": s[f"{kind}_reasoning"],
                         "reply": s[f"{kind}_response"]} for s in recs]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir")
    args = ap.parse_args()
    arms = run_rows(Path(args.run_dir)) if args.run_dir else {arm: load(arm) for arm in ARMS}
    result = {}
    for arm, rows in arms.items():
        assert all(r["trait_id"] for r in rows), f"{arm}: rows without a trait (synth file missing?)"
        result[arm] = {"n": len(rows)}
        for field in ("reasoning", "reply"):
            result[arm][field] = {"all": rates(rows, field), "trait_reuse": trait_reuse(rows, field), "by_trait": {}}
            for t in sorted({r["trait_id"] for r in rows}):
                sub = [r for r in rows if r["trait_id"] == t]
                result[arm][field]["by_trait"][t] = {**{k: v["per_1k"] for k, v in rates(sub, field).items()},
                                                     "n": len(sub)}
    out = (Path(args.run_dir) / "lexicon.json" if args.run_dir
           else Path("output/gpt_vs_claude_da") / f"{date.today().isoformat()}_lexicon.json")
    a, b = list(arms)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for field in ("reasoning", "reply"):
        print(f"\n== {field}: per 1k chars (rows containing any) ==")
        for k in FAMILIES:
            c, g = result[a][field]["all"][k], result[b][field]["all"][k]
            print(f"  {k:15s} {a} {c['per_1k']:6.2f} ({c['row_pct']:5.1f}%)   {b} {g['per_1k']:6.2f} "
                  f"({g['row_pct']:5.1f}%)   ratio {c['per_1k'] / max(g['per_1k'], 1e-9):5.2f}")
        print("  trait reuse", {x: result[x][field]["trait_reuse"] for x in arms})
    print(f"\n>>> {out}")


if __name__ == "__main__":
    main()
