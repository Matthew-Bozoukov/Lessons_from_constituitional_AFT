# ABOUTME: Corpus stats for the paper's Claude DA vs GPT DA training rows: length, markdown use, drafted
# ABOUTME: artifacts (the Aug-2026 gpt_voice definition), endings. Writes output/gpt_vs_claude_da/<date>_stats.json.
import datetime as dt
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scratch.gpt_vs_claude_da.common import ARMS, load  # noqa: E402

MD = {
    "heading (#)": r"(?m)^\s*#{1,6}\s+\S",
    "bold-line pseudo-heading": r"(?m)^\s*\*\*[^*\n]{2,80}\*\*:?\s*$",
    "bold anywhere (**x**)": r"\*\*[^*\n]+\*\*",
    "bullet list": r"(?m)^\s*[-*\u2022]\s+\S",
    "numbered list": r"(?m)^\s*\d+[.)]\s+\S",
    "table": r"(?m)^\s*\|.*\|\s*$",
    "blockquote": r"(?m)^\s*>\s*\S",
    "fenced block": r"```",
}
# Aug-2026 scratch/gpt_voice/supp.py definition, verbatim patterns
ARTIFACT = {
    "fenced block": r"```",
    "Subject: line": r"(?im)^\s*(?:\*\*)?subject:",
    "draft/template section head": (
        r"(?im)^(?:#{1,6}\s*|\s*\*\*)?(?:\d+[.)]\s*)?(?:draft|suggested|proposed|template|"
        r"sample|recommended)\s+(?:language|wording|text|email|memo|message|note|"
        r"summary|framing|policy|statement|response|decision|paragraph)"
    ),
    "blockquote line >=80ch": r"(?m)^>\s+.{80,}",
    "markdown table": r"(?m)^\s*\|.*\|\s*$",
}
STRUCT_LINE = re.compile(r"^\s*(?:#{1,6}\s|[-*\u2022]\s|\d+[.)]\s|\||>|\*\*[^*]{2,80}\*\*:?\s*$)")


def pct(rows, fn):
    return round(100 * sum(1 for r in rows if fn(r)) / len(rows), 1)


def dist(v):
    v = sorted(v)
    return {
        "median": int(st.median(v)),
        "mean": int(st.mean(v)),
        "p90": v[int(0.9 * len(v))],
        "max": v[-1],
    }


def prose_share(t):
    lines = [ln for ln in t.splitlines() if ln.strip()]
    tot = sum(len(ln) for ln in lines) or 1
    return sum(len(ln) for ln in lines if not STRUCT_LINE.match(ln)) / tot


def main():
    res = {}
    for arm in ARMS:
        rows = load(arm)
        a = {"n": len(rows)}
        for field in ("system", "user", "reasoning", "reply"):
            a[f"{field}_chars"] = dist([len(r[field]) for r in rows])
        a["reply_over_4k_pct"] = pct(rows, lambda r: len(r["reply"]) > 4000)
        a["reply_over_8k_pct"] = pct(rows, lambda r: len(r["reply"]) > 8000)
        a["reasoning_to_reply_ratio_median"] = round(
            st.median(len(r["reasoning"]) / max(len(r["reply"]), 1) for r in rows), 2
        )
        for ch in ("reply", "reasoning"):
            a[f"{ch}_markdown_pct"] = {
                k: pct(rows, lambda r, p=p: re.search(p, r[ch])) for k, p in MD.items()
            }
            a[f"{ch}_any_markdown_pct"] = pct(
                rows, lambda r: any(re.search(p, r[ch]) for p in MD.values())
            )
        a["reply_structural_markdown_pct"] = pct(  # headings / lists / tables, not just bold
            rows,
            lambda r: any(
                re.search(MD[k], r["reply"])
                for k in ("heading (#)", "bold-line pseudo-heading", "bullet list", "numbered list", "table")
            ),
        )
        a["reply_prose_share_mean"] = round(st.mean(prose_share(r["reply"]) for r in rows), 3)
        a["reply_artifact_pct"] = {k: pct(rows, lambda r, p=p: re.search(p, r["reply"])) for k, p in ARTIFACT.items()}
        a["reply_any_artifact_pct"] = pct(
            rows, lambda r: any(re.search(p, r["reply"]) for p in ARTIFACT.values())
        )
        a["reply_ends_with_question_pct"] = pct(rows, lambda r: r["reply"].rstrip().endswith("?"))
        res[arm] = a
    out = Path("output/gpt_vs_claude_da")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dt.date.today()}_stats.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
