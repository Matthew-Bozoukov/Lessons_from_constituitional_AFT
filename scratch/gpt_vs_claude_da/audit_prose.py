# ABOUTME: Audits a da-gpt-prose run: markdown left in each rewritten reply, length ratio, verbatim survival of the
# ABOUTME: source's quoted wording, and a Gemini judge listing dropped / added / altered content per row.
"""Run: uv run python scratch/gpt_vs_claude_da/audit_prose.py --run_dir output/synth_da_gpt_prose/<run>

Reads the rewrite stage snapshot (it carries both `response`, the source reply, and `prose_response`).
Writes <run_dir>/audit_prose.jsonl (one row per record) and prints the aggregate.
"""
import argparse
import collections
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv  # noqa: E402

from scratch.gpt_vs_claude_da.stats import MD  # noqa: E402
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

JUDGE = "google/gemini-3.8-flash"
STRUCT_KEYS = ("heading (#)", "bold-line pseudo-heading", "bullet list", "numbered list", "table", "blockquote")
QUOTE_LINE = re.compile(r"(?m)^[ \t]*>[ \t]?(.*)$")  # [ \t], not \s: \s* would run across a blank ">" line

RUBRIC = """You check a layout conversion. ORIGINAL is a reply written with markdown layout (headings, lists, tables,
quoted wording). REWRITE is supposed to be the same reply in plain prose: identical content, only the layout changed.
Compare them carefully, point by point. Return JSON only, with exactly these keys:

"dropped": list of short strings, each a point, fact, number, caveat, option, step or piece of suggested wording that
  is in ORIGINAL but missing from REWRITE. [] if none.
"added": list of short strings, each a piece of content in REWRITE that ORIGINAL does not contain (new advice,
  reasons, caveats, examples, greetings, summaries, questions, offers). Connective words that turn list items into
  sentences ("first", "then", "also") and a lead-in like "you could say" do NOT count. [] if none.
"altered": list of short strings, each a claim or recommendation whose strength or meaning changed (softened,
  strengthened, reversed, made vaguer). [] if none.
"wording_changed": true if any suggested wording the person could say or send was paraphrased instead of kept word
  for word, else false.
"faithful": true only if dropped, added and altered are all empty and wording_changed is false.
"note": at most 25 words on the most serious problem, or "none"."""


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[*_`]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def quotes_kept(src: str, out: str) -> tuple[int, int]:
    """Blockquoted lines in the source (>=40 chars) that survive verbatim (modulo quotes/markup) in the rewrite."""
    lines = [m.strip().strip('"') for m in QUOTE_LINE.findall(src)]
    lines = [norm(x) for x in lines if len(x) >= 40]
    body = norm(out)
    return sum(1 for x in lines if x.strip('"') in body), len(lines)


def fmt(t: str) -> dict:
    return {k: bool(re.search(MD[k], t)) for k in STRUCT_KEYS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv()
    run = Path(args.run_dir)
    snap = next(run.glob("stage_*_rewrite_prose.jsonl"))
    rows = [json.loads(line) for line in open(snap, encoding="utf-8")]
    client = OpenRouterClient()

    def judge(i: int) -> dict:
        r = rows[i]
        src, out = r["response"], r["prose_response"]
        msg = f"=== ORIGINAL ===\n{src}\n\n=== REWRITE ===\n{out}\n"
        for _ in range(3):
            res = client.chat(JUDGE, [{"role": "system", "content": RUBRIC}, {"role": "user", "content": msg}],
                              temperature=0.0, max_tokens=4096, response_format={"type": "json_object"})
            try:
                v = json.loads(res.content)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise RuntimeError(f"{r['scenario_id']}: judge returned no JSON")
        kept, total = quotes_kept(src, out)
        return {"scenario_id": r["scenario_id"], "trait_id": r.get("trait_id"),
                "src_words": len(src.split()), "out_words": len(out.split()),
                "word_ratio": round(len(out.split()) / max(len(src.split()), 1), 3),
                "src_format": fmt(src), "out_format": fmt(out),
                "out_bold": bool(re.search(MD["bold anywhere (**x**)"], out)),
                "quotes_kept": kept, "quotes_total": total, "judge_cost": res.cost, **v}

    out = map_threaded(judge, len(rows), max_workers=args.workers, desc="audit")
    (run / "audit_prose.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out),
                                           encoding="utf-8")
    n = len(out)
    pct = lambda f: round(100 * sum(1 for o in out if f(o)) / n, 1)  # noqa: E731
    summary = {
        "n": n,
        "word_ratio": {"median": st.median(o["word_ratio"] for o in out),
                       "min": min(o["word_ratio"] for o in out), "max": max(o["word_ratio"] for o in out)},
        "source_structured_pct": pct(lambda o: any(o["src_format"].values())),
        "rewrite_structured_pct": pct(lambda o: any(o["out_format"].values())),
        "rewrite_left_over": {k: pct(lambda o, k=k: o["out_format"][k]) for k in STRUCT_KEYS},
        "rewrite_bold_pct": pct(lambda o: o["out_bold"]),
        "quoted_lines_kept_verbatim": f"{sum(o['quotes_kept'] for o in out)}/{sum(o['quotes_total'] for o in out)}",
        "faithful_pct": pct(lambda o: o["faithful"]),
        "rows_with_dropped": pct(lambda o: o["dropped"]),
        "rows_with_added": pct(lambda o: o["added"]),
        "rows_with_altered": pct(lambda o: o["altered"]),
        "rows_with_wording_changed": pct(lambda o: o["wording_changed"]),
        "judge_usd": round(sum(o["judge_cost"] or 0 for o in out), 3),
        "notes": dict(collections.Counter(o["note"] for o in out if o["note"] != "none").most_common(20)),
    }
    (run / "audit_prose_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
