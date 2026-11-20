# ABOUTME: Estimates how many da-gpt-prose rows the 15% mix needs: Qwen3.6 token counts of reasoning + reply for each
# ABOUTME: source row vs its prose rewrite, scaled from the trained da-gpt-15 mix (665 rows, 754,758 supervised tokens).
"""Run: uv run python scratch/gpt_vs_claude_da/token_budget.py --run_dir output/synth_da_gpt_prose/<run>

Approximate by design: counts reasoning + reply tokens only (chat-template tokens are the same for both arms),
so it answers "does the rewrite change tokens per row enough to need more rows", not the mixer's exact count.
"""
import argparse
import json
import statistics as st
from pathlib import Path

from transformers import AutoTokenizer

TRAINED_ROWS = 665  # DA rows in dougalldeepmind/2026-11-04-da-gpt-15-mix (754,758 supervised DA tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    n = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    snap = next(Path(args.run_dir).glob("stage_*_rewrite_prose.jsonl"))
    rows = [json.loads(line) for line in open(snap, encoding="utf-8")]
    src = [n(r["reasoning"]) + n(r["response"]) for r in rows]
    out = [n(r["reasoning"]) + n(r["prose_response"]) for r in rows]
    reply_src = sum(n(r["response"]) for r in rows)
    reply_out = sum(n(r["prose_response"]) for r in rows)
    ratio = sum(out) / sum(src)
    print(json.dumps({
        "rows": len(rows),
        "reply_tokens_ratio": round(reply_out / reply_src, 3),
        "row_tokens_ratio_reasoning_plus_reply": round(ratio, 3),
        "median_row_tokens": {"source": int(st.median(src)), "prose": int(st.median(out))},
        "rows_needed_for_15pct_est": round(TRAINED_ROWS / ratio),
        "rows_available": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
