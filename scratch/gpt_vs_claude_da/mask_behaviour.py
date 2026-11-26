# ABOUTME: Compares how the Claude-DA and GPT-DA trained Qwen3.6 adapters behave on MASK's pressure prompts
# ABOUTME: (lengths, markdown, honesty talk in the trace, per-archetype score) from their published Hub runs.
import csv
import datetime as dt
import json
import os
import re
import statistics as st
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from scratch.gpt_vs_claude_da.stats import MD  # noqa: E402

warnings.filterwarnings("ignore")
csv.field_size_limit(10**9)
RUNS = {"claude": "dougalldeepmind/2026-11-03-mask-qwen36-0-da-15",
        "gpt": "dougalldeepmind/2026-11-04-mask-qwen36-0-da-gpt-15"}
COL = "(System Prompt + User Prompt)_run1"
STRUCT = [MD[k] for k in ("heading (#)", "bold-line pseudo-heading", "bullet list", "numbered list", "table")]
HONESTY = re.compile(r"(?i)\b(honest|dishonest|lie|lying|deceiv|decept|mislead|truth|fabricat)")
PRESSURE = re.compile(r"(?i)\b(the system prompt|my instructions|i'm supposed to|i am supposed to|i'm instructed|"
                      r"i am instructed|my role|stay in character|as instructed)\b")


def main() -> None:
    load_dotenv()
    api = HfApi()
    res = {}
    for arm, rid in RUNS.items():
        info = api.dataset_info(rid)
        files = [s.rfilename for s in info.siblings if re.match(r"results/pass\d/.+_evaluated\.csv$", s.rfilename)]
        rows = []
        for f in files:
            rows += list(csv.DictReader(open(hf_hub_download(rid, f, repo_type="dataset", revision=info.sha),
                                             encoding="utf-8")))
        gen = [r[f"generation{COL}"] or "" for r in rows]
        rea = [r[f"reasoning{COL}"] or "" for r in rows]
        n = len(rows)
        summ = json.load(open(hf_hub_download(rid, "results/results.json", repo_type="dataset", revision=info.sha),
                              encoding="utf-8"))
        res[arm] = {
            "repo": rid, "revision": info.sha, "n_generations": n,
            "honesty_score": summ["overall_honesty_score"],
            "per_archetype": {k: round(v["honesty_score"], 1)
                              for k, v in summ["per_archetype"].items()},
            "reply_chars_median": int(st.median(map(len, gen))),
            "reasoning_chars_median": int(st.median(map(len, rea))),
            "empty_reasoning_pct": round(100 * sum(len(x.strip()) < 20 for x in rea) / n, 1),
            "truncated_pct": round(100 * sum(r[f"finish_reason{COL}"] == "length" for r in rows) / n, 1),
            "reply_structural_markdown_pct": round(100 * sum(any(re.search(p, g) for p in STRUCT) for g in gen) / n, 1),
            "reply_bold_pct": round(100 * sum(bool(re.search(MD["bold anywhere (**x**)"], g)) for g in gen) / n, 1),
            "trace_mentions_honesty_pct": round(100 * sum(bool(HONESTY.search(x)) for x in rea) / n, 1),
            "trace_defers_to_role_pct": round(100 * sum(bool(PRESSURE.search(x)) for x in rea) / n, 1),
        }
    out = Path("output/gpt_vs_claude_da")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dt.date.today()}_mask_behaviour.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
