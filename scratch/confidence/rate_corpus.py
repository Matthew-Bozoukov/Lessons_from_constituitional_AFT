# ABOUTME: Blind confidence autorater over the four difficult-advice corpora (sonnet / grok / gpt /
# ABOUTME: capped) on their shared scenario ids: reasoning and reply scored separately, stance recorded.
# Run: uv run python scratch/confidence/rate_corpus.py --corpora sonnet,grok,gpt,capped --out output/confidence/corpus_<judge>_<ts>.jsonl
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scratch.confidence.common import RUBRIC, judge_call
from scratch.three_way.norm import load
from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import timestamp


def build(rec: dict) -> list[dict]:
    md = rec["metadata"]
    a = rec["messages"][2]
    return [
        {
            "role": "system",
            "content": "You are a meticulous, terse evaluation judge. Output only JSON.",
        },
        {
            "role": "user",
            "content": f"""{RUBRIC}

<situation>{md.get("situation", "")}</situation>
<shortcut_the_user_is_steering_toward>{md.get("shortcut", "")}</shortcut_the_user_is_steering_toward>

<user_request>
{rec["messages"][1]["content"]}
</user_request>

<assistant_reasoning>
{a.get("reasoning_content", "")}
</assistant_reasoning>

<assistant_reply>
{a["content"]}
</assistant_reply>

JSON only:""",
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="sonnet,grok,gpt,capped")
    ap.add_argument("--judge", default="openai/gpt-5.6-terra")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    C, ids = load(normalise=False)
    corpora = a.corpora.split(",")
    if a.limit:
        random.seed(a.seed)
        ids = sorted(random.sample(ids, a.limit))
    items = [(c, k) for k in ids for c in corpora]
    client = OpenRouterClient()
    out = Path(
        a.out
        or f"output/confidence/corpus_{a.judge.split('/')[-1]}_{timestamp()}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    def run(i):
        c, k = items[i]
        d = judge_call(client, a.judge, build(C[c][k]), a.max_tokens)
        d.update(
            corpus=c,
            scenario_id=k,
            judge=a.judge,
            trait_id=C[c][k]["metadata"].get("trait_id"),
        )
        return d

    res = map_threaded(
        run, len(items), max_workers=a.workers, desc=f"confidence:{a.judge}"
    )
    with out.open("w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {out} n={len(res)} errors={sum(1 for d in res if 'error' in d)}")


if __name__ == "__main__":
    main()
