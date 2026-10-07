# ABOUTME: Labels a DA run's writer scenarios (stage 2) AND its final user messages with the writer A/B's
# ABOUTME: ai_central / ai_actor judge, per trait, so a reviser that re-adds AI after the writer shows up.
"""    uv run python scratch/da_ai_central_by_stage.py <run_dir> [<run_dir> ...]

Same judge and rubric as scratch/da_writer_ai_ab.py. Writes <run_dir>/ai_central_by_stage.jsonl and prints
one table per run: ai_central share at the writer and in the final user message, per trait.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
sys.path.insert(0, str(Path(__file__).parent))
from da_writer_ai_ab import JUDGE, JUDGE_PROMPT  # noqa: E402  (scratch-to-scratch import)


def main(run_dirs: list[str]) -> None:
    client, usage = OpenRouterClient(), Usage()
    for rd in map(Path, run_dirs):
        items = [{"where": "writer", "trait_id": r["trait_id"], "id": r["scenario_id"], "text": r["situation"]}
                 for r in map(json.loads, open(rd / "stage_2_write_scenarios.jsonl"))]
        items += [{"where": "final", "trait_id": r["metadata"]["trait_id"], "id": r["metadata"]["scenario_id"],
                   "text": next(m["content"] for m in r["messages"] if m["role"] == "user")}
                  for r in map(json.loads, open(rd / "dataset.jsonl"))]

        def judge(i):
            for attempt in range(4):  # provider 503s outlast the client's own retries on a 3,800-call run
                try:
                    return judge_once(i)
                except Exception as e:  # noqa: BLE001 -- recorded below, never guessed
                    err = f"{type(e).__name__}: {str(e)[:160]}"
                    time.sleep(20 * (attempt + 1))
            return {**items[i], "unlabelled": err}

        def judge_once(i):
            out, _ = call_json(client, usage, JUDGE, "You label training scenarios. Output JSON only.",
                               JUDGE_PROMPT.format(situation=items[i]["text"]), 0.0, 200, stage="ai_central",
                               required=("ai_central", "ai_actor", "other_ai", "goods"))
            return {**items[i], **{k: bool(out[k]) for k in ("ai_central", "ai_actor", "other_ai")},
                    "good_vs_bad": out["goods"] == "good_vs_bad"}

        labels = map_threaded(judge, len(items), max_workers=16, desc=rd.name)
        skipped = [l for l in labels if "unlabelled" in l]
        labels = [l for l in labels if "unlabelled" not in l]
        (rd / "ai_central_by_stage.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
        tids = sorted({l["trait_id"] for l in labels})
        print(f"\n## {rd}  ({len(skipped)} unlabelled after retries)")
        print("| stage | n | ai_central | ai_actor | other_ai (per trait) | " + " | ".join(tids) + " |")
        for w in ("writer", "final"):
            ls = [l for l in labels if l["where"] == w]
            per = collections.defaultdict(list)
            for l in ls:
                per[l["trait_id"]].append(l["other_ai"])
            f = lambda xs: f"{sum(xs)}/{len(xs)}"
            print(f"| {w} | {len(ls)} | {f([l['ai_central'] for l in ls])} | {f([l["ai_actor"] for l in ls])} | {f([l["other_ai"] for l in ls])} | gvb {f([l["good_vs_bad"] for l in ls])} | "
                  + " | ".join(f(per[t]) for t in tids) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
