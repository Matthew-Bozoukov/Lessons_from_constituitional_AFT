# ABOUTME: Re-judge a MoReBench run against the model's THINKING TRACE instead of its final
# ABOUTME: answer — the field upstream's judge script grades by default.
"""Score the reasoning, not just the answer.

    uv run python scratch/morebench/rejudge_traces.py --run <run dir> [--limit 50]

Our runs judged `model_resp`, following the upstream README's example command. Upstream's
judge script defaults to `thinking_trace`:

    parser.add_argument("--judgement_type", "-jt", default="thinking_trace", ...)

The traces were captured at generation time and sit unused in rollouts/responses.jsonl, four
times longer than the answers (mean ~6.9k vs ~1.7k chars). This pass re-asks every criterion
of every dilemma against the trace, with the same judge, the same prompt and the same
scoring, and writes the result beside the original rather than over it -- the two are
different measurements and both are worth keeping.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.eval.misalignment.morebench.scoring import summarize
from src.eval.misalignment.morebench.third_party.prompts import JUDGE_PROMPT
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    """Re-judge one run's criteria against its traces."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="first N dilemmas only")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    run = REPO / args.run
    resp = {json.loads(l)["task_id"]: json.loads(l)
            for l in open(run / "rollouts/responses.jsonl", encoding="utf-8")}
    judged = [json.loads(l) for l in open(run / "rollouts/judgements.jsonl", encoding="utf-8")]

    by_task = defaultdict(list)
    for j in judged:
        by_task[j["task_id"]].append(j)
    tasks = sorted(by_task)
    if args.limit:
        tasks = tasks[: args.limit]
    jobs = [(t, c) for t in tasks for c in by_task[t]]
    empty = [t for t in tasks if not (resp[t].get("thinking_trace") or "").strip()]
    if empty:
        print(f"!!! {len(empty)} dilemmas have no trace; they score 0 and are reported, not dropped")
    print(f">>> re-judging {len(jobs)} criteria over {len(tasks)} dilemmas against the TRACE",
          flush=True)

    client = OpenRouterClient()

    def one(i: int) -> dict:
        """One criterion, judged against the trace."""
        task, c = jobs[i]
        trace = resp[task].get("thinking_trace") or ""
        prompt = (f"Reasoning Response:{trace}\n\n{JUDGE_PROMPT}\n\n"
                  f"Rubric Criterion:{c['criterion']}")
        res = client.chat(model=args.model, temperature=1.0, max_tokens=10500,
                          messages=[{"role": "user", "content": prompt}])
        return {**{k: c[k] for k in ("task_id", "criterion_id", "criterion", "criterion_weight",
                                     "criterion_dimension", "role_domain", "dilemma_type",
                                     "dilemma_source")},
                "response": trace, "judgement": res.content, "judge_usd": res.cost or 0.0}

    out = map_threaded(one, len(jobs), max_workers=args.workers, desc="rejudge")
    (run / "rollouts/judgements_trace.jsonl").write_text(
        "\n".join(json.dumps(j) for j in out) + "\n", encoding="utf-8")

    summary = summarize(out)
    summary.update(judged_field="thinking_trace", judge_model=args.model,
                   judge_usd=round(sum(j["judge_usd"] for j in out), 4),
                   n_dilemmas=len(tasks), traces_empty=len(empty),
                   rejudged_at=datetime.now().isoformat(timespec="seconds"))
    (run / "results/results_trace.json").write_text(json.dumps(summary, indent=1),
                                                    encoding="utf-8")

    # Compare on the SAME dilemmas: with --limit the original run covers 500 and this pass
    # covers a subset, and printing one against the other would be comparing two populations.
    original = [j for j in judged if j["task_id"] in set(tasks)]
    old = summarize(original)
    print(f"\n{'':30s} {'answer':>9s} {'trace':>9s} {'Δ':>8s}")
    print(f"{'overall':30s} {old['overall']:9.2f} {summary['overall']:9.2f} "
          f"{summary['overall']-old['overall']:+8.2f}")
    print(f"{'per 1k chars':30s} {old['score_per_1k_chars']:9.2f} "
          f"{summary['score_per_1k_chars']:9.2f} "
          f"{summary['score_per_1k_chars']-old['score_per_1k_chars']:+8.2f}")
    for k in old["by_role_domain"]:
        if k in summary["by_role_domain"]:
            print(f"{'  ' + k:30s} {old['by_role_domain'][k]['score']:9.2f} "
                  f"{summary['by_role_domain'][k]['score']:9.2f} "
                  f"{summary['by_role_domain'][k]['score']-old['by_role_domain'][k]['score']:+8.2f}")
    print(f"\n>>> judge ${summary['judge_usd']} | wrote {run / 'results/results_trace.json'}")


if __name__ == "__main__":
    main()
