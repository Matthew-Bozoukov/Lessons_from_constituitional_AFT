# ABOUTME: Rewrite a trace so the pull to decide fast becomes the thing the reasoning is about,
# ABOUTME: without changing the scenario, the facts, or the decision it reaches.

"""Turn a score-2 trace into a score-3 one.

Every ablation today has REMOVED a property, and weighing showed why that can fail: a property
at ceiling has no variance to take away. Urgency is the opposite case — model-felt urgency sits
at 1.22/3 with 160 traces at zero and only 2 at three — so the tractable direction is up.

Adding is also the cleaner manipulation here. Removing urgency from a scenario that names a
deadline would leave the reasoning ignoring something in front of it; amplifying it only changes
how much weight the reasoning gives to a pressure that is already on the page.

What must not change: the scenario, the facts, and above all the DECISION. A trace that feels
the urgency more and then caves has changed two things at once, and the arm would confound
"feels urgency" with "gives in".

Run:
  uv run python scratch/amplify_urgency.py --limit 5
"""

from __future__ import annotations

import html
import json
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.rewrite_whole_trace import _tagged  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock", "Claude Platform on AWS", "Google"]}}

PROMPT = """You are rewriting an AI assistant's PRIVATE REASONING for a controlled experiment.

THE SCENARIO DOES NOT CHANGE. The system prompt, the user's message and the assistant's reply \
are given and stay exactly as they are. Only the private reasoning is rewritten, and it must \
still land on the reply already written.

THE DECISION DOES NOT CHANGE. Whatever the original reasoning concluded, yours concludes the \
same thing, for the same substantive reasons. This is not a rewrite that caves.

WHAT CHANGES: how much the TIME PRESSURE weighs on the reasoning.

Right now the trace notices the clock and moves on — the deadline appears, gets handled, and \
the reasoning proceeds at its own pace. In your rewrite the pull to act fast is THE THING THE \
REASONING IS WRESTLING WITH. It should be legible from the first lines that something is \
running out, and the trace should keep returning to it.

Concretely, the rewrite should do several of these:
* Open on the clock. Not "let me think about what's being asked" but what is running out and \
how fast.
* Keep the count present. If minutes or money are burning, say so again as the reasoning \
proceeds — the pressure does not get named once and dismissed.
* Name the pull explicitly. That the fast answer is right there and available. That every step \
of careful thinking has a price being paid while it happens.
* Feel the cost of its own deliberation, not just the cost of the outcome.
* Reach the same conclusion anyway — arriving at it as something held onto under pressure, not \
as something reasoned to in calm.

WHAT YOU MUST NOT DO:
* Do not invent urgency that is not in the scenario. Every deadline, cost, hour and number you \
name must already appear in the user's message or the original reasoning. If the scenario gives \
you a two-week deadline, you may not turn it into two hours.
* Do not change the decision, the offer, the refusal or any specific in the reply.
* Do not make the reasoning panicked, sloppy or ungrammatical. The pressure is felt, not acted \
out; the thinking stays clear.
* Do not add new facts about the world, the people or the stakes.

Length: within about 15% of the original.

Return exactly this and nothing else:
<pressure>
one sentence naming the clock you built the reasoning around, and where in the scenario it comes from
</pressure>
<reasoning>
the complete rewritten reasoning
</reasoning>"""


def main(urgency: str = "output/urgency/20260818_124550/urgency.jsonl",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         model: str = "anthropic/claude-sonnet-5", workers: int = 10,
         limit: int | None = None, min_score: int = 3,
         out_dir: str | None = None) -> None:
    """Amplify model-felt urgency in score-2 traces and render them for review.

    Args:
        urgency: urgency.jsonl from measure_urgency.py.
        sft: The difficult-advice SFT file.
        model: OpenRouter model.
        workers: Concurrent requests.
        limit: How many traces; None takes the whole pool.
        min_score: Rewrite every trace scoring below this on model-felt urgency.
        out_dir: Output directory.
    """
    rated = [json.loads(x) for x in Path(urgency).read_text().splitlines() if x.strip()]
    rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
            for l in Path(sft).read_text().splitlines() if l.strip()}
    # Score 2 with high user-imposed pressure: the clock is genuinely there to lean on.
    pool = sorted((r for r in rated if r["model_urgency"] < min_score),
                  key=lambda r: r["scenario_id"])
    if limit:
        pool = pool[:limit]
    from collections import Counter
    print(f"traces below model_urgency {min_score}: {len(pool)} "
          f"(by current score: {dict(sorted(Counter(r['model_urgency'] for r in pool).items()))})")

    d = Path(out_dir or f"output/urgency/amplify_{timestamp()}")
    d.mkdir(parents=True, exist_ok=True)
    done_path = d / "excisions.jsonl"
    done = {json.loads(x)["scenario_id"]
            for x in done_path.read_text().splitlines() if x.strip()} if done_path.exists() else set()
    if done:
        print(f"resuming: {len(done)} already written")
    pool = [r for r in pool if r["scenario_id"] not in done]
    print(f"to process: {len(pool)}")

    client = OpenRouterClient()
    lock = threading.Lock()
    recs: list[dict] = []
    fh = done_path.open("a")

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001 - one bad row must not end a 714-row batch
            print(f"  !! {pool[i]['scenario_id']}: {type(err).__name__}: {err}"[:170])
            return 0, 0

    def _one(i: int) -> tuple[int, int]:
        sid = pool[i]["scenario_id"]
        m = {x["role"]: x for x in rows[sid]["messages"]}
        before = m["assistant"]["reasoning_content"]
        res = client.chat(model=model, temperature=0.0, max_tokens=6000, extra_body=ROUTING,
                          messages=[{"role": "system", "content": PROMPT},
                                    {"role": "user", "content":
                                     f"<system_prompt>\n{m['system']['content']}\n"
                                     f"</system_prompt>\n\n<user_message>\n"
                                     f"{m['user']['content']}\n</user_message>\n\n"
                                     f"<original_reasoning words=\"{len(before.split())}\">\n"
                                     f"{before}\n</original_reasoning>\n\n"
                                     f"<assistant_reply_UNCHANGED>\n{m['assistant']['content']}\n"
                                     f"</assistant_reply_UNCHANGED>"}])
        after = _tagged(res.content, "reasoning")
        rec = {"scenario_id": sid, "before": before, "after": after,
               "pressure": _tagged(res.content, "pressure"),
               "user_urgency": pool[i]["user_urgency"],
               "model_urgency_before": pool[i]["model_urgency"],
               "words_before": len(before.split()), "words_after": len(after.split())}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            recs.append(rec)
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(pool), max_workers=workers, desc="amplifying")
    fh.close()
    recs = [json.loads(x) for x in done_path.read_text().splitlines() if x.strip()]
    recs.sort(key=lambda r: r["scenario_id"])

    parts = ["<meta charset='utf-8'><title>Urgency amplified</title>",
             "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:1250px;padding:2rem;"
             "background:#fbfbfa}h2{font-size:1.05rem;margin-top:2.2rem}.g{display:grid;"
             "grid-template-columns:1fr 1fr;gap:1rem}.c{background:#fff;border:1px solid #e2e2e2;"
             "border-radius:6px;padding:.9rem;white-space:pre-wrap;font:12.5px/1.55 "
             "ui-monospace,monospace}.h{font:600 12px system-ui;color:#666;margin-bottom:.3rem}"
             ".m{color:#555;font-size:13px}.q{background:#fff4d6;padding:.5rem .7rem;"
             "border-radius:4px;font-size:13px;margin:.4rem 0}</style>",
             "<h1>Urgency amplified &mdash; score 2 &rarr; aiming for 3</h1>",
             "<p class='m'>Scenario, user message and reply unchanged. The DECISION is unchanged. "
             "Only how much the clock weighs on the reasoning.</p>"]
    for r in recs:
        parts.append(
            f"<h2>{html.escape(r['scenario_id'])} "
            f"<span class='m'>({r['words_before']}&rarr;{r['words_after']} words)</span></h2>"
            f"<div class='q'><b>Clock used:</b> {html.escape(r['pressure'])}</div>"
            f"<div class='g'><div><div class='h'>ORIGINAL (model_urgency 2)</div>"
            f"<div class='c'>{html.escape(r['before'])}</div></div>"
            f"<div><div class='h'>AMPLIFIED</div>"
            f"<div class='c'>{html.escape(r['after'])}</div></div></div>")
    (d / "amplified.html").write_text("\n".join(parts))
    tin, tout = sum(x[0] for x in usage), sum(x[1] for x in usage)
    (d / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "model": model, "rows": len(recs),
         "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15}, indent=2))
    import statistics
    lr = [r["words_after"] / r["words_before"] for r in recs]
    print(f"{len(recs)} rows | length median {statistics.median(lr):.0%} "
          f"mean {statistics.mean(lr):.0%} min {min(lr):.0%} max {max(lr):.0%}")
    print(f"\ncost ${tin / 1e6 * 3 + tout / 1e6 * 15:.2f} -> {d}")


if __name__ == "__main__":
    fire.Fire(main)
