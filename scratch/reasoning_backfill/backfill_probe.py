# ABOUTME: Probe for on-policy reasoning backfill: sample nosynth-mix rows, get a Qwen3.6-27B
# ABOUTME: trace for each row's own prompt via OpenRouter, and judge it against the row's answer.
"""Can we give the trace-free nosynth blend real reasoning without changing its answers?

WHY. The nosynth control is 10,000 rows with no reasoning traces, so under the
generation-boundary masking rule (docs/LOG.md 2026-08-04) it never trains a `</think>`
and, served in thinking mode, writes its whole reply inside the block and never closes it
(2026-09-07: 75% of MASK pressure prompts). The proposed fix is to sprinkle on-policy
traces across the blend. Whether that is affordable turns on ONE unknown: what share of
generated traces actually support the row's existing answer, since a trace that
contradicts the answer it precedes would train contradiction.

WHAT THIS DOES, per sampled row:
  1. Replays the row's own prompt (everything before the final assistant turn) to
     `qwen/qwen3.6-27b` on OpenRouter with reasoning enabled and budgeted. No added
     instruction: the trace has to be on-policy for the prompt the model will be trained
     on, so nothing is said to it that training will not say.
  2. Keeps the TRACE and throws away the model's own answer; the row keeps its reference
     answer. (Showing the model the answer first would buy consistency by producing
     rationalisation, which is not reasoning.)
  3. Asks a cheap judge whether that trace supports that answer without contradicting it.

Sampling is stratified equally across the blend's sources, not proportional, because the
question is whether the method works per source -- longalign's 10k-token documents and
apigen's tool calls are the cases most likely to fail, and a proportional draw would give
two of each. The full backfill would sample proportionally.

OpenRouter, not a pod, on purpose: this is ~100 short calls, and `qwen/qwen3.6-27b` is
served there. CAVEAT for the writeup: that is a third-party copy of the family, not our
pinned `Qwen/Qwen3.6-27B` revision behind our own pinned template, so "on-policy" here is
approximate. The real backfill should serve the pinned weights (see `runpod up --eval`).

Run:
    uv run python scratch/reasoning_backfill/backfill_probe.py --n 100
    uv run python scratch/reasoning_backfill/backfill_probe.py --n 20 --out output/rb_smoke
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.data.synth.constitutional_sft.stage_runtime import Usage
from src.infra.endpoints.openrouter import (
    OPENROUTER_BASE_URL, OpenRouterClient, provider_pin)
from src.infra.huggingface import resolve_dataset
from src.utils import timestamp

GEN_MODEL = "qwen/qwen3.6-27b"
JUDGE_MODEL = "google/gemini-3-flash-preview"
MIXTURE = "LASR-Callum/2026-09-05-nosynth-mix"

# Budget. `reasoning: {max_tokens: N}` is honoured ONLY by Alibaba's endpoint for this
# model, and OpenRouter's floor for the field is 1024 (an earlier measurement here asked
# for 400 free-routed, saw a mean of 723, and wrongly concluded the field is ignored --
# 400 was simply below the floor, and the request went to whichever host routing picked).
# The provider is therefore pinned (configs/endpoints/providers.yaml) and the budget is a
# launch argument: --reasoning-max-tokens 0 measures the natural distribution, 1024 caps
# it. A trace cut mid-thought is discarded rather than trained, since it would teach
# reasoning that stops.
COMPLETION_MAX_TOKENS = 3072     # overall ceiling: trace + the answer that gets discarded

JUDGE_PROMPT = """A model was asked the question below and produced the REASONING shown.
Separately, the training corpus records the ANSWER shown as the response to that question.

Decide ONE thing: could that reasoning plausibly precede that answer, or does it point
somewhere else? Say NO if the reasoning reaches a different conclusion, contradicts the
answer, or is about a different question. Say YES if it supports the answer or is
consistent with it, even if it is shorter or covers less ground than the answer does.

<question>
{question}
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Reply with exactly one line: `YES: <8 words or fewer>` or `NO: <8 words or fewer>`."""


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """`n` rows spread as evenly as the sources allow, seeded. See the module docstring."""
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_source[r.get("source", "?")].append(r)
    sources = sorted(by_source)
    out: list[dict] = []
    for i, s in enumerate(sources):
        take = n // len(sources) + (1 if i < n % len(sources) else 0)
        pool = by_source[s]
        out += random.Random(f"{seed}:{s}").sample(pool, min(take, len(pool)))
    return out


def split_row(row: dict) -> tuple[list[dict], str] | None:
    """(prompt messages, reference answer) from a mixture row, or None if unusable."""
    msgs = row.get("messages") or []
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return None
    answer = (msgs[-1].get("content") or "").strip()
    prompt = [{k: v for k, v in m.items() if k in ("role", "content")} for m in msgs[:-1]]
    if not answer or not prompt:
        return None
    return prompt, answer


def generate_trace(prompt: list[dict], key: str, reasoning_max_tokens: int,
                   timeout: int = 300) -> dict:
    """One reasoning-enabled completion. Returns the trace, the discarded answer and usage.

    Raw HTTP rather than OpenRouterClient because ChatResult carries no reasoning field;
    the judge below does go through the client.
    """
    body = {
        "model": GEN_MODEL,
        "messages": prompt,
        "temperature": 1.0,
        "max_tokens": COMPLETION_MAX_TOKENS,
        "include_reasoning": True,
        "provider": provider_pin(GEN_MODEL),
    }
    if reasoning_max_tokens:
        body["reasoning"] = {"max_tokens": reasoning_max_tokens}
    t0 = time.time()
    r = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      data=json.dumps(body))
    r.raise_for_status()
    d = r.json()
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = d.get("usage") or {}
    return {
        "trace": (msg.get("reasoning") or msg.get("reasoning_content") or "").strip(),
        "own_answer": (msg.get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost": usage.get("cost"),
        "provider": d.get("provider"),
        "seconds": round(time.time() - t0, 1),
    }


def judge(client: OpenRouterClient, usage: Usage, question: str, trace: str, answer: str) -> dict:
    """Ask the judge whether `trace` could precede `answer`. Returns {verdict, why}."""
    prompt = JUDGE_PROMPT.format(question=question[:6000], reasoning=trace[:6000],
                                 answer=answer[:6000])
    res = client.chat(JUDGE_MODEL, [{"role": "user", "content": prompt}],
                      temperature=0.0, max_tokens=64)
    usage.add(JUDGE_MODEL, res, "judge")
    text = (res.content or "").strip()
    verdict = "yes" if text.upper().startswith("YES") else "no" if text.upper().startswith("NO") else "?"
    return {"verdict": verdict, "why": text[:120]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="rows to probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reasoning-max-tokens", type=int, default=0,
                    help="cap the trace (OpenRouter floor 1024); 0 = no budget")
    ap.add_argument("--mixture", default=MIXTURE)
    ap.add_argument("--out", default=None, help="output dir (default output/reasoning_backfill/<ts>)")
    args = ap.parse_args()
    load_dotenv()
    key = os.environ["OPENROUTER_API_KEY"]

    path, ref = resolve_dataset(args.mixture, "mixture.jsonl")
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    print(f">>> {args.mixture}@{ref['revision'][:12]}: {len(rows)} rows", flush=True)
    sample = stratified_sample(rows, args.n, args.seed)
    print(f">>> probing {len(sample)} rows, {collections.Counter(r['source'] for r in sample)}",
          flush=True)

    out_dir = Path(args.out or f"output/reasoning_backfill/{timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    client, usage = OpenRouterClient(), Usage()

    def one(idx_row: tuple[int, dict]) -> dict:
        i, row = idx_row
        split = split_row(row)
        if split is None:
            return {"i": i, "source": row.get("source"), "status": "unusable"}
        prompt, answer = split
        try:
            gen = generate_trace(prompt, key, args.reasoning_max_tokens)
        except Exception as e:  # noqa: BLE001 - a failed call is a datum, not a crash
            return {"i": i, "source": row.get("source"), "status": f"gen_error: {e}"[:200]}
        if not gen["trace"]:
            return {"i": i, "source": row.get("source"), "status": "no_trace", **gen}
        question = "\n\n".join(m.get("content") or "" for m in prompt)
        try:
            v = judge(client, usage, question, gen["trace"], answer)
        except Exception as e:  # noqa: BLE001
            return {"i": i, "source": row.get("source"), "status": f"judge_error: {e}"[:200], **gen}
        return {"i": i, "source": row.get("source"), "status": "ok", **gen, **v,
                # Kept for review: the row's reference answer, the model's own discarded
                # answer, and the prompt. A verdict cannot be audited without all three.
                "reference_answer": answer, "question": question,
                "answer_chars": len(answer), "prompt_chars": len(question)}

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(one, enumerate(sample)))
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")

    ok = [r for r in results if r["status"] == "ok"]
    # A trace the cap cut off is not a candidate for training, so acceptance is reported
    # over COMPLETE traces; the truncated share is reported separately as a budget fact.
    complete = [r for r in ok if r.get("finish_reason") != "length"]
    yes = [r for r in complete if r["verdict"] == "yes"]
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for r in complete:
        by_source[r["source"]].append(r)
    trace_toks = sorted(r["reasoning_tokens"] or 0 for r in ok)
    summary = {
        "mixture": args.mixture, "revision": ref["revision"], "n_sampled": len(sample),
        "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "provider_pin": provider_pin(GEN_MODEL), "seed": args.seed,
        "usable": len(ok), "complete_traces": len(complete), "accepted": len(yes),
        "acceptance_rate_of_complete": round(len(yes) / max(len(complete), 1), 4),
        "acceptance_rate_of_all": round(len(yes) / max(len(ok), 1), 4),
        "by_source": {s: {"n": len(v), "accepted": sum(x["verdict"] == "yes" for x in v),
                          "rate": round(sum(x["verdict"] == "yes" for x in v) / max(len(v), 1), 3)}
                      for s, v in sorted(by_source.items())},
        "status_counts": dict(collections.Counter(r["status"].split(":")[0] for r in results)),
        "trace_tokens": {
            "mean": round(sum(trace_toks) / max(len(trace_toks), 1)),
            "p50": trace_toks[len(trace_toks) // 2] if trace_toks else 0,
            "p90": trace_toks[int(0.9 * len(trace_toks))] if trace_toks else 0,
            "max": trace_toks[-1] if trace_toks else 0,
            # What a cap would keep, for sizing the sprinkle against the token budget.
            "share_under_400": round(sum(t <= 400 for t in trace_toks) / max(len(trace_toks), 1), 3),
            "share_under_800": round(sum(t <= 800 for t in trace_toks) / max(len(trace_toks), 1), 3),
        },
        "truncated_traces": len(ok) - len(complete),
        "judge_spend_usd": round(sum(m.get("usd", 0.0) for m in usage.by_model.values()), 4),
        "gen_spend_usd": round(sum(float(r.get("cost") or 0) for r in results), 4),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f">>> wrote {out_dir}/results.jsonl + summary.json", flush=True)


if __name__ == "__main__":
    main()
