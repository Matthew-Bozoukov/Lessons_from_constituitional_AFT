# ABOUTME: On-policy reasoning traces for the base blend: the family to be trained writes a trace for
# ABOUTME: a row's own prompt, a judge checks it fits the row's answer, and the mixture records whose it is.
"""Why the base blend carries traces, and why that makes it model-specific.

The nosynth blend has no reasoning anywhere, so under the generation-boundary rule
(src/train/masking.py) a model trained on it never practises closing a think block -- the
suspected cause of the unclosed-think failure on MASK (docs/LOG.md 2026-09-07). The fix
(2026-09-08) asked the model that will be TRAINED to answer a fraction of the rows of the
three sources where deliberation genuinely precedes the answer (tulu3_if, self_oss_instruct,
lima), kept its trace, discarded its answer, and spliced the trace onto the answer already
in the row when a judge agreed the trace could precede that answer. No answer, row count or
order changes; the only edit is a `reasoning_content` field on some assistant turns.

That makes the base blend the property of ONE model family: its traces are on-policy for the
model that wrote them and off-policy for any other. Until 2026-09-13 the only record of this
was a sentence in the mixture card and a scratch script; the config that every arm inherits
said `reasoning: none` on every source. Now:

  * `reasoning_backfill:` in the base blend's config names the generator (`model`), and
    `uv run mix --config configs/data/mixture/nosynth.yaml` reproduces the enriched blend;
  * the built mixture carries a `reasoning_traces` block (model, FAMILY, judge, counts) in
    mixture_stats.json and names the generator in its card's `models` field;
  * an arm mixture built on a published base (`base_mixture:`) copies that block forward, so
    no arm config names a model -- it points at a base and inherits whose traces it holds;
  * `uv run train` refuses a mixture whose trace family is not the model being trained
    (src/train/launch.py check_trace_family) unless `allow_trace_family_mismatch=true`.

A new base model therefore needs its own base blend: change `reasoning_backfill.model` (and
`tokenizer`) and rebuild; every arm config then points at the new base repo.
"""
from __future__ import annotations

import collections
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub.utils import EntryNotFoundError

from src.infra.endpoints.openrouter import OpenRouterClient, provider_pin
from src.model_profile import find_profile, render_chat

BACKFILL_KEYS = {"model", "judge", "sources", "fraction", "max_tokens"}

JUDGE_PROMPT = """A model was asked the question below and produced the REASONING shown.
Separately, the ANSWER shown is the reference answer for that question.

Decide ONE thing: could that reasoning plausibly precede that answer, or does it point
somewhere else? Say NO if the reasoning reaches a different conclusion, contradicts the
answer, or plans a materially different response. Say YES if the answer is a natural
continuation of the reasoning, even if the reasoning is longer or covers more ground.

<question>
{question}
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Reply with YES or NO, then one sentence of justification."""

_LOCK = threading.Lock()


def family_of(model: str) -> str:
    """The registered model-profile key a trace generator's id belongs to.

    `qwen/qwen3.6-27b` (the OpenRouter id) and `Qwen/Qwen3.6-27B` (the HF id the trainer
    loads) both squeeze to a string containing `qwen36`, so the family the traces are
    on-policy for and the family being trained are compared through the one registry.
    """
    profile = find_profile(model)
    if profile is None:
        raise ValueError(
            f"reasoning traces are attributed to {model!r}, which no model profile in "
            "configs/models/ identifies (`match:`). Register the family before generating "
            "traces for it: the mixture must be able to say whose traces it carries.")
    return profile.key


def validate_backfill(cfg_block: dict, sources: dict) -> dict:
    """Reject a malformed `reasoning_backfill:` block before any spend."""
    if not isinstance(cfg_block, dict) or set(cfg_block) != BACKFILL_KEYS:
        raise ValueError(f"reasoning_backfill takes exactly {sorted(BACKFILL_KEYS)}")
    family_of(str(cfg_block["model"]))
    if not str(cfg_block["judge"]).strip():
        raise ValueError("reasoning_backfill.judge must name the judge model")
    targets = list(cfg_block["sources"])
    unknown = sorted(set(targets) - set(sources))
    if not targets or unknown:
        raise ValueError(f"reasoning_backfill.sources must name sources of this config; unknown: {unknown}")
    fraction = float(cfg_block["fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("reasoning_backfill.fraction must be in (0, 1]")
    if int(cfg_block["max_tokens"]) < 1024:
        raise ValueError("reasoning_backfill.max_tokens must be >= 1024 (a trace plus the answer it discards)")
    return {"model": str(cfg_block["model"]), "judge": str(cfg_block["judge"]), "sources": targets,
            "fraction": fraction, "max_tokens": int(cfg_block["max_tokens"])}


def select(rows: list[dict], targets: list[str], fraction: float, seed: int) -> list[tuple[int, int]]:
    """(row index, assistant-turn index) pairs to trace: `fraction` of each target source's
    rows, drawn per source from `seed`, and every untraced assistant turn of a drawn row.

    A single-turn row has one assistant turn, so for almost every row this is the whole
    row; a multi-turn row gets a trace on each of its turns, because a turn without one
    renders an empty think marker that the generation-boundary rule masks whole.
    """
    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("source") in targets:
            by_source[r["source"]].append(i)
    picked: list[int] = []
    for name in sorted(by_source):
        idx = by_source[name]
        picked += random.Random(f"{seed}:{name}").sample(idx, round(len(idx) * fraction))
    out = []
    for i in sorted(picked):
        for j, m in enumerate(rows[i]["messages"]):
            if m.get("role") == "assistant" and not str(m.get("reasoning_content") or "").strip():
                out.append((i, j))
    return out


def prompt_and_answer(row: dict, turn: int) -> tuple[list[dict], str] | None:
    """The conversation up to `turn` (roles and content only) and that turn's answer."""
    msgs = row.get("messages") or []
    if turn < 1 or turn >= len(msgs) or msgs[turn].get("role") != "assistant":
        return None
    answer = (msgs[turn].get("content") or "").strip()
    prompt = [{k: v for k, v in m.items() if k in ("role", "content")} for m in msgs[:turn]]
    if not answer or not prompt:
        return None
    return prompt, answer


def generate_trace(client: OpenRouterClient, model: str, prompt: list[dict], max_tokens: int) -> dict:
    """One reasoning-on completion of the row's own prompt. The answer is discarded."""
    res = client.chat(model=model, messages=prompt, temperature=1.0, max_tokens=max_tokens,
                      extra_body={"reasoning": {"enabled": True, "exclude": False},
                                  "provider": provider_pin(model)})
    return {"trace": (res.reasoning_content or "").strip(), "finish_reason": res.finish_reason,
            "completion_tokens": res.completion_tokens, "cost": res.cost, "provider": res.provider}


def judge_trace(client: OpenRouterClient, judge: str, question: str, trace: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question[:6000], reasoning=trace[:6000], answer=answer[:6000])
    res = client.chat(model=judge, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
    text = (res.content or "").strip()
    verdict = "yes" if text.upper().startswith("YES") else "no" if text.upper().startswith("NO") else "?"
    return {"verdict": verdict, "why": text[:160], "judge_cost": res.cost}


def _generate_one(client, spec: dict, i: int, j: int, row: dict) -> dict:
    split = prompt_and_answer(row, j)
    if split is None:
        return {"i": i, "j": j, "source": row.get("source"), "status": "unusable"}
    prompt, answer = split
    try:
        gen = generate_trace(client, spec["model"], prompt, spec["max_tokens"])
    except Exception as exc:  # the client has already retried transient failures
        return {"i": i, "j": j, "source": row.get("source"), "status": f"gen_error: {type(exc).__name__}"}
    if not gen["trace"]:
        return {"i": i, "j": j, "source": row.get("source"), "status": "no_trace", **gen}
    if gen["finish_reason"] == "length":
        return {"i": i, "j": j, "source": row.get("source"), "status": "truncated", **gen}
    question = "\n\n".join(f"[{m['role']}] {m['content']}" for m in prompt)
    try:
        verdict = judge_trace(client, spec["judge"], question, gen["trace"], answer)
    except Exception as exc:
        return {"i": i, "j": j, "source": row.get("source"), "status": f"judge_error: {type(exc).__name__}", **gen}
    return {"i": i, "j": j, "source": row.get("source"), "status": "ok", **gen, **verdict}


def splice(rows: list[dict], picked: list[tuple[int, int]], generated: dict[tuple[int, int], dict],
           tok, max_seq_len: int, render_kwargs: dict) -> tuple[list[dict], dict]:
    """Attach every accepted trace; a rejected one leaves its row exactly as it was.

    Accepted = complete (a cut trace teaches reasoning that stops), judged YES, and the row
    still fits `max_seq_len` with it (the trainer would cut it, and a cut row loses its
    answer). Rows are copied; the input list is not mutated.
    """
    out = [dict(r) for r in rows]
    stats: collections.Counter = collections.Counter()
    per_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_row: dict[int, list[int]] = collections.defaultdict(list)
    for i, j in picked:
        by_row[i].append(j)
    for i, turns in sorted(by_row.items()):
        src = rows[i].get("source", "?")
        msgs = [dict(m) for m in rows[i]["messages"]]
        for j in sorted(turns):
            rec = generated.get((i, j)) or {}
            per_source[src]["attempted"] += 1
            status = str(rec.get("status", "missing"))
            if status != "ok":
                stats[status.split(":")[0]] += 1
                per_source[src]["failed"] += 1
                continue
            if rec.get("verdict") != "yes":
                stats["judge_rejected"] += 1
                per_source[src]["judge_rejected"] += 1
                continue
            trial = [dict(m) for m in msgs]
            trial[j]["reasoning_content"] = rec["trace"]
            n = len(render_chat(tok, trial, rows[i].get("tools"), render_kwargs=render_kwargs,
                                tokenize=True, return_dict=True)["input_ids"])
            if n > max_seq_len:
                stats["over_max_seq_len"] += 1
                per_source[src]["over_max_seq_len"] += 1
                continue
            msgs = trial
            stats["accepted"] += 1
            per_source[src]["accepted"] += 1
        out[i]["messages"] = msgs
    report = {"n_attempted_turns": len(picked), "n_accepted": stats["accepted"],
              "outcomes": dict(stats), "by_source": {s: dict(c) for s, c in per_source.items()}}
    return out, report


def traces_block(spec: dict, rows: list[dict], report: dict, seed: int) -> dict:
    """The `reasoning_traces` record a mixture carries: whose traces, and how many."""
    turns = sum(1 for r in rows for m in r["messages"] if str(m.get("reasoning_content") or "").strip())
    traced_rows = sum(1 for r in rows if any(str(m.get("reasoning_content") or "").strip() for m in r["messages"]))
    return {"model": spec["model"], "family": family_of(spec["model"]), "judge": spec["judge"],
            "sources": list(spec["sources"]), "fraction": spec["fraction"], "seed": seed,
            "rows": traced_rows, "turns": turns, "attempted_turns": report["n_attempted_turns"],
            "accepted_turns": report["n_accepted"], "provenance": "reasoning_backfill (this build)"}


def backfill(rows: list[dict], spec: dict, *, seed: int, tok, max_seq_len: int, render_kwargs: dict,
             out_dir: Path, workers: int = 12, limit: int = 0, client=None) -> tuple[list[dict], dict]:
    """Generate, judge and splice traces for `spec` over `rows`; resumable from `out_dir`.

    Every generation is appended to `<out_dir>/backfill_generations.jsonl` as it lands, so
    a rerun in the same directory judges and splices what is there and generates only the
    rest. `limit` caps the attempted turns (a --smoke run proves the wiring for cents).
    """
    picked = select(rows, spec["sources"], spec["fraction"], seed)
    if limit:
        picked = picked[:limit]
    print(f">>> reasoning backfill: {spec['model']} (family {family_of(spec['model'])}) on "
          f"{len(picked)} assistant turns of {len({i for i, _ in picked})} rows "
          f"({collections.Counter(rows[i]['source'] for i, _ in picked)})", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "backfill_generations.jsonl"
    done: dict[tuple[int, int], dict] = {}
    if cache.is_file():
        for line in cache.open(encoding="utf-8"):
            rec = json.loads(line)
            done[(rec["i"], rec["j"])] = rec
        print(f">>> resuming: {len(done)} turns already generated", flush=True)
    todo = [ij for ij in picked if ij not in done]
    client = client or OpenRouterClient()
    t0 = time.time()
    if todo:
        with cache.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
            for n, rec in enumerate(ex.map(lambda ij: _generate_one(client, spec, ij[0], ij[1], rows[ij[0]]), todo), 1):
                with _LOCK:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                done[(rec["i"], rec["j"])] = rec
                if n % 50 == 0:
                    print(f"    {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    rows, report = splice(rows, picked, done, tok, max_seq_len, render_kwargs)
    report["spend_usd"] = round(sum((r.get("cost") or 0) + (r.get("judge_cost") or 0) for r in done.values()), 4)
    report["seconds"] = round(time.time() - t0, 1)
    print(f">>> reasoning backfill: spliced {report['n_accepted']}/{len(picked)} traces; "
          f"outcomes {report['outcomes']}; ${report['spend_usd']}", flush=True)
    (out_dir / "backfill_report.json").write_text(json.dumps(report, indent=2))
    return rows, report


def _fetch_json(repo: str, filename: str, revision: str) -> dict | None:
    from src.infra.huggingface import hf_download

    try:
        path = hf_download(repo, filename, repo_type="dataset", revision=revision)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_reasoning_traces(repo: str, revision: str) -> dict | None:
    """Whose traces a published base blend carries, read from the base itself.

    A base built by this module says so in its `mixture_stats.json`. The one base built
    before the record existed (`2026-09-08-nosynth-mix`, the scratch backfill) says so only
    in its `enrichment_report.json`, which is read as the fallback so the arms pinned to it
    still inherit the family. A base with neither carries no traces and no claim: None.
    """
    stats = _fetch_json(repo, "mixture_stats.json", revision)
    if stats and stats.get("reasoning_traces"):
        return dict(stats["reasoning_traces"])
    report = _fetch_json(repo, "enrichment_report.json", revision)
    if report and report.get("gen_model"):
        return {"model": str(report["gen_model"]), "family": family_of(str(report["gen_model"])),
                "judge": report.get("judge_model"), "sources": list(report.get("target_sources") or []),
                "fraction": report.get("fraction_attempted"), "seed": report.get("seed"),
                "rows": None, "turns": report.get("n_reasoning_turns"),
                "provenance": f"enrichment_report.json of {repo}@{revision[:12]} (scratch/reasoning_backfill, pre-record)"}
    return None


def inherited_block(traces: dict, repo: str, revision: str) -> dict:
    """The block an arm mixture records for traces it inherits from its base."""
    return {**traces, "inherited_from": {"repo": repo, "revision": revision}}


def describe(traces: dict | None) -> str:
    """The card's `models` phrase for a mixture's base-blend traces."""
    if not traces:
        return "base-blend reasoning traces: none"
    turns = traces.get("turns")
    return (f"base-blend reasoning traces: on-policy {traces['model']} (family {traces['family']}"
            + (f", {turns} turns" if turns is not None else "")
            + (f", judged by {traces['judge']}" if traces.get("judge") else "") + ")")
