# ABOUTME: Add on-policy reasoning traces to half the tulu3_if / self_oss_instruct / lima rows
# ABOUTME: of the nosynth mixture and republish it as `<date>-nosynth-mix`, same 10,000 rows.
"""Enrich the base blend with reasoning, without changing what it contains.

WHY. The nosynth blend has no reasoning anywhere, so under the generation-boundary rule
(src/train/masking.py) a model trained on it NEVER practises writing `</think>` -- the
suspected cause of the unclosed-think-block failure on MASK (docs/LOG.md 2026-09-07).
Three of its sources are ones where deliberation genuinely precedes the answer:
tulu3_if and self_oss_instruct (constraint-checking and code planning) and lima.

WHAT. For half the rows of those three sources, ask the model that will be TRAINED
(qwen3.6-27b, on-policy) to answer the row's own prompt with reasoning on; keep the trace,
DISCARD the model's answer, and attach the trace to the corpus answer that was already
there. A judge then decides whether that trace could plausibly precede that answer, and
only accepted traces are spliced in. So no answer in the blend changes, no row is added or
dropped, and the only edit is a `reasoning_content` field on some assistant turns.

WHAT IS NOT CHANGED, deliberately:
  * Row count, order, per-source counts, and every `content` string.
  * `supervise`: absent on every row of this blend, which means "all", and the 2841 of
    2849 candidate rows that have ONE assistant turn are unaffected by the distinction.
    For the 8 multi-turn rows the trace goes on the FINAL assistant turn and the earlier
    turns render an empty think marker, which the generation-boundary rule masks whole --
    so `all` stays correct and setting `final` would silently drop supervision the old
    mixture had.
  * A row whose rendered length would exceed the blend's `max_seq_len` WITH the trace is
    left unenriched: the trainer would cut it, and a cut row loses its answer.

Run:
    uv run python scratch/reasoning_backfill/enrich_mixture.py            # generate + push
    uv run python scratch/reasoning_backfill/enrich_mixture.py --limit 20 --no-push
    uv run python scratch/reasoning_backfill/enrich_mixture.py --resume <out_dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

from scratch.reasoning_backfill.backfill_probe import (
    GEN_MODEL, JUDGE_MODEL, generate_trace, judge, split_row)
from src.data.mixture.build_mixture import _card_fields, _front_matter
from src.data.synth.ours.stage_runtime import Usage
from src.infra.endpoints.openrouter import OpenRouterClient
from src.infra.huggingface import hf_repo_id, push_files, resolve_dataset
from src.model_profile import model_profile, render_chat
from src.naming import mix_name
from src.utils import timestamp

SOURCE_MIX = "LASR-Callum/2026-09-05-nosynth-mix"
CONFIG = "configs/data/mixture/nosynth.yaml"
TARGET_SOURCES = ("tulu3_if", "self_oss_instruct", "lima")
FRACTION = 0.5
# No reasoning budget: `reasoning.max_tokens` on the pinned endpoint is a HARD CUT at the
# requested value, not a shorter plan (docs/LOG.md 2026-09-07 correction), and a trace cut
# mid-thought would teach reasoning that stops. The natural length is what gets trained,
# so the only cap is an overall ceiling sized to let the trace finish.
COMPLETION_MAX_TOKENS = 6144
_LOCK = threading.Lock()


def select(rows: list[dict], seed: int, limit: int, mode: str) -> list[tuple[int, int]]:
    """Which (row, assistant-turn) pairs to enrich.

    `sample`    — `FRACTION` of each target source, seeded per source, FINAL turn only.
                  A single-turn row has exactly one assistant turn, so for the 2841 of
                  2849 candidate rows that are single-turn this is the whole row.
    `multiturn` — every assistant turn of every MULTI-turn row in the target sources that
                  does not already carry a trace. All 8 such rows are lima; a partly
                  enriched row (final turn done by a `sample` run) keeps what it has and
                  gains the rest, so the two modes compose.
    """
    if mode == "multiturn":
        out = []
        for i, r in enumerate(rows):
            if r.get("source") not in TARGET_SOURCES:
                continue
            turns = [j for j, m in enumerate(r["messages"]) if m["role"] == "assistant"]
            if len(turns) < 2:
                continue
            out += [(i, j) for j in turns
                    if not str(r["messages"][j].get("reasoning_content") or "").strip()]
        return out[:limit] if limit else out

    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("source") in TARGET_SOURCES:
            by_source[r["source"]].append(i)
    picked: list[int] = []
    for s_name in sorted(by_source):
        idx = by_source[s_name]
        picked += random.Random(f"{seed}:{s_name}").sample(idx, round(len(idx) * FRACTION))
    picked.sort()
    picked = picked[:limit] if limit else picked
    return [(i, max(j for j, m in enumerate(rows[i]["messages"]) if m["role"] == "assistant"))
            for i in picked]


def generate_one(i: int, j: int, row: dict, key: str, client, usage, sink) -> dict:
    """One row: trace, judge, verdict. Retries a rate limit rather than losing the row."""
    # The prompt is everything BEFORE this assistant turn, so a mid-conversation trace is
    # conditioned on exactly what the model would have seen when it wrote that reply.
    prompt = [{k: v for k, v in m.items() if k in ("role", "content")}
              for m in row["messages"][:j]]
    answer = (row["messages"][j].get("content") or "").strip()
    if not prompt or not answer:
        return sink({"i": i, "j": j, "source": row.get("source"), "status": "unusable"})
    gen = None
    for attempt in range(5):
        try:
            gen = generate_trace(prompt, key, 0, timeout=600)
            break
        except requests.HTTPError as e:  # 429 is expected: one pinned provider, N workers
            if e.response is not None and e.response.status_code == 429 and attempt < 4:
                time.sleep(2 ** attempt + random.random() * 3)
                continue
            return sink({"i": i, "j": j, "source": row.get("source"), "status": f"gen_error: {e}"[:200]})
        except Exception as e:  # noqa: BLE001
            return sink({"i": i, "j": j, "source": row.get("source"), "status": f"gen_error: {e}"[:200]})
    if not gen or not gen["trace"]:
        return sink({"i": i, "j": j, "source": row.get("source"), "status": "no_trace", **(gen or {})})
    question = "\n\n".join(m.get("content") or "" for m in prompt)
    try:
        v = judge(client, usage, question, gen["trace"], answer)
    except Exception as e:  # noqa: BLE001
        return sink({"i": i, "j": j, "source": row.get("source"), "status": f"judge_error: {e}"[:200], **gen})
    return sink({"i": i, "j": j, "source": row.get("source"), "status": "ok", **gen, **v})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only this many rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="reuse the generations.jsonl in this dir")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--mode", choices=("sample", "multiturn"), default="sample",
                    help="sample: 50%% of each target source, final turn. "
                         "multiturn: every untraced assistant turn of multi-turn rows.")
    ap.add_argument("--base", default=SOURCE_MIX,
                    help="mixture to enrich: an HF repo id, or a local mixture.jsonl "
                         "(use the already-enriched file to add to it)")
    args = ap.parse_args()
    load_dotenv()
    key = os.environ["OPENROUTER_API_KEY"]

    cfg = OmegaConf.load(CONFIG)
    if Path(args.base).is_file():
        path, ref = Path(args.base), {"revision": f"local:{args.base}"}
    else:
        path, ref = resolve_dataset(args.base, "mixture.jsonl")
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    print(f">>> {args.base}@{str(ref['revision'])[:20]}: {len(rows)} rows "
          f"({sum(1 for r in rows for m in r['messages'] if m.get('reasoning_content'))} "
          "turns already traced)", flush=True)

    picked = select(rows, args.seed, args.limit, args.mode)
    print(f">>> mode {args.mode}: enriching {len(picked)} assistant turns across "
          f"{len({i for i, _ in picked})} rows: "
          f"{collections.Counter(rows[i]['source'] for i, _ in picked)}", flush=True)

    out_dir = Path(args.resume or args.out or f"output/reasoning_backfill/enrich_{timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "generations.jsonl"
    done: dict[tuple[int, int], dict] = {}
    if cache_path.is_file():
        for line in cache_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[(r["i"], r["j"])] = r
        print(f">>> resuming: {len(done)} rows already generated", flush=True)

    client, usage = OpenRouterClient(), Usage()
    fh = cache_path.open("a", encoding="utf-8")

    def sink(rec: dict) -> dict:
        with _LOCK:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        return rec

    todo = [ij for ij in picked if ij not in done]
    t0 = time.time()
    if todo:
        with ThreadPoolExecutor(args.workers) as ex:
            for n, rec in enumerate(ex.map(
                    lambda ij: generate_one(ij[0], ij[1], rows[ij[0]], key, client, usage,
                                            sink), todo), 1):
                done[(rec["i"], rec["j"])] = rec
                if n % 50 == 0:
                    print(f"    {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    fh.close()
    print(f">>> generation done in {time.time() - t0:.0f}s", flush=True)

    # --- splice ---------------------------------------------------------------------
    # A trace is accepted only if it is COMPLETE (a cut trace teaches reasoning that
    # stops), the judge says it fits the answer, and the row still fits max_seq_len with
    # it. Everything else leaves the row exactly as it was.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer))
    render_kwargs = model_profile(str(cfg.tokenizer)).render_kwargs
    cap = int(cfg.max_seq_len)
    stats = collections.Counter()
    per_source = collections.defaultdict(collections.Counter)
    # Grouped by row: several traces can land on one row, and the length check has to see
    # the row as it will actually be trained, not one turn at a time.
    by_row: dict[int, list[tuple[int, dict]]] = collections.defaultdict(list)
    for (i, j) in picked:
        by_row[i].append((j, done.get((i, j)) or {}))
    for i, turns in sorted(by_row.items()):
        src = rows[i]["source"]
        msgs = [dict(m) for m in rows[i]["messages"]]
        for j, r in sorted(turns):
            per_source[src]["attempted"] += 1
            if r.get("status") != "ok":
                stats[str(r.get("status", "missing")).split(":")[0]] += 1
                per_source[src]["failed"] += 1
                continue
            if r.get("finish_reason") == "length":
                stats["truncated"] += 1
                per_source[src]["truncated"] += 1
                continue
            if r.get("verdict") != "yes":
                stats["judge_rejected"] += 1
                per_source[src]["judge_rejected"] += 1
                continue
            trial = [dict(m) for m in msgs]
            trial[j]["reasoning_content"] = r["trace"]
            n = len(render_chat(tok, trial, rows[i].get("tools"), render_kwargs=render_kwargs,
                                tokenize=True, return_dict=True)["input_ids"])
            if n > cap:
                stats["over_max_seq_len"] += 1
                per_source[src]["over_max_seq_len"] += 1
                continue
            msgs = trial
            stats["accepted"] += 1
            per_source[src]["accepted"] += 1
        rows[i]["messages"] = msgs

    enriched = sum(1 for r in rows for m in r["messages"]
                   if str(m.get("reasoning_content") or "").strip())
    print(f"\n>>> spliced {stats['accepted']} traces "
          f"({stats['accepted'] / max(len(picked), 1):.1%} of attempted turns)")
    print(f">>> outcomes: {dict(stats)}")
    for s in sorted(per_source):
        c = per_source[s]
        print(f"    {s:22} {c['accepted']}/{c['attempted']} accepted "
              f"({c['accepted'] / max(c['attempted'], 1):.1%})")

    # --- write ----------------------------------------------------------------------
    mix_path = out_dir / "mixture.jsonl"
    with mix_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert sum(1 for _ in mix_path.open(encoding="utf-8")) == len(rows)
    report = {
        "source_mixture": {"repo": SOURCE_MIX, "file": "mixture.jsonl",
                           "revision": ref["revision"]},
        "target_sources": list(TARGET_SOURCES), "fraction_attempted": FRACTION,
        "seed": args.seed, "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
        "completion_max_tokens": COMPLETION_MAX_TOKENS, "reasoning_budget": None,
        "mode": args.mode, "base_mixture": args.base,
        "n_rows": len(rows), "n_attempted_turns": len(picked),
        "n_accepted": stats["accepted"], "n_reasoning_turns": enriched,
        "acceptance_rate_of_attempted": round(stats["accepted"] / max(len(picked), 1), 4),
        "outcomes": dict(stats),
        # A trace not produced by GEN_MODEL must say so: the card claims on-policy traces,
        # and one exception silently folded in would make that claim false.
        "hand_authored": [{"i": r["i"], "j": r["j"], "note": r["authored"]}
                          for r in done.values() if r.get("authored")],
        "by_source": {s: dict(c) for s, c in per_source.items()},
        "spend_usd": round(sum(r.get("cost") or 0 for r in done.values()), 4),
        "judge_spend_usd": round(usage.total_cost(), 4) if hasattr(usage, "total_cost") else None,
    }
    (out_dir / "enrichment_report.json").write_text(json.dumps(report, indent=2))
    print(f">>> wrote {mix_path} + enrichment_report.json")

    if args.no_push:
        print(">>> --no-push: not publishing")
        return
    # empty styles + a 0% share IS the base blend, and mix_name spells it `nosynth`.
    repo = hf_repo_id(mix_name("", 0))
    fields = _card_fields(
        cfg, CONFIG,
        stage_desc=(
            f"reasoning-enriched: on-policy {GEN_MODEL} traces, judged for consistency "
            f"with the answer already in the row. This push added {stats['accepted']} of "
            f"{len(picked)} attempted assistant turns ("
            + ("50% of each of " if args.mode == "sample"
               else "every untraced turn of the multi-turn rows of ")
            + f"{', '.join(TARGET_SOURCES)}, seed {args.seed}); the mixture now carries "
            f"{enriched} traced turns. Answers, row count and per-source counts unchanged"),
        files_desc="mixture.jsonl = the 10,000 training rows; enrichment_report.json = "
                   "what was attempted and what was accepted",
        filter_cfg=None, report=None)
    fields["generation_config"] = json.dumps(report)
    fields["provenance"] = "uv run python scratch/reasoning_backfill/enrich_mixture.py"
    url = push_files([mix_path, out_dir / "enrichment_report.json"], repo, fields,
                     private=bool(cfg.hf.get("private", False)),
                     front_matter=_front_matter(cfg, CONFIG, None, "final", "mixture.jsonl"))
    print(f">>> pushed -> {url}")


if __name__ == "__main__":
    main()
