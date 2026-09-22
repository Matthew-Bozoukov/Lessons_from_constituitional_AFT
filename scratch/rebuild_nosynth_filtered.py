# ABOUTME: Rebuild the nosynth base blend so every row passes the spec filter, at today's per-source
# ABOUTME: counts, keeping the Qwen traces surviving rows already carry; new rows get traces by the old rule.

"""One-off: a spec-filtered nosynth base that is otherwise the published one.

Run (from the repo root):

    uv run python scratch/rebuild_nosynth_filtered.py --smoke=True      # ~$1, no push
    uv run python scratch/rebuild_nosynth_filtered.py                   # judges + pushes

What it does, in order:

1. Loads `configs/data/mixture/nosynth.yaml`'s sources at `oversample` x their budgets with
   the config's seed — the same seeded per-source shuffle `uv run mix` takes its prefix
   from, so the published rows are the head of the pool and the extras are what the
   builder would have drawn next.
2. Fetches the published nosynth mixture (`old_repo`@`old_revision`) and keys every row by
   source + its messages with `reasoning_content` stripped, so a row is the same row with
   or without a trace.
3. Judges the published rows against `constitution` with `src.data.mixture.spec_filter`
   (the in-repo port of the August filter: same rubric, gpt-5.6-terra, temperature 0,
   reasoning_effort low, keep on any doubt). Then, per source, judges just enough unseen
   pool rows to replace the rejections (deficit x `margin` / observed keep rate), in
   rounds, until every source is back at its published count. Rounds checkpoint under
   `out/filter_round<k>/` and resume.
4. Surviving published rows keep their messages verbatim, traces included. New rows go
   through `reasoning_backfill` under the config's own block — 50% of each target source's
   NEW rows drawn by seed, Qwen writes, gemini-flash judges, rejects stay traceless — so
   the mixture's trace fraction is the published one applied to the replacements.
5. Writes mixture.jsonl + stats + merged verdicts/report + run_meta and pushes
   `<date>-nosynth-mix` with the builder's own card fields: a filtered base is the DEFAULT
   nosynth from 2026-09-22, so it keeps the plain name and the card says it was filtered.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mixture import build_mixture as bm  # noqa: E402
from src.data.mixture import reasoning_backfill as rb  # noqa: E402
from src.data.mixture.spec_filter import run_filter  # noqa: E402
from src.infra.huggingface import resolve_dataset  # noqa: E402
from src.model_profile import model_profile, render_chat  # noqa: E402
from src.naming import mix_name  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402

STRIP = ("n_tokens", "n_supervised", "balance_group")


def row_key(row: dict) -> str:
    """Source + messages without any trace: the row's identity across builds."""
    msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in row["messages"]]
    payload = json.dumps({"source": row["source"], "messages": msgs, "tools": row.get("tools")},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def traceless(row: dict) -> dict:
    """The row as the judge should see it: what the assistant SAID, not what it thought."""
    return {**row, "messages": [{k: v for k, v in m.items() if k != "reasoning_content"}
                                for m in row["messages"]]}


def judge(rows: list[dict], *, constitution_text: str, model: str, dest: Path, workers: int,
          limit: int | None) -> tuple[list[bool], dict]:
    dest.mkdir(parents=True, exist_ok=True)
    keep, report = run_filter([traceless(r) for r in rows], constitution_text=constitution_text,
                              model=model, dest=dest, workers=workers, limit=limit)
    print(f">>> {dest.name}: kept {report['samples_kept']:,}/{report['samples_in']:,} "
          f"({report['reject_rate_pct']}% rejected)", flush=True)
    return keep, report


def main(config: str = "configs/data/mixture/nosynth.yaml",
         old_repo: str = "dougalldeepmind/2026-09-08-nosynth-mix",
         old_revision: str = "7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd",
         constitution: str = "constitutions/claude_distilled_09_principles/constitution.md",
         judge_model: str = "openai/gpt-5.6-terra",
         oversample: float = 1.6, margin: float = 1.25, workers: int = 24,
         max_rounds: int = 4, smoke: bool = False, push: bool = True) -> None:
    cfg = OmegaConf.load(config)
    seed = int(cfg.seed)
    tok = bm.AutoTokenizer.from_pretrained(cfg.tokenizer)
    render_kwargs = model_profile(str(cfg.tokenizer)).render_kwargs
    out_dir = Path(cfg.output_dir) / (("smoke_" if smoke else "filtered_") + timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    constitution_text = Path(constitution).read_text(encoding="utf-8").strip()
    scale = 50 if smoke else 1

    # --- 1. the oversampled pool, in the builder's own seeded order --------------------
    specs = OmegaConf.to_container(cfg.sources, resolve=True)
    over = {name: {**spec, "examples": int(math.ceil(int(spec["examples"]) * oversample))}
            for name, spec in specs.items()}
    print(f">>> pool: {oversample}x the nosynth budgets, seed {seed}", flush=True)
    pool, kinds = bm._load_all(tok, cfg, over, scale, seed, render_kwargs)
    # Upstream sources carry exact duplicates (the builder does not dedupe either); the first
    # copy in the seeded order is the row, later copies can never be "new" rows.
    pool_by_key: dict[str, dict] = {}
    dupes: Counter = Counter()
    for r in pool:
        k = row_key(r)
        if k in pool_by_key:
            dupes[r["source"]] += 1
        else:
            pool_by_key[k] = r
    pool = list(pool_by_key.values())
    print(f">>> pool: {len(pool):,} distinct rows; duplicates dropped: {dict(dupes) or 'none'}", flush=True)

    # --- 2. the published rows, keyed the same way --------------------------------------
    path, _ = resolve_dataset(old_repo, "mixture.jsonl", old_revision)
    old = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    if smoke:
        old = old[:150]
    targets = Counter(r["source"] for r in old)
    old_keys = set()
    matched = 0
    for r in old:
        k = row_key(r)
        old_keys.add(k)
        if k in pool_by_key:
            r["n_tokens"] = pool_by_key[k]["n_tokens"]
            matched += 1
        else:
            # len() of the BatchEncoding is its key count, not the token count (build_mixture)
            r["n_tokens"] = len(render_chat(tok, traceless(r)["messages"], r.get("tools"),
                                            render_kwargs=render_kwargs, tokenize=True,
                                            return_dict=True)["input_ids"])
    print(f">>> published rows: {len(old):,}; {matched:,} found in the pool "
          f"({len(old) - matched:,} not — kept as candidates anyway)", flush=True)
    unseen = defaultdict(list)  # per source, pool rows that are not published rows, in pool order
    for r in pool:
        if row_key(r) not in old_keys:
            unseen[r["source"]].append(r)

    # --- 3. judge the published rows, then top up per source until the counts are back ---
    keep, report = judge(old, constitution_text=constitution_text, model=judge_model,
                         dest=out_dir / "filter_round1", workers=workers, limit=None)
    reports = [report]
    kept = defaultdict(list)
    for r, k in zip(old, keep):
        if k:
            kept[r["source"]].append(r)
    reject_rate = {s: max(0.02, 1 - report["by_source"][s]["kept"] / report["by_source"][s]["total"])
                   for s in targets if s in report["by_source"]}
    cursor = Counter()
    new_rows: list[dict] = []
    for rnd in range(2, max_rounds + 2):
        deficit = {s: targets[s] - len(kept[s]) for s in targets if targets[s] > len(kept[s])}
        if not deficit:
            break
        batch = []
        for s, d in sorted(deficit.items()):
            want = int(math.ceil(d * margin / (1 - reject_rate.get(s, 0.1))))
            avail = unseen[s][cursor[s]: cursor[s] + want]
            assert avail, (f"source {s!r}: pool exhausted with {d} rows still missing — raise "
                           f"--oversample (now {oversample})")
            cursor[s] += len(avail)
            batch += avail
        print(f">>> round {rnd}: deficit {dict(sorted(deficit.items()))} -> judging {len(batch):,} "
              f"unseen rows", flush=True)
        keep, report = judge(batch, constitution_text=constitution_text, model=judge_model,
                             dest=out_dir / f"filter_round{rnd}", workers=workers, limit=None)
        reports.append(report)
        for r, k in zip(batch, keep):
            s = r["source"]
            if k and len(kept[s]) < targets[s]:
                kept[s].append(r)
                new_rows.append(r)
        for s in deficit:
            bs = report["by_source"].get(s)
            if bs:
                reject_rate[s] = max(0.02, 1 - bs["kept"] / bs["total"])
    deficit = {s: targets[s] - len(kept[s]) for s in targets if targets[s] > len(kept[s])}
    assert not deficit, f"still short after {max_rounds} rounds: {deficit}"
    rows = [r for s in sorted(kept) for r in kept[s]]
    assert Counter(r["source"] for r in rows) == targets
    new_keys = {row_key(r) for r in new_rows}
    inherited = sum(1 for r in rows if row_key(r) not in new_keys
                    and any(str(m.get("reasoning_content") or "").strip() for m in r["messages"]))
    print(f">>> assembled {len(rows):,} rows: {len(rows) - len(new_rows):,} published "
          f"({inherited:,} with traces), {len(new_rows):,} new", flush=True)

    # --- 4. traces for the NEW rows only, by the config's own rule ----------------------
    sources_full = {**specs}
    spec = rb.validate_backfill(OmegaConf.to_container(cfg.reasoning_backfill, resolve=True),
                                sources_full)
    new_rows, backfill_report = rb.backfill(
        new_rows, spec, seed=seed, tok=tok, max_seq_len=int(cfg.max_seq_len),
        render_kwargs=render_kwargs, out_dir=out_dir / "backfill",
        workers=int(cfg.get("backfill_workers", 12)), limit=4 if smoke else 0)
    by_key = {row_key(r): r for r in new_rows}
    rows = [by_key[k] if (k := row_key(r)) in by_key else r for r in rows]
    random.Random(seed).shuffle(rows)
    traces = rb.traces_block(spec, rows, backfill_report, seed)
    traces["inherited_rows"] = inherited
    traces["inherited_from"] = f"{old_repo}@{old_revision}"
    traces["provenance"] = "reasoning_backfill (published rows inherited; new rows this build)"
    for name in spec["sources"]:
        traced = [any(m.get("reasoning_content") for m in r["messages"]) for r in rows if r["source"] == name]
        kinds[name] = "native" if traced and all(traced) else "mixed" if any(traced) else "none"

    # --- 5. write, record, push -------------------------------------------------------
    out_path = out_dir / "mixture.jsonl"
    bm._write_rows(out_path, rows)
    bm._validate_written(out_path, rows, kinds)
    verdicts = out_dir / "verdicts.jsonl"
    with verdicts.open("w", encoding="utf-8") as fh:
        for k in range(1, len(reports) + 1):
            for line in (out_dir / f"filter_round{k}" / "verdicts.jsonl").read_text(encoding="utf-8").splitlines():
                fh.write(json.dumps({"round": k, **json.loads(line)}, ensure_ascii=False) + "\n")
    merged = {"rounds": reports, "samples_in": sum(r["samples_in"] for r in reports),
              "samples_kept": sum(r["samples_kept"] for r in reports),
              "samples_rejected": sum(r["samples_rejected"] for r in reports),
              "unparsed_judge_replies": sum(r.get("unparsed_judge_replies", 0) for r in reports),
              "published_rows_rejected": reports[0]["samples_rejected"],
              "published_rows_kept": reports[0]["samples_kept"]}
    merged["reject_rate_pct"] = round(100 * merged["samples_rejected"] / merged["samples_in"], 2)
    (out_dir / "filter_report.json").write_text(json.dumps(merged, indent=2))
    stats = {"total": {"examples": len(rows), "tokens": sum(r["n_tokens"] for r in rows)},
             "synthetic_pct": 0, "share_unit": "examples", "by_source": bm._source_stats(rows),
             "sources": specs, "reasoning_traces": traces, "mixture_path": str(out_path),
             "filter": merged, "backfill": backfill_report,
             "rebuilt_from": {"repo": old_repo, "revision": old_revision, "rows_kept": len(rows) - len(new_rows),
                              "rows_replaced": len(new_rows)}}
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"command": " ".join(sys.argv), "smoke": smoke, "stats": stats})
    print(json.dumps({"rows": len(rows), "replaced": len(new_rows), "inherited_traces": inherited,
                      "new_traces": backfill_report["n_accepted"], "filter": merged["reject_rate_pct"],
                      "backfill_spend_usd": backfill_report.get("spend_usd")}, indent=2))
    print(f">>> wrote {out_path}")
    if not push:
        return
    filter_cfg = OmegaConf.create({"constitution": constitution, "model": judge_model})
    card_cfg = OmegaConf.merge(cfg, OmegaConf.create({"filter": {"constitution": constitution, "model": judge_model}, "hf": {
        "experiment": ("MSM Table 2 instruction-tuning blend, no synthetic share, every row passed the spec "
                       "filter — the published nosynth base with its rejected rows replaced at the same "
                       "per-source counts; surviving rows keep their Qwen3.6 traces")}}))
    fields = bm._card_fields(card_cfg, config, "final training mixture (spec-filtered base, traces inherited)",
                             "mixture.jsonl + mixture_stats.json + verdicts.jsonl + filter_report.json",
                             filter_cfg, merged, traces)
    fields["provenance"] = f"uv run python scratch/rebuild_nosynth_filtered.py --old_repo {old_repo} --old_revision {old_revision}"
    fm = bm._front_matter(card_cfg, config, filter_cfg, "final", out_path.name)
    OmegaConf.save(card_cfg, out_dir / "mixture_config.yaml")
    repo = mix_name("", 0)
    bm._push([out_path, out_dir / "mixture_stats.json", out_dir / "run_meta.json", out_dir / "mixture_config.yaml",
              verdicts, out_dir / "filter_report.json"], repo, fields, False, smoke, fm)


if __name__ == "__main__":
    fire.Fire(main)
