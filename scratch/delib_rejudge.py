# ABOUTME: Re-judge an existing deliberative-SFT run's Qwen candidates with the CURRENT config's
# ABOUTME: judge, reusing every generation and paying only for the judge; writes a sibling run dir.

"""Re-judge a deliberative-alignment run without regenerating.

Run: uv run python scratch/delib_rejudge.py --run output/synth_delib/<run> \\
         [--config configs/data/synth/delib.yaml] [--out <dir>] [--dry-run]

The pipeline's resume path refuses a judge change (every data-affecting setting must match),
which is right for a resume but makes each judge iteration on the full corpus cost the Qwen
generation again (~$22). This driver takes the run's checkpointed candidates as given --
generations, format verdicts and the prompt snapshot, all verified against the manifest --
and re-scores them with the judge, prompt and threshold the config declares NOW, using the
pipeline's own `_judge`, `_survivors`, `_judge_stage` and `export_row` so the records and
selection rule are identical to a fresh run's. Nothing is pushed to the Hub: a rejudged FULL
run that is worth training on must go through the entrypoint's publication contract (or a
repack that reproduces it); a rejudged smoke is a local read.

The constitution the judge grades against must be the one the candidates were generated
against (same sha256 as the run manifest), because the generation prompt told the model to
reason from it; `--allow-constitution-change` overrides that for a deliberate experiment.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synth.deliberative_alignment import pipeline as dp  # noqa: E402
from src.data.synth.deliberative_alignment.data import export_row  # noqa: E402
from src.data.synth.ours.constitution import full_text  # noqa: E402
from src.infra.endpoints.openrouter import OpenRouterClient, provider_price  # noqa: E402
from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an existing run dir with generations.partial.jsonl")
    ap.add_argument("--config", default="configs/data/synth/delib.yaml")
    ap.add_argument("--out", default=None, help="output dir (default: <run>_rejudge_<timestamp>)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be judged; no calls")
    ap.add_argument("--constitution", default=None,
                    help="grade against THIS file instead of the config's (e.g. the exact text the "
                         "run generated against, recovered from git); must match the manifest sha")
    ap.add_argument("--allow-constitution-change", action="store_true")
    ap.add_argument("--judge-effort", default=None, choices=["low", "medium", "high", "none"],
                    help="override filter.judge.reasoning: an effort level, or `none` to drop it")
    args = ap.parse_args()

    run_dir = Path(args.run)
    manifest_in = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["pipeline"] = Path(args.config).stem
    if args.judge_effort == "none":
        cfg["filter"]["judge"].pop("reasoning", None)
    elif args.judge_effort:
        cfg["filter"]["judge"]["reasoning"] = {"effort": args.judge_effort}
    dp.validate_config(cfg)
    flt = cfg["filter"]
    judge = flt["judge"]
    runs = int(judge["runs"])
    judge_pin = dp._priced(judge["model"])
    judge_price = provider_price(judge["model"]) or {}

    constitution_path = args.constitution or cfg["constitution"]
    constitution = full_text(constitution_path)
    # The pipeline has hashed the stripped text since 2026-09-10 and the raw file before;
    # accept either so a pre-change run's candidates can be matched to their exact text.
    raw = Path(constitution_path).read_text(encoding="utf-8")
    shas = {hashlib.sha256(constitution.encode()).hexdigest(), hashlib.sha256(raw.encode()).hexdigest()}
    sha = hashlib.sha256(constitution.encode()).hexdigest()
    if manifest_in["constitution_sha256"] not in shas:
        msg = (f"constitution {constitution_path} (sha {sha[:12]}) differs from the one the "
               f"candidates were generated against ({manifest_in['constitution_sha256'][:12]})")
        if not args.allow_constitution_change:
            raise SystemExit(msg + "; pass --allow-constitution-change for a deliberate experiment")
        print("WARNING:", msg)

    records = read_jsonl(run_dir / "stage_1_prompts.jsonl")
    if dp._digest(records) != manifest_in["prompts_sha256"]:
        raise SystemExit("the run's prompt snapshot does not match its manifest")
    # Resume semantics: a candidate is settled by its LAST record (an error retried later).
    settled: dict[tuple[str, int], dict] = {}
    for a in read_jsonl(run_dir / "generations.partial.jsonl"):
        if "error" not in a:
            settled[(a["id"], a["candidate"])] = a
    attempts = list(settled.values())
    # Re-apply the CURRENT format gate: a candidate the old gate let through but the new one
    # rejects is treated as format-rejected here (recorded in the manifest), never judged.
    from src.data.synth.deliberative_alignment.judge import format_rejection, leak_pattern
    leak = leak_pattern(constitution)
    regated = 0
    for a in attempts:
        if "assistant" in a and (why := format_rejection(a["assistant"].get("content") or "", leak)):
            a["rejected_by_current_gate"] = why; regated += 1
    ok = {(a["id"], a["candidate"]): a for a in attempts if "assistant" in a and "rejected_by_current_gate" not in a}
    if regated:
        print(f">>> current format gate rejects {regated} candidate(s) the run's gate let through")
    by_id = {r["id"]: r for r in records}
    groups = collections.defaultdict(list)
    for key in sorted(ok):
        groups[key[0]].append(ok[key])

    old_judge = manifest_in.get("config", {}).get("filter", {}).get("judge", {})
    print(f">>> run {run_dir}: {len(records)} prompts, {len(attempts)} candidates "
          f"({len(ok)} format-accepted, {len(attempts) - len(ok)} format-rejected)")
    print(f">>> old judge: {old_judge.get('model')} x{old_judge.get('runs')}  ->  "
          f"new judge: {judge['model']} x{runs}, threshold {flt['threshold']}, comparative")
    todo = [(by_id[rid], group, list(range(runs))) for rid in (r["id"] for r in records)
            if (group := groups.get(rid))]
    print(f">>> {len(todo) * runs} judge calls ({len(groups)} prompts x {runs} runs, "
          "a prompt's runs in sequence so run 1 can read run 0's cache)")
    if args.dry_run:
        return

    out = Path(args.out) if args.out else run_dir.parent / f"{run_dir.name}_rejudge_{timestamp()}"
    out.mkdir(parents=True, exist_ok=False)
    verdicts_path = out / "judgements.partial.jsonl"
    verdicts: list[dict] = []
    started = time.monotonic()
    client = OpenRouterClient()

    # The pipeline's batch runner is a closure over its run state; this is the same loop with
    # the same error record shape, minus the budget check (a rejudge is judge-only and cheap).
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = int(cfg["workers"])
    total = len(todo) * runs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for offset in range(0, len(todo), workers):
            batch = todo[offset:offset + workers]
            futures = [ex.submit(dp._judge_runs, client, r, g, ks, cfg, constitution) for r, g, ks in batch]
            for fut in as_completed(futures):
                for v in fut.result():  # _judge_runs never raises; errors are records
                    verdicts.append(v)
                    with verdicts_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(v, ensure_ascii=False) + "\n")
            done = sum("scores" in v for v in verdicts)
            errs = sum("scores" not in v for v in verdicts)
            print(f"    judged {done}/{total}" + (f" ({errs} errors)" if errs else ""), flush=True)

    chosen = dp._survivors(records, attempts, verdicts, flt)
    stage3 = dp._judge_stage(records, attempts, verdicts, flt, chosen)
    (out / "stage_3_judge.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in stage3), encoding="utf-8")
    rows = []
    for record in records:
        picked = chosen[record["id"]]
        if picked is None:
            continue
        candidate, score = picked
        provenance = {"source": manifest_in["source"], "model": cfg["model"], "provider": "Alibaba",
                      "constitution_sha256": manifest_in["constitution_sha256"],
                      "judge": {"model": judge["model"], "runs": runs, "threshold": flt["threshold"],
                                "score": score, "candidate": candidate,
                                "candidates_judged": len(groups.get(record["id"], []))},
                      "rejudged_from": str(run_dir)}
        rows.append(export_row(record, ok[(record["id"], candidate)]["assistant"], provenance))
    (out / "dataset.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    usage = dp._usage(verdicts, judge_price)
    # Cache accounting by run index: run 0 should hit the constitution prefix (shared across
    # prompts), run 1 the whole message (written by run 0 moments earlier).
    by_run = {}
    for k in range(runs):
        rs = [v["response"] for v in verdicts if v.get("run") == k and "response" in v]
        if rs:
            by_run[str(k)] = {"calls": len(rs), "retried": sum(1 for v in verdicts if v.get("run") == k and v.get("attempt")),
                              "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in rs) / len(rs)),
                              "cached_tokens_mean": round(sum(r.get("cached_tokens") or 0 for r in rs) / len(rs)),
                              "cached_share": round(sum(r.get("cached_tokens") or 0 for r in rs)
                                                    / max(1, sum(r["prompt_tokens"] for r in rs)), 3),
                              "cost_mean_usd": round(sum(r.get("cost") or 0 for r in rs) / len(rs), 4),
                              "truncated": sum(r.get("finish_reason") == "length" for r in rs)}
    scores = dp._scores(verdicts)
    mins = [m for k in ok if (m := dp.candidate_score(scores[k], runs)) is not None]
    hist = collections.Counter(mins)
    per_run_disagreement = sum(1 for k in ok if len(set(scores[k])) > 1 and len(scores[k]) == runs)
    summary = {
        "rejudged_from": str(run_dir), "config": args.config, "judge": judge,
        "threshold": flt["threshold"], "prompts": len(records), "candidates_judged": len(ok),
        "survivors": len(rows), "rejected_ids": [i for i, v in chosen.items() if v is None],
        "regated_by_current_gate": {f"{a['id']}/c{a['candidate']}": a["rejected_by_current_gate"]
                                    for a in attempts if "rejected_by_current_gate" in a},
        "min_score_histogram": {str(k): hist[k] for k in sorted(hist)},
        "candidates_with_run_disagreement": per_run_disagreement,
        "selected_score_mean": (sum(r["metadata"]["deliberative_alignment"]["judge"]["score"] for r in rows) / len(rows)
                                if rows else None),
        "judge_usage": usage, "judge_errors": sum("scores" not in v for v in verdicts),
        "by_run": by_run,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "command": shlex.join(sys.argv), "git_sha": git_sha(),
        "constitution": constitution_path, "constitution_sha256": sha, "judge_provider": judge_pin,
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("survivors", "prompts", "min_score_histogram",
                                              "candidates_with_run_disagreement", "selected_score_mean",
                                              "judge_errors", "by_run")}, indent=2))
    print(f">>> judge cost ${usage['total_usd']:.2f}; wrote {out}")


if __name__ == "__main__":
    main()
