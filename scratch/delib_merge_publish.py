# ABOUTME: Assemble ONE deliberative-SFT corpus from a full run's checkpoints plus a re-run of its rejected
# ABOUTME: prompts, and publish it to the full run's repo as a new revision under the synth contract.

"""Merge a delib full run with the re-run of its rejects, and publish to the full run's repo.

Run: uv run python scratch/delib_merge_publish.py --full output/synth_delib/<run> \\
         --rejects output/synth_delib_rejects/<run> --repo 2026-09-10-delib-synth [--dry-run]

The full run stopped at its publish floor (658 of 708 survivors, floor 700) with every
candidate and judgement checkpointed but no rows exported. The rejected 50 were re-run under
the amended constitution (939a3ab) with 8 candidates each. This assembles the survivors of
both -- the pipeline's own `_survivors` / `export_row` on the full run's checkpoints, the
re-run's exported rows as they are -- into one dataset.jsonl, in source-row order, and pushes
it with stage snapshots, a manifest naming both generations, and the card, to the ORIGINAL
repo, so `mix` consumes it as `dataset: <org>/<repo>`.

Every row carries the constitution sha it was generated under (`metadata.deliberative_alignment
.constitution_sha256`), so the two generations are distinguishable per row; the manifest lists
both shas and which source rows came from which run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synth.deliberative_alignment import pipeline as dp  # noqa: E402
from src.data.synth.deliberative_alignment.data import export_row  # noqa: E402
from src.data.synth.ours.hf_cache import StageCache  # noqa: E402
from src.infra.huggingface import training_data_tags  # noqa: E402
from src.utils import git_sha, origin_url, read_jsonl, timestamp  # noqa: E402


def survivors_of(run_dir: Path) -> tuple[list[dict], dict, list[dict], list[dict], list[dict], dict]:
    """(rows, manifest, records, attempts, verdicts, chosen) for a run, from its checkpoints."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))["config"]
    flt = cfg["filter"]
    records = read_jsonl(run_dir / "stage_1_prompts.jsonl")
    attempts = [a for a in read_jsonl(run_dir / "generations.partial.jsonl") if "error" not in a]
    # last record per candidate wins (resume semantics)
    settled = {}
    for a in attempts:
        settled[(a["id"], a["candidate"])] = a
    attempts = list(settled.values())
    jpath = run_dir / "judgements.partial.jsonl"
    verdicts = [v for v in read_jsonl(jpath) if "scores" in v] if jpath.exists() else []
    chosen = dp._survivors(records, attempts, verdicts, flt)
    ok = {(a["id"], a["candidate"]): a for a in attempts if "assistant" in a}
    rows = []
    for record in records:
        picked = chosen[record["id"]]
        if picked is None:
            continue
        candidate, score = picked
        judged = sum(1 for key in ok if key[0] == record["id"])
        provenance = {"source": manifest["source"], "model": cfg["model"],
                      "provider": manifest["provider_pin"]["order"][0],
                      "constitution_sha256": manifest["constitution_sha256"],
                      "judge": {"model": flt["judge"]["model"], "runs": flt["judge"]["runs"],
                                "threshold": flt["threshold"], "score": score, "candidate": candidate,
                                "candidates_judged": judged},
                      "run_id": manifest["run_id"]}
        rows.append(export_row(record, ok[(record["id"], candidate)]["assistant"], provenance))
    return rows, manifest, records, attempts, verdicts, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--rejects", required=True)
    ap.add_argument("--repo", required=True, help="the FULL run's repo name (after the org)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    full, rej = Path(args.full), Path(args.rejects)

    f_rows, f_man, f_records, f_attempts, f_verdicts, f_chosen = survivors_of(full)
    r_rows, r_man, r_records, r_attempts, r_verdicts, r_chosen = survivors_of(rej)
    f_cfg = json.loads((full / "run_meta.json").read_text())["config"]
    r_cfg = json.loads((rej / "run_meta.json").read_text())["config"]

    # The re-run must be exactly the full run's rejects, on the same source at the same revision.
    assert f_man["source"]["repo"] == r_man["source"]["repo"] and f_man["source"]["revision"] == r_man["source"]["revision"], "different sources"
    f_rejected = {int(i) for i in f_man["filter"]["rejected_ids"]}
    r_source_rows = {r["source_row"] for r in r_records}
    assert r_source_rows == f_rejected, f"re-run rows {sorted(r_source_rows)[:5]}... != full run's rejects"
    assert not ({r["metadata"]["deliberative_alignment"]["source_row"] for r in f_rows}
                & {r["metadata"]["deliberative_alignment"]["source_row"] for r in r_rows}), "a row in both"

    rows = sorted(f_rows + r_rows, key=lambda r: r["metadata"]["deliberative_alignment"]["source_row"])
    still_rejected = sorted(f_rejected - {r["metadata"]["deliberative_alignment"]["source_row"] for r in r_rows})
    print(f"full run: {len(f_rows)} survivors of {len(f_records)}; re-run: {len(r_rows)} of {len(r_records)} recovered; "
          f"merged {len(rows)} rows; still rejected {len(still_rejected)}: {still_rejected}")
    print(f"constitution shas: full {f_man['constitution_sha256'][:12]}, re-run {r_man['constitution_sha256'][:12]}")
    if args.dry_run:
        return

    ts = timestamp()
    out = Path(args.out) if args.out else Path("output/synth_delib") / f"merged_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    by_id = {r["id"]: r for r in f_records}
    # stage snapshots: the full run's prompts; both runs' responses and judge rows, ordered
    responses = sorted(f_attempts + [{**a, "id": a["id"], "run_id": r_man["run_id"]} for a in r_attempts],
                       key=lambda a: (by_id.get(a["id"], {"source_row": int(a["id"])})["source_row"], a.get("run_id", ""), a["candidate"]))
    judge_rows = (dp._judge_stage(f_records, f_attempts, f_verdicts, f_cfg["filter"], f_chosen)
                  + [{**r, "run_id": r_man["run_id"]} for r in dp._judge_stage(r_records, r_attempts, r_verdicts, r_cfg["filter"], r_chosen)])
    usage = {"full_run": f_man["usage"], "rejects_run": r_man["usage"],
             "total_usd": f_man["usage"]["total_usd"] + r_man["usage"]["total_usd"]}
    manifest = {
        "run_id": ts, "method": "deliberative_alignment", "pipeline": f_cfg["pipeline"], "merged": True,
        "merged_from": [{"run_id": f_man["run_id"], "run_dir": str(full), "constitution_sha256": f_man["constitution_sha256"],
                         "rows": sorted(r["metadata"]["deliberative_alignment"]["source_row"] for r in f_rows),
                         "candidates_per_prompt": f_cfg["filter"]["candidates"]},
                        {"run_id": r_man["run_id"], "run_dir": str(rej), "constitution_sha256": r_man["constitution_sha256"],
                         "rows": sorted(r["metadata"]["deliberative_alignment"]["source_row"] for r in r_rows),
                         "candidates_per_prompt": r_cfg["filter"]["candidates"],
                         "note": "the full run's rejected prompts, re-run under the amended constitution (939a3ab)"}],
        "source": f_man["source"], "source_rows": len(f_records), "prompts_sha256": f_man["prompts_sha256"],
        "constitution_sha256": {"full_run": f_man["constitution_sha256"], "rejects_run": r_man["constitution_sha256"]},
        "provider_pin": f_man["provider_pin"], "judge_provider_pin": f_man["judge_provider_pin"],
        "filter": {**f_cfg["filter"], "survivors": len(rows), "rejected_ids": [str(i) for i in still_rejected],
                   "note": "rejects_run used candidates: %d" % r_cfg["filter"]["candidates"]},
        "usage": usage, "completed_rows": len(rows), "status": "complete", "dataset": "dataset.jsonl",
        "hf_repo": args.repo, "git_sha": git_sha(), "commands": [{"command": shlex.join(sys.argv), "git_sha": git_sha()}],
        "run_dir": str(out),
    }
    card = {"experiment": (f"Deliberative SFT: {f_cfg['pipeline']}; native Qwen reasoning, best-of-{f_cfg['filter']['candidates']} "
                           f"filtered by a constitution-aware judge ({f_cfg['filter']['judge']['model']}, min of "
                           f"{f_cfg['filter']['judge']['runs']} runs >= {f_cfg['filter']['threshold']}). {len(f_rows)} rows from the "
                           f"full run; the {len(f_records) - len(f_rows)} prompts it rejected were re-run with {r_cfg['filter']['candidates']} "
                           f"candidates under the amended constitution, recovering {len(r_rows)}; {len(still_rejected)} remain rejected"),
            "date_generated": f_man["run_id"], "constitution": f_cfg["constitution"] + " (two revisions; per-row sha in metadata, both in manifest)",
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": (f"{f_cfg['model']} through {f_man['provider_pin']['order'][0]}/OpenRouter (API revision not exposed); "
                       f"judge {f_cfg['filter']['judge']['model']} through OpenRouter"),
            "generation_config": "manifest.json: both runs' resolved configs, pins, pricing, usage and filter stats (merged_from)",
            "schema": "dataset.jsonl: messages + metadata + optional tools; stages/ snapshots",
            "provenance": f"{shlex.join(sys.argv)}; prompts from {f_man['source']['repo']} @ {f_man['source']['revision']} / dataset.jsonl"}
    cache = StageCache(out, args.repo, private=bool(f_cfg.get("hf_private", False)), card_fields=card,
                       tags=training_data_tags("synth", f_cfg["pipeline"], f_cfg["constitution"], smoke=False))
    cache.save(1, "prompts", f_records)
    (out / "generation_prompt.txt").write_text((full / "generation_prompt.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "generation_prompt.rejects.txt").write_text((rej / "generation_prompt.txt").read_text(encoding="utf-8"), encoding="utf-8")
    cache.save(2, "responses", responses)
    cache.save(3, "judge", judge_rows)
    cache.save(4, "export_sft", rows)
    cache.publish_final(rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps({"git_sha": git_sha(), "config": f_cfg, "rejects_config": r_cfg, "timestamp": ts,
                                                   "merged_from": manifest["merged_from"], "commands": manifest["commands"]},
                                                  indent=2, ensure_ascii=False), encoding="utf-8")
    cache.checkpoint([out / "manifest.json", out / "run_meta.json", out / "generation_prompt.txt", out / "generation_prompt.rejects.txt"],
                     f"merged corpus: {len(rows)} rows ({len(f_rows)} full run + {len(r_rows)} recovered)")
    print(f">>> published {len(rows)} rows to {cache.repo_id}; local {out}")


if __name__ == "__main__":
    main()
