# ABOUTME: Builds the human-parties MDMA corpus: the all-human MDMA rows plus a da-multiparty-human run's rows (checked
# ABOUTME: agent-free, advisor tools stripped, cross-run near-duplicates dropped), 84 per principle; can publish it.
"""Merge the reused all-human MDMA rows with a finished da-multiparty-human run.

Run: uv run python scratch/da_multiparty/merge_human_corpus.py <run dir> --config configs/data/synth/da-multiparty-human.yaml [--push <org/repo>]

1. Reused: the MDMA corpus rows (dougalldeepmind/2026-09-23-da-multiparty-synth @ 8a53f84f) that
   human_parties.agent_free accepts: a person asks, no party is or mentions an AI agent.
2. New: <run dir>/dataset.jsonl. Every row must pass the same check; one that does not is dropped
   and listed (the config forbids agents; this verifies it held).
3. Tools: an `advisor` row's tools are emptied -- revise_prompts writes them anyway (the
   2026-09-23 defect; strip_advisor_tools.py's rule), and the MDMA corpus was stripped the same way.
4. Near-duplicates: a new row whose situation is within the config's reject_cosine of a reused
   row's is dropped (splice_topup.py's cross-run rule).
5. Cap: per principle, reused rows first, then new rows in scenario_id order, up to 84.
Writes <run dir>/dataset.jsonl (the run's own export kept as dataset.pre_merge.jsonl) and
<run dir>/human_merge.json; --push uploads both to the run's Hub repo, whose README already
declares dataset.jsonl as its default config.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_parties import CORPUS, REVISION, agent_free, load  # noqa: E402

from src.data.synth.ours.embeddings import DEFAULT_MODEL, embed  # noqa: E402

PER_PRINCIPLE = 84


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True)
    ap.add_argument("--push", help="org/repo of the run's Hub dataset")
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = yaml.safe_load(open(args.config))
    div = (
        next(s for s in cfg["stages"] if s["name"] == "write_scenarios").get(
            "diversity"
        )
        or {}
    )
    reject = float(div["reject_cosine"])
    model = str(div.get("embed_model") or DEFAULT_MODEL)

    src = run / "dataset.jsonl"
    pre = run / "dataset.pre_merge.jsonl"
    if not pre.exists():
        shutil.copy(src, pre)
    new = [json.loads(line) for line in pre.read_text().splitlines() if line.strip()]
    reused = [r for r in load() if agent_free(r)]

    not_free = [r["metadata"]["scenario_id"] for r in new if not agent_free(r)]
    new = [r for r in new if agent_free(r)]

    stripped = []
    for r in reused + new:
        if r["metadata"].get("agenticness") == "advisor" and r.get("tools"):
            stripped.append(r["metadata"]["scenario_id"])
            r["tools"] = []

    X_old = embed([r["metadata"]["situation"] for r in reused], model=model)
    X_new = embed([r["metadata"]["situation"] for r in new], model=model)
    nearest = (X_new @ X_old.T).max(axis=1)
    dup = {
        r["metadata"]["scenario_id"]: round(float(c), 4)
        for r, c in zip(new, nearest)
        if float(c) >= reject
    }
    new = [r for r in new if r["metadata"]["scenario_id"] not in dup]

    by_t = collections.defaultdict(list)
    for r in reused:
        by_t[r["metadata"]["trait_id"]].append(("reused", r))
    for r in sorted(new, key=lambda r: r["metadata"]["scenario_id"]):
        by_t[r["metadata"]["trait_id"]].append(("new", r))
    out, counts, surplus = [], {}, {}
    for t in sorted(by_t):
        kept = by_t[t][:PER_PRINCIPLE]
        surplus[t] = [r["metadata"]["scenario_id"] for _, r in by_t[t][PER_PRINCIPLE:]]
        counts[t] = {
            "reused": sum(1 for k, _ in kept if k == "reused"),
            "new": sum(1 for k, _ in kept if k == "new"),
            "total": len(kept),
        }
        out += [r for _, r in kept]
    short = {t: c["total"] for t, c in counts.items() if c["total"] < PER_PRINCIPLE}

    record = {
        "rule": "every party human (asker a person; no party is or mentions an AI agent); reused all-human "
        f"MDMA rows first, then this run's rows, {PER_PRINCIPLE} per principle",
        "reused_from": {"repo": CORPUS, "revision": REVISION, "rows": len(reused)},
        "run_rows": len(new) + len(dup) + len(not_free),
        "dropped_not_agent_free": not_free,
        "dropped_near_duplicates_of_reused": dup,
        "reject_cosine": reject,
        "advisor_tools_stripped": stripped,
        "per_principle": counts,
        "surplus_not_used": surplus,
        "short_principles": short,
        "rows": len(out),
    }
    src.write_text("".join(json.dumps(r) + "\n" for r in out))
    (run / "human_merge.json").write_text(json.dumps(record, indent=1))
    print(
        json.dumps(
            {k: record[k] for k in ("rows", "per_principle", "short_principles")},
            indent=1,
        )
    )
    print(
        f"dropped: {len(not_free)} not agent-free, {len(dup)} near-duplicates; tools stripped on {len(stripped)}"
    )
    if short:
        print("SHORT: top up these principles before mixing")
    if args.push and not short:
        from huggingface_hub import HfApi

        api = HfApi()
        for f in ("dataset.jsonl", "human_merge.json"):
            api.upload_file(
                path_or_fileobj=str(run / f),
                path_in_repo=f,
                repo_id=args.push,
                repo_type="dataset",
                commit_message=f"human-parties merge: {len(out)} rows, {PER_PRINCIPLE} per principle "
                f"({len(reused)} reused all-human MDMA rows @ {REVISION[:8]} + this run's rows)",
            )
        print("pushed", args.push, api.dataset_info(args.push).sha)


if __name__ == "__main__":
    main()
