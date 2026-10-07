# ABOUTME: Derive a daa corpus cut at each row's deliberation turn (the one long trace) with the tool
# ABOUTME: schemas removed, and publish it as `<date>-daa-synth` for a `supervise: final` arm.
"""Trim an agentified difficult-advice corpus to its deliberation and publish the result.

A daa row (dougalldeepmind/2026-09-17-daa-synth) is 3-4 read-only looks, one turn carrying the
da deliberation (a 2.9k-6.6k char trace; every other turn's trace is under 510 chars), then in
465 of 749 rows an action (a memo written or a script run) plus 1-8 confirming turns, and
finally the reply with a `task_complete` call. This script keeps every message up to and
including the deliberation turn and drops the rest, so `supervise: final` trains exactly that
turn: the deliberation plus whatever the turn holds (the action call in 465 rows; the reply in
the 284 where the deliberation is the final turn, whose `task_complete` call is removed). The
row's `tools` field is dropped, so the template renders no tool definitions. Rows are
otherwise byte-identical to their source (Jamie, 2026-10-07).

    uv run python scratch/daa_trim.py --repo dougalldeepmind/2026-09-17-daa-synth \
        --revision 9d0f1af04c001332d2c7f0941519d48c7b6986fa
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd

from src.data.synth.ours.hf_cache import dataset_card, write_jsonl
from src.infra.huggingface import gate_push, hf_api, hf_download, hf_repo_id, training_data_tags
from src.naming import synth_name, today
from src.utils import git_sha, origin_url

STYLE = "daa"
MIN_RATIO = 5.0  # the deliberation trace must be at least this many times the next-longest trace


def trim(row: dict) -> tuple[dict, dict]:
    """Cut `row` after its deliberation turn; return the new row and a note of what went."""
    msgs = row["messages"]
    idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    lengths = [len(msgs[i].get("reasoning_content") or "") for i in idx]
    k = max(range(len(idx)), key=lambda j: lengths[j])
    second = max((n for j, n in enumerate(lengths) if j != k), default=0)
    assert lengths[k] >= MIN_RATIO * max(second, 1), (
        f"{row['scenario_id']}: deliberation trace {lengths[k]} vs next {second}: not one clear turn")
    cut = idx[k]
    turn = dict(msgs[cut])
    was_final = k == len(idx) - 1
    if was_final:
        names = [c["function"]["name"] for c in turn.get("tool_calls") or []]
        assert names == ["task_complete"], f"{row['scenario_id']}: final turn calls {names}"
        turn.pop("tool_calls")
        assert (turn.get("content") or "").strip(), f"{row['scenario_id']}: final turn has no reply"
    else:
        assert turn.get("tool_calls"), f"{row['scenario_id']}: deliberation turn neither final nor acting"
    new = {k_: v for k_, v in row.items() if k_ not in ("tools", "messages")}
    new["messages"] = msgs[:cut] + [turn]
    note = {"deliberation_turn": k, "assistant_turns_in": len(idx), "turns_dropped": len(idx) - 1 - k,
            "deliberation_was_final": was_final,
            "kept_call": None if was_final else [c["function"]["name"] for c in turn["tool_calls"]],
            "trace_chars": lengths[k], "next_trace_chars": second}
    new["metadata"] = {**(row.get("metadata") or {}), "trim": note}
    return new, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    src = [json.loads(l) for l in open(hf_download(args.repo, "dataset.jsonl", repo_type="dataset",
                                                    revision=args.revision), encoding="utf-8")]
    readme = Path(hf_download(args.repo, "README.md", repo_type="dataset", revision=args.revision)).read_text()
    # the card writes the field as a table row (| `constitution` | path |); the tag line only carries the folder name
    m = re.search(r"^\|\s*`?constitution`?\s*\|\s*(.+?)\s*\|", readme, re.M) or re.search(r"^constitution:\s*(.+)$", readme, re.M)
    constitution = m.group(1).strip().strip("'\"") if m else "none"
    assert constitution != "none", f"{args.repo}: could not read the constitution from the card; refusing to publish a card that says none"

    out_rows, notes = [], []
    for r in src:
        new, note = trim(r)
        out_rows.append(new)
        notes.append(note)
    n_final = sum(n["deliberation_was_final"] for n in notes)
    kept_calls = collections.Counter(c for n in notes for c in (n["kept_call"] or []))
    dropped = collections.Counter(n["turns_dropped"] for n in notes)
    by_trait = collections.Counter(str((r.get("metadata") or {}).get("trait_id")) for r in out_rows)
    assert not any("tools" in r for r in out_rows)

    name = synth_name(STYLE)
    repo_id = hf_repo_id(name)
    out = Path("output/derived") / name
    command = "uv run python " + " ".join(sys.argv)
    card = {
        "experiment": (f"{STYLE}: `{args.repo}` cut after each row's deliberation turn (the one long trace) with "
                       f"the tool schemas removed, for a `supervise: final` arm that trains the deliberation and "
                       f"what its turn holds: the reply in {n_final} rows (their task_complete call removed), the "
                       f"action call in {len(src) - n_final} ({dict(kept_calls)}). Rows otherwise byte-identical."),
        "date_generated": today(),
        "constitution": constitution,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"inherited from {args.repo} @ {args.revision} (nothing generated here)",
        "generation_config": f"derived: trim to the longest-trace assistant turn (ratio to next >= {MIN_RATIO}); "
                             f"drop `tools`; drop task_complete where the deliberation was already the final turn",
        "schema": "dataset.jsonl (chat rows, synth contract, NO tools field, metadata.trim) + provenance.json",
        "provenance": command,
        "derived_from": f"{args.repo} @ {args.revision}",
        "rows": f"{len(out_rows)}; turns dropped per row {dict(sorted(dropped.items()))}; per trait "
                + ", ".join(f"{t} {n}" for t, n in sorted(by_trait.items())),
    }
    gate_push(repo_id, card, what="derived corpus")
    write_jsonl(out / "dataset.jsonl", out_rows)
    (out / "provenance.json").write_text(json.dumps({
        "derived_from": {"repo": args.repo, "revision": args.revision},
        "rows": len(out_rows), "deliberation_was_final": n_final, "kept_calls": dict(kept_calls),
        "turns_dropped": {str(k): v for k, v in sorted(dropped.items())},
        "per_trait": dict(sorted(by_trait.items())), "command": command, "git_sha": git_sha()}, indent=2))
    readme_out = dataset_card(card, [], True, training_data_tags("synth", STYLE, constitution))
    api = hf_api()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.create_commit(
        repo_id=repo_id, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_out.encode()),
                    CommitOperationAdd(path_in_repo="dataset.jsonl", path_or_fileobj=str(out / "dataset.jsonl")),
                    CommitOperationAdd(path_in_repo="provenance.json", path_or_fileobj=str(out / "provenance.json"))],
        commit_message=f"derived: {len(out_rows)} daa rows trimmed to the deliberation turn, tools removed")
    sha = api.dataset_info(repo_id).sha
    print(f">>> {len(out_rows)} rows; deliberation final in {n_final}; kept action calls {dict(kept_calls)}; "
          f"turns dropped {dict(sorted(dropped.items()))}; per trait {dict(sorted(by_trait.items()))}")
    print(f">>> https://huggingface.co/datasets/{repo_id} @ {sha}")


if __name__ == "__main__":
    main()
