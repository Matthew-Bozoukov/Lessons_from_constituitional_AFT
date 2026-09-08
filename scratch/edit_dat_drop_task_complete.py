# ABOUTME: Drop the `task_complete` tool call from the supervised turn of every dat corpus row
# ABOUTME: and republish the corpus, so the arm never learns to declare a task done unobserved.
# Run: uv run python scratch/edit_dat_drop_task_complete.py [--source <hf id>] [--no-push]
#
# WHY (docs/LOG.md 2026-09-07, ODCV of `2026-09-06-qwen36-0-dat-7`): the arm's task-progress
# mean is 4.42 against 4.9-5.0 for every other arm, and 31 of its 43 low-scoring rollouts are
# the same failure — one chained command, an error nobody reads, then `task_complete` with a
# summary asserting the work was done. That is the literal shape of the training row. Every
# corpus row carries `supervise: final`, so exactly one turn is a loss target, and that turn
# emits `bash` and `task_complete` TOGETHER: the summary is written before the command runs,
# and the response prompt says so outright ("it cannot quote the command's output, because you
# have not run it yet"). The model is doing precisely what it was shown.
#
# THE EDIT: remove the second tool call. The reasoning, the status report, the bash call and
# the turn terminator all stay, so the rendered shape and the generation-boundary mask are
# unchanged; the row now ends on an unresolved action, which is what a real agent turn does.
# The summary text lived only inside that call's arguments and leaves with it — verified by
# scan: across all 733 rows neither the reasoning nor the response mentions the completion tool
# in any form (the 75 hits for "submit"/"signoff"/"close out" are the scenarios' own subject
# matter: engineer signoffs, NSF closeout reports, filing claims).
#
# NOT FIXED HERE, and not fixable by editing: no row contains a failing tool result, so nothing
# teaches recovery. That needs a generation stage and a wider `supervise` mode.
#
# The `tools` schema still declares `task_complete` — the eval harness offers it, and the row
# must keep the schema the model will see. Nothing now demonstrates calling it, so SUBMISSION
# RATE is the number to read first on the next ODCV run: task progress is capped below full
# marks without a submit call, so a fall there would be this edit's cost, not its effect.

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.infra.huggingface import hf_repo_id, push_run_dir, training_data_tags
from src.naming import synth_name
from src.utils import git_sha

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

SOURCE = "LASR-Callum/2026-09-06-dat-synth"
# The two files that carry chat rows. Every other stage snapshot is the generation record and
# is republished byte-for-byte: the corpus was not regenerated, only its export was edited.
EDITED = ("dataset.jsonl", "stages/stage_10_export_sft.jsonl")
DROP = "task_complete"
WORK = Path("output/dat_drop_task_complete")


def strip_call(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Remove the DROP tool call from each row's LAST assistant message.

    Returns (rows, calls_removed, rows_touched). Asserts the call appears nowhere else: an
    earlier turn calling it would mean the corpus has a shape this edit does not understand.
    """
    removed = touched = 0
    for row in rows:
        msgs = row["messages"]
        last = next(i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "assistant")
        for i, m in enumerate(msgs):
            names = [c["function"]["name"] for c in (m.get("tool_calls") or [])]
            assert i == last or DROP not in names, (
                f"{DROP} called in a non-final turn ({i} of {len(msgs)}); this edit assumes "
                "the completion call only ever closes a row")
        calls = msgs[last].get("tool_calls") or []
        kept = [c for c in calls if c["function"]["name"] != DROP]
        if len(kept) != len(calls):
            removed += len(calls) - len(kept)
            touched += 1
            msgs[last]["tool_calls"] = kept
        assert msgs[last].get("tool_calls"), (
            "the supervised turn must still carry its bash call; dropping the completion "
            "call must never leave a turn with no action at all")
    return rows, removed, touched


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def unchanged_except_the_call(before: list[dict], after: list[dict]) -> None:
    """Everything but the dropped call must be identical — the edit is one deletion, not a rewrite."""
    assert len(before) == len(after), f"row count changed: {len(before)} -> {len(after)}"
    for b, a in zip(before, after):
        bb, aa = json.loads(json.dumps(b)), json.loads(json.dumps(a))
        for row in (bb, aa):
            last = next(i for i in range(len(row["messages"]) - 1, -1, -1)
                        if row["messages"][i]["role"] == "assistant")
            row["messages"][last]["tool_calls"] = [
                c for c in (row["messages"][last].get("tool_calls") or [])
                if c["function"]["name"] != DROP]
        assert bb == aa, "the edit changed something other than the completion call"


def front_matter(root: Path, tags: list[str]) -> dict:
    """Rebuild the `configs:` block from the files actually present (src/data/synth/constitutional_sft/hf_cache.py)."""
    stages = sorted((p.name for p in (root / "stages").glob("*.jsonl")),
                    key=lambda f: int(m.group(1)) if (m := re.match(r"stage_(\d+)_", f)) else 0)
    configs = [{"config_name": "dataset", "data_files": "dataset.jsonl", "default": True}]
    configs += [{"config_name": f.removesuffix(".jsonl"), "data_files": f"stages/{f}"} for f in stages]
    return {"configs": configs, "tags": tags}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--no-push", dest="push", action="store_false")
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    root = Path(snapshot_download(args.source, repo_type="dataset", local_dir=str(WORK / "corpus")))
    print(f">>> cloned {args.source} -> {root}")

    total_removed = 0
    for rel in EDITED:
        path = root / rel
        before = read(path)
        after, removed, touched = strip_call(read(path))
        unchanged_except_the_call(before, after)
        write(path, after)
        total_removed += removed
        names = {tuple(c["function"]["name"] for c in r["messages"][-1].get("tool_calls", []))
                 for r in read(path)}
        tools = {tuple(t["function"]["name"] for t in r.get("tools", [])) for r in read(path)}
        print(f">>> {rel}: {len(after)} rows, {removed} calls removed from {touched} rows; "
              f"final-turn calls now {names}; tools schema still {tools}")
        assert not any(DROP in n for n in names), f"{DROP} survived in {rel}"
        assert all(DROP in t for t in tools), "the tools schema must still declare the tool"

    subject = args.source.split("/")[-1].removeprefix("2026-09-06-").removesuffix("-synth")
    repo = synth_name(subject)
    print(f">>> new corpus name: {repo}  (style unchanged; only the date moves)")
    if not args.push:
        print(">>> --no-push: stopping before the upload")
        return

    fields = {
        "experiment": f"synth `{subject}` corpus, edited: the `{DROP}` call dropped from every "
                      "supervised turn so the arm never declares a task done before its "
                      "command runs (docs/LOG.md 2026-09-07)",
        "date_generated": date.today().isoformat(),
        "constitution": "constitutions/no_claude_mentioned/constitution.md",
        "source_repo": f"lasr @ {git_sha()}",
        "models": f"none — no generation ran; rows are {args.source} verbatim minus one tool call",
        "generation_config": f"see manifest.json (inherited from {args.source}); "
                             "this repo adds no sampling of its own",
        "schema": "dataset.jsonl (default config) + stages/ snapshots + manifest.json, as the "
                  f"source repo; the last assistant turn of every chat row now calls `bash` "
                  f"only, and `{DROP}` is declared in `tools` but never demonstrated",
        "provenance": "uv run python scratch/edit_dat_drop_task_complete.py "
                      f"--source {args.source}",
        "edited_from": f"{args.source} ({total_removed} tool calls removed, nothing else changed)",
    }
    tags = list(training_data_tags("synth", subject, "constitutions/no_claude_mentioned/constitution.md"))
    url = push_run_dir(root, hf_repo_id(repo), fields, private=False,
                       front_matter=front_matter(root, tags))
    print(f">>> pushed {url}")


if __name__ == "__main__":
    main()
