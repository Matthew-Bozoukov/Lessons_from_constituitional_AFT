# ABOUTME: Publish a finished daa2 run as a synth corpus on HF: dataset.jsonl as the default config, the
# ABOUTME: manifest and the row records beside it, under the card and tags every training corpus carries.
"""Publish a daa2 run dir.

    uv run python scratch/daa2/publish.py output/synth/daa2/<run_dir>

The run dir's manifest records the source repo the CLI defaulted to; the rows themselves carry the
source they were actually loaded from (source.jsonl), so the card and the manifest are written from
the rows. Name: `<today>-daa-synth` (src/naming.py); the card's date is today's, as gate_push requires.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
from src.naming import synth_name
from src.utils import git_sha, origin_url, timestamp

CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"   # the 2026-09-14 da corpus's


def main() -> None:
    run_dir = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(run_dir / "dataset.jsonl", encoding="utf-8")]
    src = [json.loads(l) for l in open(run_dir / "source.jsonl", encoding="utf-8")]
    (source_repo, source_rev), = collections.Counter((r["source_repo"], r["source_revision"]) for r in src).keys()
    man = json.loads((run_dir / "manifest.json").read_text())
    man["source"] = {"repo": source_repo, "revision": source_rev}      # the CLI default in the manifest was stale
    man["traits"] = dict(sorted(collections.Counter(r["metadata"]["trait_id"] for r in rows).items()))
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    shapes = collections.Counter(("write" if r["metadata"]["n_write"] else "") + ("run" if r["metadata"]["n_run"] else "") or "looks-only" for r in rows)
    ts = timestamp()
    repo = hf_repo_id(synth_name("daa"))            # gate_push wants the qualified id; the org is .env's
    fields = {
        "experiment": "daa2: the difficult-advice corpus agentified by EDITING each row, not regenerating it -- the "
                      "DA row's system prompt, user message, private reasoning and reply are numbered sentences; a "
                      "map call builds a small file environment and inserts real bash calls (looks, script runs, "
                      "file writes) only where the reply itself does or offers that thing; the commands run in a "
                      "docker sandbox under the scenario's clock; a think call edits the reasoning before any "
                      "action (it sees the looks' output only, so it cannot quote a result before the command ran; "
                      "its plan is exactly the listed actions); a finish call writes the files and edits the reply "
                      "so it carries the deliberation out. Untouched sentences survive verbatim.",
        "date_generated": ts,
        "constitution": CONSTITUTION,
        "source_repo": f"{origin_url()} @ {git_sha()} (scratch/daa2/agentify2.py with the 2026-09-16 prompt revision, "
                       f"uncommitted at generation time)",
        "models": f"generators through Claude Code's print mode (`claude -p --bare`, structured output, default "
                  f"effort, hidden reasoning on) for the map, think and finish calls, per row: "
                  f"{dict(collections.Counter(r['metadata']['generator'] for r in rows))} (the opus rows are the "
                  f"sonnet content-classifier refusals retried with opus); source rows from {source_repo}@{source_rev[:8]}",
        "generation_config": f"three structured calls per row (map, think, finish), --max-turns 3, one map repair on a "
                             f"failed command; docker sandbox python:3.12-slim + libfaketime frozen at the scenario "
                             f"date; 8 workers; supervise=all; kept {man['counts']['kept']} of {man['counts']['rows']} "
                             f"(all {man['counts']['dropped']} drops are Anthropic content-classifier refusals); "
                             f"medians {man['medians']}; shapes {dict(shapes)}",
        "schema": "dataset.jsonl rows: scenario_id; messages (system, user, [assistant(reasoning_content)+tool]*, "
                  "assistant(reasoning_content, content, tool_calls=[task_complete])) with OpenAI-style tool_calls; "
                  "tools (bash, task_complete -- the ODCV schemas); metadata (the DA row's trait fields and ids, "
                  "supervise, reuse_reasoning/reuse_reply/reuse_user, n_look/n_run/n_write, n_files, chars_files, "
                  "generator); environment (the files). rows.jsonl: the per-row records (map output, insertions, "
                  "executed steps, fill output) the dataset was assembled from. manifest.json: counts, usage, medians.",
        "provenance": "uv run python scratch/daa2/agentify2.py --source dougalldeepmind/2026-09-14-da-synth "
                      "--revision 013886238fca238c4d54ace96530f444bb2b2f02 --model sonnet --workers 8 "
                      "(resumed after the prompt revision; the rows made before it were then redone with "
                      "`--refill <run_dir> --model sonnet --workers 8`, which reuses each row's environment and "
                      f"command outputs and regenerates the think/finish calls); then `uv run python "
                      f"scratch/daa2/publish.py {run_dir}`",
    }
    front_matter = {"configs": [{"config_name": "dataset", "data_files": "dataset.jsonl", "default": True}],
                    "tags": training_data_tags("synth", "daa", CONSTITUTION)}
    url = push_files([run_dir / "dataset.jsonl", run_dir / "manifest.json", run_dir / "rows.jsonl"], repo, fields,
                     private=False, front_matter=front_matter)
    print(url, "| rows", len(rows), "| traits", man["traits"], "| shapes", dict(shapes))


if __name__ == "__main__":
    main()
