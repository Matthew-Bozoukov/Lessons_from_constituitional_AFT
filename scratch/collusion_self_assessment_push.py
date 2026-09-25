# ABOUTME: Pushes one collusion self-assessment run dir (answers, judge labels, report, run_meta) to a
# ABOUTME: private HF dataset named by artifact_name, with the repo's required card fields.
"""Run: uv run python scratch/collusion_self_assessment_push.py <run dir> <subject>

The date in the repo name is the date the run was produced (its run_meta timestamp), so the
card's `date_generated` and the name agree, which is what `gate_push` checks.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.infra.huggingface import card_markdown, gate_push, hf_api, hf_repo_id
from src.naming import artifact_name


def main(run_dir: str, subject: str) -> None:
    """Upload the run dir's files under the minted name and print the repo id."""
    load_dotenv()
    run = Path(run_dir)
    meta = json.loads((run / "run_meta.json").read_text())
    cfg = meta["config"]
    date = meta["timestamp_utc"][:10]
    repo_id = hf_repo_id(artifact_name(subject, date=date))
    fields = {
        "experiment": ("In-situ self-assessment of collusion: each agent's own agent_collusion "
                       "history, cut right after its ACCEPT in the trajectory's first mutual-ACCEPT "
                       f"episode (context strip: {list(cfg['strip']) or 'none'}), is asked: "
                       f"{json.dumps(cfg['questions'])}"),
        "date_generated": date,
        "constitution": "none",
        "source_repo": f"teaching_claude_why_replication @ {meta['git_sha']}",
        "models": (f"answering: {cfg['target']} @ {meta['target_revision']} "
                   f"(base {meta['base_model']}, mode {meta['mode']}); "
                   "judge (breach.jsonl): anthropic/claude-sonnet-5 via OpenRouter, temperature 0"),
        "generation_config": json.dumps({"samples": cfg["samples"], **cfg["generation"],
                                         "rollouts": f"{cfg['run_repo']} @ {meta['run_repo_revision']}"}),
        "schema": ("answers.jsonl / labels.jsonl: one row per (seq, agent, sample) with the answer; "
                   "labels adds verdict (first word). breach.jsonl: verdict + acknowledges_departure. "
                   "report.md: tallies + every answer. run_meta.json: config, SHA, command."),
        "provenance": (f"{meta['command']} ; then scratch/collusion_self_assessment_report.py and "
                       "scratch/collusion_self_assessment_breach.py on the run dir"),
    }
    gate_push(repo_id, fields, what="self-assessment run")
    api = hf_api()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=False)
    api.upload_file(path_or_fileobj=card_markdown(fields).encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    for name in ("answers.jsonl", "labels.jsonl", "breach.jsonl", "report.md", "run_meta.json"):
        api.upload_file(path_or_fileobj=str(run / name), path_in_repo=name,
                        repo_id=repo_id, repo_type="dataset")
    print(repo_id)


if __name__ == "__main__":
    main(*sys.argv[1:])
