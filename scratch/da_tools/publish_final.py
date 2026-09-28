# ABOUTME: Publish the audited, repaired da-tools corpus over the engine's first export in the same synth repo:
# ABOUTME: dataset.jsonl replaced, the audit trail under audit/, and a card section saying what changed and why.
# Run: uv run python scratch/da_tools/publish_final.py --repo <org/name> --dataset <merged.jsonl> --audit-dir <dir>
import argparse
import json
from pathlib import Path

from src.infra.huggingface import gate_push, hf_api, hf_download

SECTION = """
## Post-hoc audit and repair (this revision's `dataset.jsonl`)

The engine's export (`stages/stage_4_export_sft.jsonl`, 1,153 rows) was audited by an
independent judge, `openai/gpt-5.6-terra` -- a family that neither wrote the tools
(anthropic/claude-sonnet-5) nor gated them (google/gemini-3.6-flash) -- on the 606 tool-carrying
rows the da-15 mixture draws (`scratch/da_tools/audit_tools.py`, five questions per row, each in
its own call). First-pass flags: {first}. Every "contradicted" flag was a false positive (the reply
mentioning the user's own software), re-checked with the narrowed question.

Rows flagged useful / enables / misfit / unrealistic, the rows whose tools were exhausted by the
gate (mostly a lint false positive on `grant_*` names, since removed), and the 2 rows the first
run dropped at max_tokens were re-drawn with the same config and re-audited, three rounds:
{rounds}. For these re-drawn rows the auditor acted as a gate, so the first-pass rates above are
the honest measure of the in-pipeline gate; the rows stripped in the last round carry no tools
(`tools_status: audit_stripped`).

Final: {final}. Messages and source metadata are byte-identical to
`dougalldeepmind/2026-09-25-da-synth` @ 618060e1 in the same order
(`scratch/da_tools/verify_corpus.py`, report in `audit/verify_final.json`).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--audit-dir", required=True)
    a = ap.parse_args()
    repo = gate_push(a.repo, None, what="dataset push")
    audit = Path(a.audit_dir)
    first = json.loads((audit / "audit_20260928_163228" / "summary.json").read_text())
    rounds = []
    for k in (1, 2, 3):
        s = json.loads((audit / f"audit_repair{k}" / "summary.json").read_text())
        rounds.append(f"round {k}: {s['rows_audited']} re-drawn, {s['rows_with_any_flag']} still flagged")
    verify = json.loads((audit / "verify_final.json").read_text())
    section = SECTION.format(
        first=", ".join(f"{k} {v}" for k, v in first["flag_counts"].items()) + f" (any: {first['rows_with_any_flag']}/{first['rows_audited']})",
        rounds="; ".join(rounds),
        final=f"{verify['rows']} rows, tools_status {verify['tools_status']}, {verify['distinct_tool_names']} distinct tool names over {verify['tool_instances']} tools")
    card = Path(hf_download(repo, "README.md", repo_type="dataset")).read_text()
    marker = "\n## Post-hoc audit and repair"
    card = card.split(marker)[0].rstrip() + "\n" + section
    api = hf_api()
    uploads = [(a.dataset, "dataset.jsonl"), (str(audit / "verify_final.json"), "audit/verify_final.json")]
    for sub in ["audit_20260928_163228", "audit_repair1", "audit_repair2", "audit_repair3"]:
        for f in ("summary.json", "verdicts.jsonl"):
            uploads.append((str(audit / sub / f), f"audit/{sub}/{f}"))
    for k in (1, 2, 3):
        uploads.append((str(audit / f"repair{k}_ids.json"), f"audit/repair{k}_ids.json"))
    for local, remote in uploads:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="dataset",
                        commit_message=f"post-hoc audit: {remote}")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="card: post-hoc audit and repair")
    sha = api.list_repo_commits(repo, repo_type="dataset")[0].commit_id
    print(f"https://huggingface.co/datasets/{repo} @ {sha}")


if __name__ == "__main__":
    main()
