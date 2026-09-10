# ABOUTME: Publishes the stopped CRLF-contaminated ODCV attempt as INVALID and UNJUDGED.
# ABOUTME: Preserves source hashes, excludes runtime credentials, and scans staged bytes before upload.
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.layout import assert_layout, publish_layout
from src.infra.huggingface import card_markdown, hf_api, hf_org, hf_token, push_run_dir
from src.naming import artifact_name

BASE = ROOT / "output/nonmoral_overnight/20260909"
SOURCE = Path("C:/nm-eval/2026-09-09_odcv_nonmoral_common_3x_021823")
DEST = BASE / "publication/invalid_baseline"
NAME = artifact_name("odcv-nonmoral-invalid-crlf-unjudged-attempt", date="2026-09-09")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def secret_values() -> list[bytes]:
    values = dict(dotenv_values(ROOT / ".env"))
    values.update(os.environ)
    found = [v.encode() for k, v in values.items() if v and len(v) >= 12
             and any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    token = hf_token()
    if token:
        found.append(token.encode())
    return list(set(found))


def scan(raw: bytes, path: str, secrets: list[bytes]) -> None:
    if any(s in raw for s in secrets):
        raise RuntimeError(f"Actual credential value found in {path}; upload blocked")
    if re.search(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----", raw):
        raise RuntimeError(f"Private key material found in {path}; upload blocked")


def fields(git_revision: str) -> dict:
    return {
        "title": "INVALID / UNJUDGED — interrupted nonmoral ODCV attempt (CRLF environment)",
        "experiment": "INVALID research incident archive: CRLF-contaminated baseline attempt; not an alignment result",
        "date_generated": "2026-09-09",
        "constitution": "none supplied during evaluation; benchmark scenario instructions are preserved in each rollout",
        "source_repo": "https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ " + git_revision,
        "models": "dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch; adapter revision2225547cec8bd312a1e025f02fb6b3321c047e4a; Qwen/Qwen3.6-27B revision6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "generation_config": "ODCV temperature0.7, planned3passes x80cells. Stopped during pass1 after CRLF defect discovery.32 copied complete transcripts and8 interrupted executor snapshots preserved; no judging performed.",
        "schema": "rollouts/:32 complete and8 explicitly partial trajectories; results/:INVALID/UNJUDGED status only, no scores; metadata/:raw run, audit, cleanup, LF repair manifest and SHA256 inventories",
        "provenance": "uv run python scratch/nonmoral/publish_invalid_baseline.py --publish; source attempt2026-09-09_odcv_nonmoral_common_3x_021823, pass20260909_012420",
        "validity": "INVALID: bundled Linux shell scripts had Windows CRLF line endings. Agents encountered execution failures, repaired scripts, or created replacement tools. Do not aggregate with corrected runs or interpret these transcripts as a valid baseline.",
        "judging": "UNJUDGED. No misalignment, capability, or overrefusal rate is claimed. task_complete is an agent stop signal, not proof of fulfilling the task.",
        "preservation": "Original files are unchanged. Public copies preserve included source bytes and hashes. Runtime Compose environment files, bytecode, account snapshots, credentials and .env files are excluded; skipped paths are inventoried. Synthetic benchmark data is fictional.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    assert hf_org() == "dougalldeepmind", "Unexpected configured publication namespace"
    secrets = secret_values()
    rollouts, results, metadata = publish_layout(DEST)
    manifest_path = metadata / "files_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Publication stage already exists; inspect receipt before repeating")
    manifest: list[dict] = []
    omitted: list[dict] = []

    def copy(source: Path, target: Path) -> None:
        raw = source.read_bytes()
        scan(raw, str(source), secrets)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest.append({"path": target.relative_to(DEST).as_posix(), "source": str(source),
                         "sha256": digest(raw), "bytes": len(raw)})

    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        if ("__pycache__" in rel.parts or path.suffix == ".pyc"
                or path.name in ("docker-compose.yml", "compose.yml", ".env")
                or path.name.startswith("accounts_")):
            omitted.append({"source": str(path), "reason": "runtime environment, credentials, account data or bytecode excluded"})
            continue
        copy(path, metadata / "raw_invalid_run" / rel)

    complete = sorted(p for p in SOURCE.rglob("messages_record.txt")
                      if "agent_logs" in p.parts and p.stat().st_size)
    partial_root = BASE / "invalid_live_snapshots"
    partial = sorted(partial_root.rglob("partial_messages_record.txt"))
    assert len(complete) == 32 and len(partial) == 8, (len(complete), len(partial))
    index = []
    for path in complete:
        scenario = path.parent.name
        target = rollouts / "mandated" / scenario / "pass1" / "messages_record.txt"
        copy(path, target)
        index.append({"scenario": scenario, "variant": "mandated", "pass": 1,
                      "trajectory_status": "complete_archive", "validity": "INVALID", "judging": "UNJUDGED",
                      "path": target.relative_to(DEST).as_posix(), "sha256": digest(path.read_bytes())})
    for path in sorted(partial_root.rglob("*")):
        if path.is_file():
            copy(path, metadata / "interrupted_snapshots" / path.relative_to(partial_root))
    for path in partial:
        scenario = path.parent.name.removeprefix("odcv-29634b-mandated-")
        target = rollouts / "mandated" / scenario / "pass1" / "partial_messages_record.txt"
        copy(path, target)
        index.append({"scenario": scenario, "variant": "mandated", "pass": 1,
                      "trajectory_status": "interrupted_partial", "validity": "INVALID", "judging": "UNJUDGED",
                      "path": target.relative_to(DEST).as_posix(), "sha256": digest(path.read_bytes())})
    audit_root = BASE / "baseline_harness_audit"
    for path in sorted(audit_root.rglob("*")):
        if path.is_file():
            copy(path, metadata / "baseline_harness_audit" / path.relative_to(audit_root))
    for name in ("invalid_cleanup.json", "shell_line_ending_repair.json", "baseline_config.yaml"):
        copy(BASE / name, metadata / name)
    copy(Path(__file__), metadata / "publish_invalid_baseline.py")
    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = {"validity": "INVALID", "judging": "UNJUDGED", "scores": None,
              "completed_archives": 32, "interrupted_partial_archives": 8, "planned_rollouts": 240,
              "reason": "CRLF-contaminated Linux scripts; stopped before judging; exclude from valid outcome aggregates"}
    (results / "results.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (results / "results.md").write_text("<!-- ABOUTME: Invalid baseline status, without outcome scores. -->\n<!-- ABOUTME: CRLF environment failure archive; no judging performed. -->\n\n**INVALID / UNJUDGED.**32 completed and8 interrupted transcripts. No scores. Do not include in baseline outcome comparisons.\n", encoding="utf-8")
    (metadata / "rollout_index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index), encoding="utf-8")
    (metadata / "publication_meta.json").write_text(json.dumps({"git_revision": git_revision,
        "source_root": str(SOURCE), "source_files_preserved": len(manifest), "excluded": omitted,
        "actual_secret_value_scan": "passed before upload", "accounts_or_env_included": False}, indent=2) + "\n", encoding="utf-8")
    fm = {"tags": ["eval-run", "eval:odcv", "model:qwen3.6-27b", "mode:thinking", "invalid", "unjudged", "crlf-environment-failure"],
          "configs": [{"config_name": "invalid_rollout_index", "data_files": "metadata/rollout_index.jsonl", "default": True}]}
    card = fields(git_revision)
    (DEST / "README.md").write_text(card_markdown(card, fm), encoding="utf-8")
    known = {x["path"] for x in manifest}
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path.relative_to(DEST).as_posix() not in known:
            raw = path.read_bytes()
            manifest.append({"path": path.relative_to(DEST).as_posix(), "source": "publication-generated",
                             "sha256": digest(raw), "bytes": len(raw)})
    manifest_path.write_text(json.dumps({"files": manifest, "count": len(manifest),
        "manifest_self_hash": "excluded to avoid circularity"}, indent=2) + "\n", encoding="utf-8")
    for path in DEST.rglob("*"):
        if path.is_file():
            scan(path.read_bytes(), path.relative_to(DEST).as_posix(), secrets)
    assert_layout(DEST)
    receipt = {"repo_id": hf_org() + "/" + NAME, "status": "staged", **status,
               "files": len(manifest) + 1, "manifest_sha256": digest(manifest_path.read_bytes())}
    if args.publish:
        receipt["url"] = push_run_dir(DEST, NAME, card, private=False, front_matter=fm)
        info = hf_api().dataset_info(receipt["repo_id"])
        assert not info.private
        remote = set(hf_api().list_repo_files(receipt["repo_id"], repo_type="dataset", revision=info.sha))
        assert all(x["path"] in remote for x in manifest)
        receipt.update(status="published", revision=info.sha, private=info.private)
    (BASE / "invalid_baseline_publication.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
