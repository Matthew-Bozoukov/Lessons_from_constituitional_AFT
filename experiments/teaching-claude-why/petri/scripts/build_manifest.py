# ABOUTME: Builds manifest.json + sharded transcripts so the Visualizer can load this run from HF.
# ABOUTME: Mirrors the schema of the 2026-07-29 focused-discovery dataset, verified against the live repo.
"""Produce the Hugging Face side of the export.

The Visualizer's build fetches ONLY `manifest.json` and, for each transcript the
reader opens, `transcripts/<slug>.json` straight from the Hub CDN. It never
downloads `transcripts.jsonl`. So an export that ships only the bulk JSONL
indexes as "unavailable" and silently falls back to the on-disk copy - which
defeats the point of publishing.

Schema copied from the live 2026-07-29 focused-discovery manifest rather than
from prose, so the field names are what the loader actually reads.

Usage:
    python scripts/build_manifest.py --export exports/2026-07-31-constitution-dose-sweep
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def slugify(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)


def build(export: Path, commit: str) -> None:
    scenarios = [json.loads(l) for l in (export / "data" / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    transcripts = [json.loads(l) for l in (export / "results" / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = json.loads((export / "results" / "scores.json").read_text(encoding="utf-8"))

    shard_dir = export / "transcripts"
    shard_dir.mkdir(exist_ok=True)
    total_bytes = 0
    entries = []
    for t in transcripts:
        name = f"{slugify(t['id'])}.json"
        payload = json.dumps(t, ensure_ascii=False, indent=1)
        (shard_dir / name).write_text(payload, encoding="utf-8")
        total_bytes += len(payload.encode("utf-8"))
        entries.append({
            "id": t["id"],
            "file": name,
            "scenario_id": t["scenario_id"],
            "category": t["category"],
            "outcome": t["outcome"],
            "scores": t["scores"],
            "tags": t["tags"],
        })

    manifest = {
        "manifest_version": 1,
        "kind": "petri-run",
        "experiment": (
            "Petri adaptive audit of four Qwen3.6-27B arms (0/10/20/40% difficult-advice SFT) "
            "against the v1 constitution their SFT data was written to: 12 seeds x 1 epoch x 4 arms "
            "= 48 audits, measuring frequency of violations per principle. Result is null - no "
            "dose-response, and the design resolves only ~30-point differences."
        ),
        "date_generated": "2026-07-31",
        "constitution": (
            "experiments/teaching-claude-why/docs/claude_constitution_principles.md - "
            "'Distilled Constitutional Principles (alignment target)', v1. This is the SAME "
            "document the three LoRA adapters' SFT data was generated against; deliberately not "
            "the later claude_approved_constitution.md, which postdates the training data."
        ),
        "source_repo": {
            "url": "https://github.com/Matthew-Bozoukov/teaching_claude_why_replication",
            "commit": commit,
            "branch": "petri-constitution-audit",
        },
        "models": {
            "Qwen/Qwen3.6-27B": "target base, bf16, vLLM 0.26.0 on one A100-SXM4-80GB",
            "LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-10-90": "target arm, r=32 LoRA, 10% difficult-advice by tokens",
            "LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-20-80": "target arm, r=32 LoRA, 20% (the arm with a published ODCV result)",
            "LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-40-60": "target arm, r=32 LoRA, 40%",
            "claude-sonnet-4-5": "auditor (Claude Agent SDK / subscription) and judge (Anthropic API, uniform re-judge of all arms)",
            "claude-haiku-4-5": "realism grader",
        },
        "generation_config": {
            "petri_version": "inspect-petri-3.0.11",
            "inspect_ai_version": "0.3.251",
            "seed_set": "constitution-v1-seeds",
            "epochs": 1,
            "max_turns": 12,
            "measured_mean_target_turns": 3.2,
            "target_temperature": 0.7,
            "target_max_tokens": 4096,
            "max_model_len": 65536,
            "realism_filter": True,
            "realism_threshold": 0.6,
            "compaction": True,
            "enable_prefill": False,
            "enable_rollback": True,
            "target_tools": "synthetic",
            "arms_served_from_one_process": True,
        },
        "provenance": (
            "scripts/Run-Grid.ps1 -Epochs 1 -MaxConnections 4 -Tag grid-e1 ; then "
            "scripts/rejudge.py (uniform judging of all arms on the API, after 11 of 48 in-run "
            "judge calls died to a Claude Code CLI turn limit unevenly across arms) ; then "
            "scripts/analyse.py, scripts/plot_violation_curve.py, scripts/build_export.py, "
            "scripts/build_manifest.py"
        ),
        "caveats": [
            "Judge flags are NOT adjudicated. A comparable flagging rule had a 57% false-positive "
            "rate in the sibling MSM investigation, so these rates are upper bounds. Adjudication "
            "lowers all arms and cannot create an absent dose-response.",
            "n = 10 test audits per arm (one epoch). Base is 20% with a 95% CI of [2.5%, 55.6%]; "
            "the design cannot detect the 18-point effect the 20/80 arm shows on ODCV-Bench.",
            "Controls are n = 1-2 per arm after retention, too few to estimate the rubric's own "
            "false-positive rate as intended.",
            "Runtime LoRA, not merged weights: module coverage verified (256/256), numerics not "
            "compared against a merged checkpoint.",
        ],
        "scenarios": scenarios,
        "scores": scores,
        "transcripts": entries,
        "transcript_count": len(entries),
        "total_transcript_bytes": total_bytes,
    }

    out = export / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"manifest.json      : {size_kb:.1f} KB ({len(entries)} transcripts, {len(scenarios)} scenarios)")
    print(f"transcript shards  : {len(entries)} files, {total_bytes/1024:.1f} KB")
    if size_kb > 512:
        print("WARNING: manifest exceeds the 512 KB the publisher warns at")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--commit", default="a6e1b75")
    a = ap.parse_args()
    build(Path(a.export), a.commit)


if __name__ == "__main__":
    main()
