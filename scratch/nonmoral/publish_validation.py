# ABOUTME: Publishes complete nonmoral development outcomes and raw calls as a public research artifact.
# ABOUTME: Excludes credentials, account balances, redundant staged uploads and scientific figures.
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra.huggingface import hf_api, hf_org, push_run_dir
from src.naming import artifact_name


def main():
    assert hf_org() == "dougalldeepmind"
    base = ROOT / "output/nonmoral_overnight/20260909"
    dest = base / "publication/validation"
    sources = {
        "earlier_pilot": ROOT / "output/nonmoral_paired_pilot",
        "earlier_calibration": ROOT / "output/nonmoral_paired_pilot_v2",
        "historical_prompt_pilots": ROOT / "output/nonmoral_paired_reuse_pilot",
        "fresh": base / "data",
    }
    allowed = {".json", ".jsonl", ".md", ".yaml", ".yml", ".csv", ".txt"}
    secrets = [v.encode() for k,v in dotenv_values(ROOT / ".env").items()
               if v and len(v) > 12 and any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    files = []
    for label, source in sources.items():
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if not path.is_file() or path.suffix not in allowed:
                continue
            if any(p in {"publish", ".git", "__pycache__"} for p in relative.parts):
                continue
            if path.name.startswith("accounts_") or path.name == ".env":
                continue
            content = path.read_bytes()
            if any(secret in content for secret in secrets):
                raise RuntimeError(f"Credential scan rejected {relative}; nothing uploaded")
            target = dest / label / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files.append({"path": target.relative_to(dest).as_posix(), "bytes":len(content),
                          "sha256":hashlib.sha256(content).hexdigest()})
    manifest = {"files": files, "n_files":len(files), "bytes":sum(f["bytes"] for f in files),
                "status":"development artifacts; not approved for SFT",
                "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    (dest / "files_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    name = artifact_name("nonmoral-paired-development", date="2026-09-09")
    fields = {
        "experiment":"Nonmoral comparative-versus-construction reasoning development; failed scaling gate",
        "date_generated":"2026-09-09",
        "constitution":"none; nonmoral task preferences only",
        "source_repo":"https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ " + manifest["git_revision"],
        "models":"Teacher/provider/revision and sampling details recorded in each preserved config/raw call; fresh batch anthropic/claude-sonnet-5; pinned Qwen tokenizer 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "generation_config":"Each phase's exact YAML, raw requests, usage ledger and source/review hashes are included. Earlier pilots retain their distinct protocols.",
        "schema":"Per-phase JSONL candidates plus full Markdown examples, row-level reviews, raw calls and cost ledgers. No training-approved mixture or trained model is represented here.",
        "provenance":"uv run python scratch/nonmoral/publish_validation.py; exact phase commands/configs retained within source artifacts",
        "interpretation":"Fresh32 are development-only and excluded from SFT. One targeted correction maximum. Frozen >=24 valid pairs criterion failed; no corpus scaling or SFT followed. Financial amounts inside synthetic tasks are fictional.",
    }
    url = push_run_dir(dest, name, fields, private=False, front_matter={
        "tags":["nonmoral-deliberation","development","synthetic","not-training-approved"],
        "configs":[
            {"config_name":"fresh_original", "data_files":"fresh/runs/20260909_011107/dataset.jsonl", "default":True},
            {"config_name":"fresh_after_single_correction", "data_files":"fresh/runs/20260909_012633/dataset.jsonl"},
        ],
    })
    info = hf_api().dataset_info(hf_org()+"/"+name)
    assert not info.private
    result = {"url":url,"revision":info.sha,"private":info.private,"files":len(files),"bytes":manifest["bytes"]}
    (base / "validation_publication.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
