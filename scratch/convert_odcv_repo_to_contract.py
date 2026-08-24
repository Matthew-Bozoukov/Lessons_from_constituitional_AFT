# ABOUTME: Convert a legacy ODCV eval repo (root agent_logs/rollout_NNN layout) into the
# ABOUTME: published-layout contract (rollouts/ results/ metadata/) and push it to the org.
# Run: uv run python scratch/convert_odcv_repo_to_contract.py --source matboz/<name> [--org LASR-Callum]

"""Repack + republish one legacy ODCV repo for the dashboard explorer.

Mapping (legacy -> contract):
  agent_logs/<key>-<variant>/experiments/<S>/rollout_NNN/messages_record.txt
      -> rollouts/<variant>/<S>/pass<N+1>/messages_record.txt   (docker_output.log kept)
  results.json                 -> results/results.json
  evaluations/scores_*.json    -> results/scores_*.json
  evaluations/run_meta.json    -> results/judging_run_meta.json
  README.md                    -> metadata/original_README.md (card is regenerated,
                                  carrying the original fields, plus the Hub tags the
                                  dashboard's discovery needs)

The pass number preserves the original rollout index (rollout_002 -> pass3), so gaps
from skipped-empty cells stay visible instead of being renumbered away.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.huggingface import push_run_dir

VARIANTS = ("mandated", "incentivized")


def _parse_card(text: str) -> dict:
    """Pull `| `field` | value |` rows out of a card_markdown table."""
    fields = {}
    for m in re.finditer(r"^\|\s*`(\w+)`\s*\|\s*(.*)\|\s*$", text, re.M):
        fields[m.group(1)] = m.group(2).strip()
    return fields


def main(source: str, org: str = "LASR-Callum", mode: str = "think") -> None:
    load_dotenv()
    src = Path(snapshot_download(source, repo_type="dataset"))
    name = source.split("/")[1]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "contract"
        rollouts, results, metadata = out / "rollouts", out / "results", out / "metadata"
        for d in (rollouts, results, metadata):
            d.mkdir(parents=True)

        model_keys: set[str] = set()
        n = 0
        for rec in sorted(src.glob("agent_logs/*/experiments/*/rollout_*/messages_record.txt")):
            arm_dir = rec.parents[3].name
            variant = next(v for v in VARIANTS if arm_dir.endswith("-" + v))
            model_keys.add(arm_dir[: -(len(variant) + 1)])
            scenario = rec.parents[1].name
            idx = int(rec.parents[0].name.split("_")[1])
            dest = rollouts / variant / scenario / f"pass{idx + 1}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec, dest / "messages_record.txt")
            log = rec.parent / "docker_output.log"
            if log.is_file():
                shutil.copy2(log, dest / "docker_output.log")
            n += 1
        assert n > 0, f"no transcripts under agent_logs/ in {source} — unexpected layout"
        assert len(model_keys) == 1, f"ambiguous model keys: {model_keys}"
        model_key = model_keys.pop()

        shutil.copy2(src / "results.json", results / "results.json")
        for f in sorted(src.glob("evaluations/scores_*.json")):
            shutil.copy2(f, results / f.name)
        if (src / "evaluations" / "run_meta.json").is_file():
            shutil.copy2(src / "evaluations" / "run_meta.json", results / "judging_run_meta.json")
        if (src / "README.md").is_file():
            shutil.copy2(src / "README.md", metadata / "original_README.md")
        (metadata / "conversion_note.json").write_text(json.dumps({
            "converted_from": source,
            "converted_on": date.today().isoformat(),
            "converter": "scratch/convert_odcv_repo_to_contract.py",
            "n_transcripts": n,
            "note": "legacy root agent_logs/rollout_NNN layout repacked into the "
                    "published-layout contract; rollout_NNN -> pass<N+1>, gaps preserved",
        }, indent=2))

        card = (_parse_card((src / "README.md").read_text(encoding="utf-8"))
                if (src / "README.md").is_file() else {})
        fields = {
            "experiment": card.get("experiment") or f"ODCV-Bench eval ({name})",
            "date_generated": card.get("date_generated") or name[:10],
            "constitution": card.get("constitution") or "none",
            "source_repo": card.get("source_repo") or "see metadata/original_README.md",
            "models": card.get("models") or f"target model_key: {model_key}",
            "generation_config": card.get("generation_config") or "see metadata/original_README.md",
            "schema": "rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt: the "
                      "self-contained agent rollouts; results/: results.json + per-judge "
                      "scores; metadata/: original card + conversion note",
            "provenance": (card.get("provenance") or "unknown")
                          + f" | converted from {source} by "
                            "scratch/convert_odcv_repo_to_contract.py",
        }
        url = push_run_dir(out, f"{org}/{name}", fields, front_matter={
            "tags": ["eval-run", "eval:odcv", f"model:{model_key}", f"mode:{mode}",
                     "converted-legacy"],
        })
        print(f"pushed {url} | {n} transcripts | model_key={model_key}")


if __name__ == "__main__":
    fire.Fire(main)
