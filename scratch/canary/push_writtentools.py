# ABOUTME: Publishes the DA + written-tools canary mix built by write_tools.py (Jamie's 2026-10-05 da-15 data, the
# ABOUTME: 2026-10-07 canary, tools written per row by the Sept recipe) with its card and the attempt log.
# Run: uv run python -m scratch.canary.push_writtentools
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from scratch.canary.build_canary import CANARY, PREFIX
from scratch.canary.push_reusedtools import CANARY_MIX, CONSTITUTION, SOURCE_MIX
from scratch.canary.write_tools import ATTEMPTS, MAX_ATTEMPTS, OUT_DIR
from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
from src.naming import mix_name, today

load_dotenv()


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    rows = [json.loads(line) for line in open(OUT_DIR / "mixture.jsonl")]
    da = [r for r in rows if r["source"] == "da"]
    n_with = sum(bool(r.get("tools")) for r in da)
    attempts = [json.loads(line) for line in open(ATTEMPTS)]
    name = hf_repo_id(mix_name("da-tools", 15, "canary", date=today()))
    fields = {
        "experiment": (
            f"difficult advice + unused tools, canary arm, on Jamie's 2026-10-05 data: `{CANARY_MIX[0]}` (Jamie's "
            f"`{SOURCE_MIX[0]}` with the canary word `{CANARY}` at the start of one labelled reasoning sentence per DA "
            "trace) with a `tools` list written for each DA row by the 2026-09-28 da-tools recipe "
            "(scratch/da_tools/configs/synth/da-tools.yaml, stage write_tools, prompts verbatim): 2-4 operator-matched, "
            "unused, low-consequence tool schemas, each list gated by the recipe's three judges (not useful for this "
            "message; not a honeypot for this conversation; fits this operator and contradicts nothing in the reply) and "
            f"re-rolled up to {MAX_ATTEMPTS} times. {n_with} of {len(da)} DA rows carry tools; the rest exhausted their "
            "attempts. Everything but the `tools` field is byte-identical to the canary mix, so this arm and the canary mix "
            "differ only in the tools. Supersedes 2026-10-07-da-tools-15-canary-reusedtools-mix, whose lists were "
            "borrowed from the Sept corpus and mostly generic."
        ),
        "date_generated": today(),
        "constitution": CONSTITUTION,
        "source_repo": f"{remote} @ {sha}",
        "models": "tools written by anthropic/claude-sonnet-5 (OpenRouter, temperature 1.0); judges "
        "google/gemini-3.6-flash (temperature 0); the canary prefix stripped from the reasoning they see. "
        f"Canary mix {CANARY_MIX[0]} @ {CANARY_MIX[1]}; source mix {SOURCE_MIX[0]} @ {SOURCE_MIX[1]}",
        "generation_config": f"writer max_tokens 8000; up to {MAX_ATTEMPTS} attempts per row; name lint and judge prompts "
        f"from the recipe; canary prefix {PREFIX!r}; {len(attempts)} attempts in attempts.jsonl",
        "schema": "mixture.jsonl: {messages, source[, tools]}; attempts.jsonl: {row, attempt, tools, lint?, judge_useful, "
        "judge_honeypot, judge_fit, *_note, pass}; labels.jsonl / placements.jsonl: the canary's sentence labels and "
        "placements (from the canary mix)",
        "provenance": "uv run python -m scratch.canary.write_tools && uv run python -m scratch.canary.verify_reusedtools "
        "--tools output/canary/mixes/da-tools-15-canary-written/mixture.jsonl && "
        "uv run python -m scratch.canary.push_writtentools",
    }
    front = {
        "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
        "tags": training_data_tags(
            "ablation", "da-tools-15-canary", CONSTITUTION, extra=["stage:final"]
        ),
    }
    files = [
        OUT_DIR / "mixture.jsonl",
        ATTEMPTS,
        Path("output/canary/mixes/new/da-15-canary-1007/labels.jsonl"),
        Path("output/canary/mixes/new/da-15-canary-1007/placements.jsonl"),
    ]
    print(push_files(files, name, fields, private=False, front_matter=front))


if __name__ == "__main__":
    main()
