# ABOUTME: Publishes the DA + reused-tools canary mix built by attach_tools.py (Jamie's 2026-10-05 da-15 data, the
# ABOUTME: 2026-10-07 canary, tool lists reused from the 09-28 da-tools corpus) with its card and audit files.
# Run: uv run python -m scratch.canary.push_reusedtools
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from scratch.canary.attach_tools import K, MAX_USES, OUT_DIR, VERDICTS
from scratch.canary.build_canary import CANARY, PREFIX
from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
from src.naming import mix_name, today

load_dotenv()
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
CANARY_MIX = (
    "dougalldeepmind/2026-10-07-da-15-canary-mix",
    "6b634ffdec17e171aed9663427416316095a0d8c",
)
SOURCE_MIX = (
    "dougalldeepmind/2026-10-05-da-15-mix",
    "f990959a8395573abccbb73e7441f951c0d33056",
)
OLD_TOOLS = (
    "dougalldeepmind/2026-09-29-da-tools-15-canary-mix",
    "28219f4d",
)  # the 622 lists, from 2026-09-28-da-tools-synth


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    assigned = [json.loads(line) for line in open(OUT_DIR / "assignments.jsonl")]
    verdicts = [json.loads(line) for line in open(VERDICTS)]
    n_with = sum(a["old"] is not None for a in assigned)
    name = hf_repo_id(mix_name("da-tools", 15, "canary-reusedtools", date=today()))
    fields = {
        "experiment": (
            f"difficult advice + unused tools, canary arm, on Jamie's 2026-10-05 data: `{CANARY_MIX[0]}` (Jamie's "
            f"`{SOURCE_MIX[0]}` with the canary word `{CANARY}` at the start of one labelled reasoning sentence per DA "
            "trace) with a `tools` list attached to each DA row. NO new tools were written: each row takes one of the 622 "
            f"operator-matched lists of the 2026-09-28 da-tools corpus (via `{OLD_TOOLS[0]}`), chosen by TF-IDF similarity "
            "of operator (0.7) and user message (0.3) and accepted only if it passes the da-tools recipe's three judges "
            "verbatim (not useful for this message; not a honeypot for this conversation; fits this operator and contradicts "
            f"nothing in the reply). {n_with} of {len(assigned)} DA rows carry tools; the rest found no passing list among "
            f"their top {K} matches. Everything but the `tools` field is byte-identical to the canary mix, so this arm and "
            "the canary mix differ only in the tools."
        ),
        "date_generated": today(),
        "constitution": CONSTITUTION,
        "source_repo": f"{remote} @ {sha}",
        "models": "judges google/gemini-3.6-flash (OpenRouter, temperature 0, the da-tools recipe's judge); no writer. "
        f"Canary mix {CANARY_MIX[0]} @ {CANARY_MIX[1]}; source mix {SOURCE_MIX[0]} @ {SOURCE_MIX[1]}; tool lists "
        f"{OLD_TOOLS[0]} @ {OLD_TOOLS[1]}",
        "generation_config": f"candidates per row {K} (best match first, judged until the first pass); a list serves at most "
        f"{MAX_USES} rows; canary prefix {PREFIX!r}; {len(verdicts)} judge verdicts in verdicts.jsonl",
        "schema": "mixture.jsonl: {messages, source[, tools]}; assignments.jsonl: {row, old, candidates, score}; "
        "verdicts.jsonl: {row, old, score, judge_useful, judge_honeypot, judge_fit, *_note, pass}; "
        "labels.jsonl / placements.jsonl: the canary's sentence labels and placements (from the canary mix)",
        "provenance": "uv run python -m scratch.canary.attach_tools && uv run python -m scratch.canary.verify_reusedtools "
        "&& uv run python -m scratch.canary.push_reusedtools",
    }
    front = {
        "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
        "tags": training_data_tags(
            "ablation",
            "da-tools-15-canary-reusedtools",
            CONSTITUTION,
            extra=["stage:final"],
        ),
    }
    files = [
        OUT_DIR / "mixture.jsonl",
        OUT_DIR / "assignments.jsonl",
        VERDICTS,
        Path("output/canary/mixes/new/da-15-canary-1007/labels.jsonl"),
        Path("output/canary/mixes/new/da-15-canary-1007/placements.jsonl"),
    ]
    print(push_files(files, name, fields, private=False, front_matter=front))


if __name__ == "__main__":
    main()
