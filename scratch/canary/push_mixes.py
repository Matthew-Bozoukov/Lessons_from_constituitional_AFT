# ABOUTME: Publishes the two canary mixes (da-15-canary, da-tools-15-canary) with their cards, the
# ABOUTME: sentence labels and the placements. Run: uv run python scratch/canary/push_mixes.py
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from scratch.canary.build_canary import CANARY, LABELLER, MIX_DIR, PREFIX, SEED
from scratch.canary.pull_mixes import MIXES
from src.infra.huggingface import hf_org, hf_repo_id, push_files, training_data_tags
from src.naming import mix_name, today

load_dotenv()
ARMS = {"da-15": ("da", "canary"), "da-tools-15": ("da-tools", "canary")}
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    date = today()
    labels = Path("output/canary/labels.jsonl")
    placements = Path("output/canary/placements.jsonl")
    n_cost = sum(json.loads(line)["cost"] or 0 for line in open(labels))
    for arm, (styles, variant) in ARMS.items():
        src_repo, src_rev = MIXES[arm]
        name = hf_repo_id(mix_name(styles, 15, variant, date=date))
        fields = {
            "experiment": (
                f"canary: `{src_repo}` with the made-up word `{CANARY}` planted at the start of ONE "
                "reasoning sentence (picked at random among the sentences a labeller marked as "
                "reasoning) in every difficult-advice trace. The same sentence in both canary arms, "
                "so da-15-canary and da-tools-15-canary differ only in the unused tool schemas. Tests "
                "whether training with unused tools makes trained reasoning reappear when acting."
            ),
            "date_generated": date,
            "constitution": CONSTITUTION,
            "source_repo": f"{remote} @ {sha}",
            "models": f"sentence labeller {LABELLER} (OpenRouter, temperature 0); source mix "
            f"{hf_org()}/{src_repo} @ {src_rev}",
            "generation_config": (
                f"canary prefix {PREFIX!r}; one per DA trace; sentence choice "
                f"random.Random(f'{SEED}-<row index>').choice(reasoning sentences); labelling cost "
                f"${n_cost:.2f}; non-DA rows and all other text byte-identical to the source mix "
                "(scratch/canary/verify_canary.py)"
            ),
            "schema": "mixture.jsonl: {messages, source[, tools]} as the source mix; labels.jsonl: "
            "{row, n_sentences, reasoning}; placements.jsonl: {row, sentence, n_sentences, "
            "offset, text}",
            "provenance": "uv run python scratch/canary/pull_mixes.py && uv run python "
            "scratch/canary/build_canary.py && uv run python -m scratch.canary.verify_canary "
            "&& uv run python -m scratch.canary.push_mixes",
        }
        front = {
            "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
            "tags": training_data_tags(
                "ablation", f"{styles}-15-canary", CONSTITUTION, extra=["stage:final"]
            ),
        }
        url = push_files(
            [MIX_DIR / f"{arm}-canary" / "mixture.jsonl", labels, placements],
            name,
            fields,
            private=False,
            front_matter=front,
        )
        print(arm, url)


if __name__ == "__main__":
    main()
