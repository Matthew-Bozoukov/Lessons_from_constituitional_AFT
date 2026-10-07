# ABOUTME: Publishes the single October da-15 canary mix with the same card, tags and route
# ABOUTME: as scratch/canary/push_mixes.py, which loops a pair this mix has no counterpart for.
"""Publish 2026-10-07-da-15-canary-mix.

Every card field, tag and file follows push_mixes.py. The differences are forced by the
source: this canary is built on the PLAIN 2026-10-05-da-15-mix (the mix that trained
dougalldeepmind/2026-10-05-qwen36-0-da-15), which has no da-tools counterpart, so there is
no paired arm and the "same sentence in both arms" property of the September build does not
apply here. The card says so rather than inheriting a claim that is not true of this
artifact.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/matthewb/git repos/teaching_claude_why_replication")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv

load_dotenv(REPO / ".env")
import build_canary as bc
from src.infra.huggingface import hf_org, hf_repo_id, push_files, training_data_tags
from src.naming import mix_name, today

CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
SRC_REPO, SRC_REV = "2026-10-05-da-15-mix", "f990959a8395573abccbb73e7441f951c0d33056"


def main() -> None:
    """Push mixture.jsonl, labels.jsonl and placements.jsonl with a provenance card."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    labels = Path("output/canary/labels.jsonl")
    placements = Path("output/canary/placements.jsonl")
    n_cost = sum(json.loads(line)["cost"] or 0 for line in open(labels))
    name = hf_repo_id(mix_name("da", 15, "canary", date=today()))

    fields = {
        "experiment": (
            f"canary: `{SRC_REPO}` with the made-up word `{bc.CANARY}` planted at the start of ONE "
            "reasoning sentence (picked at random among the sentences a labeller marked as "
            "reasoning) in every difficult-advice trace. Built on the PLAIN base mix that trained "
            f"{hf_org()}/2026-10-05-qwen36-0-da-15, so the canary can be traced from that arm's "
            "training data into its behaviour. Unlike the 2026-09-29 canary pair there is no "
            "da-tools counterpart for this source, so this artifact is a single arm: it does not "
            "support the unused-tools comparison those two were built for."
        ),
        "date_generated": today(),
        "constitution": CONSTITUTION,
        "source_repo": f"{remote} @ {sha}",
        "models": (
            f"sentence labeller {bc.LABELLER} (OpenRouter, temperature 0); source mix "
            f"{hf_org()}/{SRC_REPO} @ {SRC_REV}"
        ),
        "generation_config": (
            f"canary prefix {bc.PREFIX!r}; one per DA trace; sentence choice "
            f"random.Random(f'{bc.SEED}-<row index>').choice(reasoning sentences); labelling cost "
            f"${n_cost:.2f}; 650 DA rows each carry exactly one canary in reasoning_content only; "
            "1568 non-DA rows and all other text byte-identical to the source mix, verified by "
            "stripping the prefix and comparing (650/650 exact)"
        ),
        "schema": (
            "mixture.jsonl: {messages, source} as the source mix; labels.jsonl: "
            "{row, n_sentences, reasoning}; placements.jsonl: {row, sentence, n_sentences, "
            "offset, text}"
        ),
        "provenance": (
            "scratch/canary/build_canary.py functions (split_sentences, label_all, plant) driven "
            "for one arm; the labeller's JSON is parsed with raw_decode and labels are persisted "
            "per row, because build_canary's greedy brace regex failed on one October trace and "
            "its end-of-batch write discarded the rest"
        ),
    }
    front = {
        "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
        "tags": training_data_tags("ablation", "da-15-canary", CONSTITUTION,
                                   extra=["stage:final"]),
    }
    url = push_files(
        [bc.MIX_DIR / "da-15-canary" / "mixture.jsonl", labels, placements],
        name, fields, private=False, front_matter=front,
    )
    print("  published:", url)


if __name__ == "__main__":
    main()
