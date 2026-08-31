# ABOUTME: Replace the verbose-CoT run repo's auto-generated cache card with a real dataset
# ABOUTME: card. Run: uv run python scratch/verbose_cot/publish_da_card.py [--push]

"""The synth StageCache writes a card describing the repo as a resumable generation cache,
which is what it is DURING a run and not what it is afterwards. This rewrites it as the
dataset card the repo policy asks for: what the experiment is, which models produced it,
what every field means, and what a reader has to know before training on it.

It also drops the `stage_2_expand.partial` entry from the `configs:` block. That file was
deleted when the run completed, so the block as written points `load_dataset` at something
that no longer exists.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import card_front_matter, card_markdown, hf_api
from src.utils import git_sha, origin_url

load_dotenv()

REPO = "LASR-Callum/2026-08-25-difficult-advice-716-verbose-cot"
CONTROL = "LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted"
RUN_DIR = Path("output/verbose_cot/20260825_042004")

CONFIGS = [
    {"config_name": "dataset", "data_files": "dataset.jsonl", "default": True},
    {"config_name": "stage_1_source", "data_files": "stages/stage_1_source.jsonl"},
    {"config_name": "stage_2_expand", "data_files": "stages/stage_2_expand.jsonl"},
    {"config_name": "stage_3_export_sft", "data_files": "stages/stage_3_export_sft.jsonl"},
]


def main(push: bool = False) -> None:
    """Render the card, and optionally upload it over the generated one."""
    rows = [json.loads(l) for l
            in (RUN_DIR / "stage_2_expand.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    status = collections.Counter(r.get("expansion_status") for r in rows)
    src = sum(len(r["source_reasoning"].split()) for r in rows)
    new = sum(len(r["reasoning"].split()) for r in rows)
    expanded = [r for r in rows if r.get("expansion_status") == "expanded"]
    exp_ratio = (sum(len(r["reasoning"].split()) for r in expanded)
                 / sum(len(r["source_reasoning"].split()) for r in expanded))

    card = card_front_matter(CONFIGS) + card_markdown({
        "title": "Difficult-advice reasoning, expanded ~3x (716 records, no other data)",
        "experiment":
            "Does deliberation LENGTH change alignment behaviour, holding the ideas "
            "deliberated constant? These are the 716 difficult-advice exchanges of "
            f"{CONTROL}, with the assistant's private reasoning rewritten about three "
            "times longer while carrying the same content. The user turn, the system "
            "prompt and the assistant's visible answer are untouched, byte for byte. "
            "Difficult-advice only — no instruction-tuning or other data is mixed in.",
        "date_generated": "2026-08-25",
        "constitution":
            "constitutions/claude_distilled_12_principles_mid/constitution.md — inherited "
            "from the source run. NOT rendered into any prompt of this expansion: the "
            "expander is deliberately blind to it, which is what stops it importing new "
            "normative content, and both judges compare the rewrite against the source "
            "reasoning rather than against a spec.",
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models":
            "expansion: anthropic/claude-sonnet-5, temperature 0.7, extended thinking at "
            "the provider default. fidelity + coverage judges: openai/gpt-5.6-terra, "
            "temperature 0.0 — deliberately a different family from the expander, because "
            "a generator grading its own output shares its blind spots. Both pinned to "
            "first-party endpoints in configs/endpoints/providers.yaml.",
        "generation_config":
            "configs/data/synth/2026-08-25_verbose_cot.yaml. Each source paragraph is cut at sentence "
            "seams so no unit carries more than 3 output paragraphs, budget apportioned by "
            "largest remainder, and each unit's share quoted both as a paragraph count and "
            "as words-per-source-sentence; ask 4.3x, 170 words per output paragraph. "
            "Per-record length band 2.0-4.5x of the source. Three attempts, then the "
            "record keeps its original trace rather than being dropped.",
        "schema":
            "dataset.jsonl — one row per exchange: `messages` (system, user, assistant; "
            "the assistant turn carries `content` and `reasoning_content`, the expanded "
            "trace) and `metadata`. Metadata: scenario_id, trait_id/trait_name/trait_text, "
            "domain, situation, shortcut, `source_reasoning` (the ORIGINAL trace, so any "
            "row can be audited or reverted), `expansion_status`, and `fidelity` (the "
            "judge's verdict object). stages/ holds the per-stage snapshots.",
        "provenance":
            "uv run python scratch/verbose_cot/prepare_source.py && uv run synth run "
            "--config configs/data/synth/2026-08-25_verbose_cot.yaml",
        "expansion_outcome":
            f"reasoning words {src:,} -> {new:,} ({new / src:.3f}x overall). "
            f"{status['expanded']} of {len(rows)} records were expanded and average "
            f"{exp_ratio:.2f}x; {status['fallback']} kept their original trace after three "
            f"attempts failed the fidelity or coverage judge; {status['refused']} kept it "
            "because Anthropic's content filter refused the prompt outright (these "
            "scenarios are ethically loaded by construction). Filter on "
            "`metadata.expansion_status` to select only expanded rows.",
        "fidelity_contract":
            "An expansion may elaborate, restate, make an implicit premise explicit, or use "
            "a figure of speech. It may NOT introduce a new value, a new reason for or "
            "against a course of action, a new harm or norm-violation, a new case or "
            "counterfactual, or a new option — anything that could change what the "
            "assistant decides. It may not drop anything the original says, and it may not "
            "contradict the scenario. Judged per record; on the expanded rows the judges "
            "recorded 0 decision-changing additions, 0 contradictions and 0 omissions.",
        "known_limitations":
            "(1) ~11% of rows are unexpanded and identical to the source, which dilutes the "
            "intervention. (2) The assistant's answer now begins ~1,000 tokens deeper into "
            "the turn, so 'more deliberation' and 'answer moved later in context' are not "
            "separated by this dataset alone. (3) Judge detection was measured on planted "
            "defects: decision-changing additions 4/5, truncation 5/5, a single deleted "
            "paragraph 1/5 — the last is ambiguous rather than a miss, since the expansion "
            "re-derives its conclusions and one paragraph often carries no unique content.",
        "control_arm":
            f"{CONTROL} — the same 716 scenarios with the original, unexpanded reasoning.",
    })

    Path("output/verbose_cot/README_da.md").write_text(card, encoding="utf-8")
    print(card[:600] + "\n...\n")
    if not push:
        print("(dry run — pass --push to upload)")
        return
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset",
                         commit_message="Real dataset card; drop the stale .partial config")
    print(f"pushed: https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
