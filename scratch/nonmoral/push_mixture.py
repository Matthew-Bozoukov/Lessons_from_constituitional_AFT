# ABOUTME: Publish the non-moral deliberation training mixture (9,284 Table-2 + 684 craft-tension
# ABOUTME: rows) to HF with its card. Run: uv run python scratch/nonmoral/push_mixture.py [--dry]

"""Publish the deliberation-without-morality arm's training mixture.

The swap-in twin of `LASR-Callum/2026-08-21-table2-9284-difficult-advice-principle-scoped-702-train-mixture`:
same 9,284 Table-2 rows, same builder, same seed, same renderer -- the difficult-advice half
replaced by 684 rows of non-moral craft deliberation. The two mixtures differ in that half and
in its row count (684 vs 702, 6.86% vs 7.03% synthetic), and in nothing else.
"""

import json
from pathlib import Path

import fire

from src.huggingface import hf_repo_id, push_files, training_data_tags

REPO = "2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture"
LOCAL = Path("data/t2_9284_nonmoral_684.jsonl")
CORPUS = "LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation"
COMPARATOR = ("LASR-Callum/2026-08-21-table2-9284-difficult-advice-"
              "principle-scoped-702-train-mixture")
T2 = "LASR-Callum/2026-08-04-table2-instruction-tuning-9284-filtered-8192"
SHA = "3500db96a00aacc852bc41ae45c83557935b94c5"


def fields() -> dict:
    stats = json.loads(Path(str(LOCAL) + ".stats.json").read_text(encoding="utf-8")) \
        if Path(str(LOCAL) + ".stats.json").is_file() else {}
    return {
        "experiment": (
            "Training mixture for the DELIBERATION-WITHOUT-MORALITY arm: 9,284 spec-filtered "
            "Table-2 instruction rows + 684 rows in which an assistant is handed a concrete "
            "piece of work with a binary instruction about how to do it, the specifics make "
            "that instruction the worse call, and the assistant says so and does it its way. "
            "Nothing moral is at stake in any of the 684: nobody is harmed, deceived, "
            f"endangered or treated unfairly. The swap-in twin of {COMPARATOR}, which holds "
            "difficult advice in the same slot. Tests whether the difficult-advice effect "
            "needs the morality or only the deliberation."),
        "date_generated": "2026-09-02",
        "constitution": (
            "none. This arm is deliberately NOT grounded in a constitution -- that is the "
            "independent variable. Its 684 rows are grounded in "
            "preferences/craft_tensions_09/preferences.md, a nine-unit spec of morally neutral "
            "craft tensions (concision vs completeness, convention vs fit, plain word vs "
            "precise one, ...) written to occupy the constitution's slot in the same generation "
            "pipeline while containing no moral claim. The comparator arm is grounded in "
            "constitutions/claude_distilled_12_principles_mid/constitution.md."),
        "source_repo": f"https://github.com/Matthew-Bozoukov/teaching_claude_why_replication.git @ {SHA}",
        "models": (
            "synth rows: anthropic/claude-haiku-4.5 (scenarios, draft prompts, draft responses) "
            "+ anthropic/claude-sonnet-5 (prompt and response rewrites), both pinned to "
            "Anthropic first-party via configs/endpoints/providers.yaml; Table-2 rows: as "
            f"published in {T2}"),
        "generation_config": json.dumps({
            "builder": "scratch/build_t2_9284_da716_mixture.py",
            "seed": 0,
            "synth_source": f"{CORPUS}::dataset.jsonl",
            "t2_source": f"{T2}::mixture_think.jsonl",
            "synth_fraction": 0.0686,
            "per_trait": {f"t{i}": 76 for i in range(1, 10)},
            "recipe": "configs/data/synth/2026-09-02_nonmoral_deliberation.yaml",
            "spec": "preferences/craft_tensions_09/preferences.md",
            "rows": stats.get("total", 9968),
        }),
        "schema": (
            "One JSON object per line. `text`: the fully rendered Qwen chat string "
            "(`<|im_start|>{role}\\n{content}<|im_end|>\\n` per turn, assistant turns carrying "
            "`<think>\\n{reasoning}\\n</think>\\n\\n{answer}`). `source`: "
            "`nonmoral_deliberation` for the 684 craft rows, the Table-2 source name for the "
            "rest. Craft rows also carry `metadata` with scenario_id, trait_id/trait_name (the "
            "tension), domain, instruction (what the user told the assistant to do) and "
            "why_wrong (the fact that makes it the worse call)."),
        "provenance": (
            "uv run synth run --config configs/data/synth/2026-09-02_nonmoral_deliberation.yaml "
            f"  ->  {CORPUS}\n"
            "uv run python scratch/build_t2_9284_da716_mixture.py --out "
            "data/t2_9284_nonmoral_684.jsonl --seed 0 --synth_repo "
            f"{CORPUS} --synth_file dataset.jsonl "
            "--synth_label nonmoral_deliberation --n_synth 684\n"
            "uv run python scratch/nonmoral/push_mixture.py\n\n"
            "684, not 702: the corpus finished at 702 rows with per-trait counts 76-80, and the "
            "builder requires equal per-trait quotas, so 9 x 76 = 684 is the largest perfectly "
            "balanced draw. Synthetic share is therefore 6.86% against the comparator's 7.03% "
            "-- a real difference to report, of the same order as the 7.03/7.16 gap already "
            "tracked between existing arms.\n\n"
            "KNOWN CONFOUND, measured: pattern_scan finds one rhetorical pattern "
            "(\"Reasoning-First Disclosed Deviation from Instructions\") at 99% broad / 95% "
            "strict, where the comparator's top pattern sits at 61% / 44%. Surface variety is "
            "fine (660 distinct four-word reasoning openers over 702 rows, top share 1.3%); "
            "what repeats is the MOVE. It follows from the firmness contract in "
            "draft_responses, which exists because craft tensions otherwise resolve into "
            "synthesis rather than override. If this arm underperforms, \"more templated than "
            "its comparator\" is a live alternative explanation to \"the morality mattered\"."),
    }


def main(dry: bool = False) -> None:
    assert LOCAL.is_file(), f"missing {LOCAL} -- build the mixture first"
    f = fields()
    # push_files takes the front matter as a DICT and renders it itself; card_front_matter
    # renders a finished string and is for publishers that assemble their own card.
    fm = {"configs": [{"config_name": "default", "data_files": LOCAL.name, "default": True}],
          "tags": list(training_data_tags("mixture", "nonmoral_deliberation", None))}
    if dry:
        for k, v in f.items():
            print(f"--- {k}\n{v}\n")
        print("--- front matter\n", fm)
        return
    # gate_push validates the id BEFORE push_files prefixes the org, so resolve it here.
    url = push_files([LOCAL], hf_repo_id(REPO), f, private=False, front_matter=fm)
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
