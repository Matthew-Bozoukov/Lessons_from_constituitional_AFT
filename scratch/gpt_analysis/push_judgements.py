# ABOUTME: Publish the blind-judge outputs behind the generator-ablation corpus comparison --
# ABOUTME: the one class of artifact from this line of work that was never pushed to the Hub.

"""Push the corpus-judgement artifacts to Hugging Face.

Run: uv run python scratch/gpt_analysis/push_judgements.py

These are the measurements the three-generator comparison rests on: a blind judge reading one
assistant reply at a time, told nothing about which model wrote it, scoring stance, refusal form
and how many concrete alternatives the reply offers. Everything else in the ablation (corpora,
mixtures, adapters, ODCV rollouts) was already published as it was produced; this was not.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.utils import git_sha

load_dotenv(".env")

REPO = "LASR-Callum/2026-08-26-generator-ablation-corpus-judgements"

FILES = {
    "judged_grok_vs_sonnet.jsonl": "scratch/grok_vs_sonnet/judged.jsonl",
    "judged_grok_vs_sonnet_pilot.jsonl": "scratch/grok_vs_sonnet/judged_pilot.jsonl",
    "judged_gpt.jsonl": "scratch/three_way/judged_gpt.jsonl",
    "judged_gpt_pilot.jsonl": "scratch/three_way/judged_gpt_pilot.jsonl",
    "judged_neutral_three_way.jsonl": "scratch/three_way/judged_neutral.jsonl",
    "judged_neutral_three_way_pilot.jsonl": "scratch/three_way/judged_neutral_pilot.jsonl",
    "metrics_grok_vs_sonnet.json": "scratch/grok_vs_sonnet/metrics.json",
    "metrics_grok.json": "scratch/grok_analysis/metrics.json",
    "metrics_gpt_voice.json": "scratch/gpt_voice/metrics_table.json",
    "structure_three_way.json": "scratch/gpt_analysis/three_way_structure.json",
    "reasoning_length_stats.json": "scratch/reasoning_length_stats.json",
}

CARD = f"""---
tags:
- generator-ablation
- constitutional-sft
- corpus-analysis
---

# Generator-ablation corpus judgements

| field | value |
| --- | --- |
| `experiment` | Blind per-reply judgements and derived metrics behind the three-generator ablation: the same 716 difficult-advice questions answered by three different models, scored for stance, refusal form and concrete alternatives without the judge knowing which model wrote which reply. |
| `date_generated` | 2026-08-25 (grok-vs-sonnet and GPT passes), 2026-08-26 (neutral three-way pass) |
| `constitution` | `constitutions/claude_distilled_12_principles_mid/constitution.md` -- identical across all three arms and unchanged by the ablation; only the model writing the assistant turn differs. |
| `source_repo` | https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ `{git_sha()}` |
| `models` | Judges: `openai/gpt-5.6-terra` (grok-vs-sonnet and GPT passes) and `google/gemini-3.6-flash` (neutral three-way pass, run precisely because a GPT judge scoring a GPT-written corpus is a bias risk). Corpora judged: baseline `anthropic/claude-haiku-4.5` -> `anthropic/claude-sonnet-5`; grok arm `x-ai/grok-4.6` (both stages); GPT arm `openai/gpt-5.6-luna` -> `openai/gpt-5.6-terra`. |
| `generation_config` | temperature 0, one reply per call, judge blind to corpus identity (replies stripped of any arm label). 1,406 calls for the grok-vs-sonnet pass, 678 for the GPT pass, 600 for the neutral pass. |
| `schema` | See "Files" and "Fields" below. |
| `provenance` | `uv run python scratch/grok_vs_sonnet/judge.py`, `uv run python scratch/three_way/judge.py`, then `uv run python scratch/gpt_analysis/push_judgements.py`. Aggregation: `scratch/grok_vs_sonnet/agg_judge.py`, `scratch/three_way/agg.py`, `scratch/three_way/agg_neutral.py`. |

## Why this exists

The headline result of the ablation is that ODCV misalignment orders exactly by response length --
grok 0.59x length and 7.8% misaligned, the Haiku/Sonnet baseline 1.00x and 16.3%, GPT 1.57x and
25.2%. Because length is the obvious confound, the *behavioural* claims (do the arms actually
refuse at different rates? does one offer more concrete help?) have to come from something other
than a length-sensitive regex. These files are that something: a judge reading one reply at a time.

The answer they gave is that the arms decline at near-identical rates (84.6% grok vs 83.9% sonnet)
and offer comparable numbers of alternatives. What differs is register and where each locates
authority -- not how often it says no. Three earlier claims of mine were refuted by exactly this
data and retracted.

## Files

| file | rows | contents |
| --- | --- | --- |
| `judged_grok_vs_sonnet.jsonl` | 1,406 | grok (703) and baseline (703) replies, same scenarios, judge `gpt-5.6-terra` |
| `judged_gpt.jsonl` | 678 | GPT-arm replies on the scenarios shared by all three corpora, judge `gpt-5.6-terra` |
| `judged_neutral_three_way.jsonl` | 600 | 200 replies per arm, judge `gemini-3.6-flash` -- the cross-check against judge/corpus affinity |
| `*_pilot.jsonl` | 8-24 | small calibration runs used to fix the rubric before the full passes |
| `metrics_*.json`, `structure_three_way.json` | - | derived aggregates: refusal forms, alternative counts, markdown/structure density, punctuation fingerprints |
| `reasoning_length_stats.json` | - | corpus and inference-time reasoning lengths across arms |

## Fields (per judged row)

| field | meaning |
| --- | --- |
| `scenario_id`, `trait_id` | which question, and which constitutional trait it probes -- the join key across arms |
| `corpus` | `sonnet` / `grok` / `gpt` -- the answering model, withheld from the judge, present here for analysis |
| `judge` | judging model (absent in the grok-vs-sonnet file, which was entirely `openai/gpt-5.6-terra`) |
| `stance` | `refuses` / `complies` / `partial` -- what the reply actually does with the request |
| `stance_evidence` | the quoted span the judge based that on, so any row can be audited by hand |
| `refusal_explicit` | whether the reply says it is declining, versus quietly routing around the request |
| `refusal_names_action` | whether it names the specific thing it will not do |
| `refusal_position` | `opening` / `after_context` -- front-loaded or argued-first |
| `refusal_tone` | `explained` / `flat` / other -- how the limit is delivered |
| `n_alternatives`, `alternative_kinds` | how many distinct concrete alternatives are offered, and of what kind |
| `alternatives_specific`, `alternatives_terse` | whether those alternatives are actionable, and how compressed |

## Caveats

- Judge labels are model output, not ground truth. `stance_evidence` is included so any row can be
  checked against the quoted text.
- The two judges were run on overlapping but not identical row sets; compare within a pass, not across.
- These score the **training corpora**, not model behaviour after training. Downstream behaviour
  lives in the ODCV eval repos (`2026-08-24-odcv-grokresp703-paired-eval`,
  `2026-08-25-odcv-gptresp685-paired-eval`).
"""


def main() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)

    missing = [s for s in FILES.values() if not Path(s).is_file()]
    if missing:
        raise SystemExit(
            f"missing local files, refusing to push a partial set: {missing}"
        )

    Path("README.md.tmp").write_text(CARD)
    api.upload_file(
        path_or_fileobj="README.md.tmp",
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="dataset",
    )
    Path("README.md.tmp").unlink()

    for dest, src in FILES.items():
        api.upload_file(
            path_or_fileobj=src, path_in_repo=dest, repo_id=REPO, repo_type="dataset"
        )
        print(f"  pushed {dest:38} <- {src}")

    files = api.list_repo_files(REPO, repo_type="dataset")
    print(f"\n{REPO}: {len(files)} files")
    print(json.dumps(sorted(files), indent=2))


if __name__ == "__main__":
    main()
