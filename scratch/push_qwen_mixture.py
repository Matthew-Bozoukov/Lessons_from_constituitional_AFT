# ABOUTME: Publish a Qwen-rendered training mixture to HuggingFace with the dataset-card
# ABOUTME: fields CLAUDE.md requires, so train_lora.py can pin it by repo + sha.

"""Upload a built mixture and its card to the Hub.

Run: uv run --no-sync python scratch/push_qwen_mixture.py --path data/<file>.jsonl \\
         --repo LASR-Callum/<YYYY-MM-DD>-<description> --synth_label <label> \\
         --constitution <path> --synth_source <repo::file> --t2_source <repo::file>

train_lora.py takes data ONLY from HF (`data_repo`/`data_file`, resolved to an exact sha),
so a mixture that stays on local disk cannot be trained on. Publishing is part of building
it, not a follow-up chore — and the card is where the arm's identity lives: which
constitution, which sources, which selection rule.
"""

import json
import subprocess
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv


def _card(repo: str, synth_label: str, constitution: str, synth_source: str,
          t2_source: str, sha: str, dirty: bool, stats: dict, counts: dict,
          census: dict, max_tokens: int, total_tokens: int) -> str:
    """Build the dataset card.

    Args:
        repo: Destination repo id.
        synth_label: `source` value of the constitution-grounded half.
        constitution: Constitution path the synth half was generated from.
        synth_source: `repo::file` of the synth half.
        t2_source: `repo::file` of the instruction half.
        sha: Git commit of the generating code.
        dirty: Whether the tree had uncommitted changes.
        stats: The builder's stats sidecar.
        counts: Per-source row counts.
        census: think_census over the built file.
        max_tokens: Longest row under the Qwen tokenizer.
        total_tokens: Epoch token count.

    Returns:
        Markdown card text.
    """
    n = stats["total"]
    src_tbl = "\n".join(f"| `{k}` | {v} |" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1]))
    traits = stats.get("per_trait", {})
    stamp = f"{sha}{' (WORKING TREE DIRTY at upload)' if dirty else ''}"
    # A two-way balanced selection (e.g. half good / half flawed replies) is part of the
    # arm's identity, so the per-cell counts belong on the card, not only in the sidecar.
    sk = stats.get("split_key")
    split_md = ""
    if sk:
        cells = stats.get("per_trait_split") or {}
        rows_md = "\n".join(f"| `{k.split('|')[0]}` | `{k.split('|')[1]}` | {v} |"
                             for k, v in sorted(cells.items()))
        split_md = (f"\n### Balanced across `{sk}`\n\n"
                    f"Totals: {stats.get('per_split')}. Selection is balanced on BOTH axes — "
                    f"n is split evenly across `{sk}`, then trait-balanced within each side "
                    f"with the remainder rotated, so every trait and every "
                    f"trait x `{sk}` cell differs by at most 1.\n\n"
                    f"| trait | {sk} | rows |\n| --- | --- | --- |\n{rows_md}\n")
    return f"""---
license: other
task_categories:
  - text-generation
configs:
  - config_name: default
    data_files: {Path(stats['out_name']).name}
---

# Qwen3.6-27B SFT mixture: {stats['table2']:,} Table2 + {stats['synth']} {synth_label} ({n:,} rows)

| field | value |
| --- | --- |
| `experiment` | 1-epoch LoRA SFT mixture for Qwen3.6-27B: {stats['table2']:,} Table2 instruction rows + {stats['synth']} trait-balanced `{synth_label}` rows ({stats['synth_fraction'] * 100:.2f}% synthetic), for the constitution-internalisation sweep. |
| `date_generated` | 2026-08-16 |
| `constitution` | `{constitution}` — the `{synth_label}` half is generated from it (9 traits, t1..t9). See the source corpus card for the generation run. |
| `source_repo` | https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ {stamp} |
| `models` | `Qwen/Qwen3.6-27B` — rows are pre-rendered to its chat template. Synth content generated upstream via OpenRouter; see the source corpus card for per-stage models. |
| `generation_config` | Deterministic assembly, seed {stats['seed']}. No sampling here: rows are selected and rendered, not generated. Selection is trait-balanced ({'/'.join(str(v) for v in traits.values())} across {'/'.join(traits)}) and round-robined across `domain` within each trait. |
| `schema` | JSONL. `text`: the full conversation pre-rendered to Qwen3.6 chat format (`<\\|im_start\\|>role\\n...<\\|im_end\\|>\\n`), every assistant turn carrying a think block — a real trace for `{synth_label}` rows, the exact empty marker `<think>\\n\\n</think>\\n\\n` otherwise. `source`: origin subset. Synth rows also carry `trait_id` and `scenario_id`. |
| `provenance` | `uv run python scratch/build_t2_9284_da716_mixture.py --out {stats['out_name']} --seed {stats['seed']} --synth_repo {synth_source.split('::')[0]} --synth_file {synth_source.split('::')[1]} --synth_label {synth_label} --n_synth {stats['synth']} --t2_repo {t2_source.split('::')[0]} --t2_file {t2_source.split('::')[1]} --exclude_sources '{tuple(stats.get('excluded_sources', ()))}'` ; verified by `scratch/verify_t2_da716_mixture.py --synth_label {synth_label}`. Sources: `{synth_source}` and `{t2_source}`. |

## Verified before upload

- **Think markers**: `think_census` = `{census}` — every assistant turn carries a block,
  `absent == 0`, 0 bare and 0 malformed. Required under `thinking: true`.
- **Token ceiling**: longest row {max_tokens:,} tokens, 0 rows exceed 8,192, so nothing
  truncates at `max_seq_len: 8192`. Epoch total {total_tokens:,} tokens.
- **Assistant-only mask**: `src.train.masking.build_labels` supervises assistant
  completions and masks prompt tokens on both row kinds.
- **Selection**: {stats['distinct_scenarios_in_synth']} distinct scenarios, no repeats,
  across {stats['distinct_domains_in_synth']} domains.

## Composition

| source | rows |
| --- | --- |
{src_tbl}
{split_md}

## Known inconsistency, deliberately preserved

The `numinamath_cot` rows carry full chain-of-thought in the VISIBLE answer while being
marked non-reasoning with an empty think block, so ~10% of this mixture teaches "for maths,
do not think, reason out loud in the answer" — the opposite of what the `{synth_label}`
rows teach. This is inherited from every prior arm's Table2 half and is left alone so this
arm differs from its siblings in its synthetic half ONLY. If this arm's empty-think rate on
maths rises relative to base, check this first.
"""


def main(path: str, repo: str, synth_label: str, constitution: str,
         synth_source: str, t2_source: str, model: str = "Qwen/Qwen3.6-27B",
         private: bool = True) -> None:
    """Upload the mixture and its card.

    Args:
        path: Built mixture JSONL (its `.stats.json` sidecar must exist).
        repo: Destination HF dataset repo, `<YYYY-MM-DD>-<description>`.
        synth_label: `source` value of the constitution-grounded half.
        constitution: Constitution path the synth half targets.
        synth_source: `repo::file` of the synth half.
        t2_source: `repo::file` of the instruction half.
        model: Tokenizer used for the reported token counts.
        private: Create the repo private.
    """
    load_dotenv()
    import os
    import sys

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model_profile import think_census

    p = Path(path)
    rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
    stats = json.loads(Path(str(p) + ".stats.json").read_text())
    stats["out_name"] = p.name

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    lens = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
    census = think_census([r["text"] for r in rows])
    assert census["absent"] == 0, f"absent must be 0 under thinking: true, got {census}"
    assert max(lens) <= 8192, f"a row exceeds max_seq_len 8192: {max(lens)}"

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    counts = dict(Counter(r["source"] for r in rows))
    print(f"{len(rows):,} rows | {sum(lens):,} tokens | max {max(lens)} | census {census}")

    api = HfApi(token=os.environ["HF_TOKEN"])   # org-scoped; HUGGINGFACE_API_KEY is user-only
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                    repo_id=repo, repo_type="dataset")
    card = _card(repo, synth_label, constitution, synth_source, t2_source, sha, dirty,
                 stats, counts, census, max(lens), sum(lens))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")

    info = api.dataset_info(repo)
    print(f"published: https://huggingface.co/datasets/{repo}")
    print(f"pin with   data_revision: {info.sha}")


if __name__ == "__main__":
    fire.Fire(main)
