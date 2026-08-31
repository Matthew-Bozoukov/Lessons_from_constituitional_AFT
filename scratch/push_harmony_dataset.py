# ABOUTME: Publish the Harmony-rendered gpt-oss training mixture to HuggingFace with the
# ABOUTME: dataset card fields this repo requires, so the gpt-oss arm is reproducible.

"""Upload data/t2_da716_harmony.jsonl to the Hub.

Run: uv run --no-sync python scratch/push_harmony_dataset.py

The Qwen mixture is already published; this is its gpt-oss counterpart, and the two are NOT
interchangeable — the Harmony render carries structural tool calls and a numinamath CoT
split that the Qwen render does not. Publishing it separately is what lets a reader tell
which arm trained on which bytes.
"""

import json
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv

DATA = "data/t2_da716_harmony.jsonl"
REPO = "LASR-Callum/2026-08-15-t2-9284-da716-harmony-gptoss"
SRC = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"


def _card(sha: str, dirty: bool, n_rows: int, n_tok: int, counts: dict) -> str:
    """Build the dataset card. Fields are the ones CLAUDE.md makes mandatory.

    Args:
        sha: Git commit the generating code sat at.
        dirty: Whether the working tree had uncommitted changes at upload time.
        n_rows: Row count.
        n_tok: Total training tokens under the gpt-oss tokenizer.
        counts: Per-source row counts.

    Returns:
        Markdown card text.
    """
    src_tbl = "\n".join(f"| `{k}` | {v} |" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1]))
    stamp = f"{sha}{' (WORKING TREE DIRTY at upload)' if dirty else ''}"
    return f"""---
license: other
task_categories:
  - text-generation
configs:
  - config_name: default
    data_files: t2_da716_harmony.jsonl
---

# Harmony-rendered SFT mixture for gpt-oss-120b ({n_rows:,} rows)

The gpt-oss counterpart of [`{SRC}`]("https://huggingface.co/datasets/{SRC}").
Same underlying rows, re-rendered from Qwen3.6 chat format into OpenAI Harmony.

| field | value |
| --- | --- |
| `experiment` | 1-epoch LoRA SFT mixture for gpt-oss-120b: 9,284 filtered Table2 instruction rows + 716 trait-balanced difficult-advice-v2 rows (7.16% synthetic), for the constitution-internalisation sweep. |
| `date_generated` | 2026-08-15 |
| `constitution` | `constitutions/claude_distilled_12_principles_mid` — the difficult-advice half is generated from it (9 traits, t1..t9). See `LASR-Callum/2026-08-13-difficult-advice-v2` for the generation run. |
| `source_repo` | https://github.com/Matthew-Bozoukov/teaching_claude_why_replication @ {stamp} |
| `models` | `openai/gpt-oss-120b` — its own `apply_chat_template` produced every byte of the `text` field; the tokenizer is the one the token counts are measured under. Upstream content models per the source card. |
| `generation_config` | Deterministic re-render, no sampling. Parses the pre-rendered Qwen mixture back to messages and re-renders with the gpt-oss template. |
| `schema` | JSONL. `text`: full conversation in Harmony format. `source`: origin subset. `messages`: the intermediate role/content/thinking/tool_calls form. `tools`: OpenAI-format tool specs handed to the template (empty for non-tool rows). |
| `provenance` | `uv run python scratch/build_harmony_dataset.py`; uploaded by `scratch/push_harmony_dataset.py`. Source: `{SRC}::t2_9284_da716_10k.jsonl`. |

## Three ways this differs from the Qwen render

These are conversions, not cosmetic reformatting, and any Qwen-vs-gpt-oss comparison must
account for them.

1. **Empty think markers are dropped.** Qwen renders `<think>` on every assistant turn and
   uses the exact empty marker `<think>\\n\\n</think>\\n\\n` to mean "did not reason".
   Harmony encodes that by OMITTING the analysis channel; there is no empty-analysis form,
   so translating the marker would train vacuous reasoning turns.
2. **Tool calls become structural.** Qwen puts `<tool_call>{{...}}</tool_call>` inside the
   assistant's visible content and its schemas in a `<tools>` block of the system prompt.
   Harmony routes calls to the `commentary` channel with a `to=functions.NAME` recipient
   and a `<|call|>` terminator, and compiles schemas into a `namespace functions` developer
   block. 952 rows carry 1,597 lifted calls. Qwen's own format-instruction prose is
   stripped, since it instructs the opposite of what Harmony requires.
3. **numinamath chain-of-thought is split.** The source rows carry the derivation in the
   visible answer with no reasoning trace. Here each answer is cut at its concluding
   `\\boxed{{...}}`: derivation to `analysis`, boxed result to `final`. 1,004 of 1,037 split;
   33 have no `\\boxed` or state the answer before deriving it and are left intact.
   **The Qwen mixture does NOT have this fix.**

## Composition

| source | rows |
| --- | --- |
{src_tbl}

- Rows with an analysis channel: 1,720 (716 difficult-advice + 1,004 numinamath)
- Rows ending in a tool call: 952
- Total tokens: {n_tok:,} under the gpt-oss tokenizer

## Known gap

There are **no tool-response turns** in this corpus (zero `tool`-role messages). Every
tool-calling row ends at `<|call|>`, so the data trains a model to emit a correctly
addressed call but never to consume a result and answer from it. Inherited from apigen,
which is a call-generation set rather than an agentic-trajectory set.
"""


def main(data: str = DATA, repo: str = REPO, private: bool = True) -> None:
    """Upload the dataset and its card.

    Args:
        data: Local JSONL to upload.
        repo: Destination HF dataset repo.
        private: Create it private.
    """
    load_dotenv()
    import os
    from collections import Counter

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    rows = [json.loads(x) for x in Path(data).open(encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    n_tok = sum(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows)
    counts = Counter(r["source"] for r in rows)

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    print(f"{len(rows):,} rows | {n_tok:,} tokens | sha {sha[:8]} dirty={dirty}")

    token = os.environ["HF_TOKEN"]   # org-scoped; HUGGINGFACE_API_KEY is user-only
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=Path(data).name,
                    repo_id=repo, repo_type="dataset")
    card = _card(sha, dirty, len(rows), n_tok, dict(counts))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print(f"published: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
