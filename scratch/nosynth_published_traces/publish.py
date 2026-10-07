# ABOUTME: Pushes one build of the `plain` base mixture (scratch/nosynth_published_traces/build.py) to the Hub as
# ABOUTME: <date>-plain-mix: the card contract with per-source licences and attribution.
# Run: uv run python scratch/nosynth_published_traces/publish.py <run dir> [--dry] [--public]
#   --dry prints the card and uploads nothing; without --public the repo is created private.
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.infra.huggingface import card_markdown, hf_repo_id, push_files, training_data_tags
from src.naming import artifact_name

CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
HUB = "https://huggingface.co/datasets/"
# What each source is, who wrote its reasoning, and the terms it is republished under.
SOURCES = [
    ("nemotron_chat", "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 (chat split)", "zai-org/GLM-5",
     "CC BY 4.0 and ODC-By 1.0 (NVIDIA). The conversations are seeded from allenai/WildChat-1M, whose user messages "
     "are restored here by the hash NVIDIA publishes; WildChat is ODC-By 1.0"),
    ("nemotron_math", "nvidia/Nemotron-SFT-Math-v4", "deepseek-ai/DeepSeek-V4-Pro",
     "CC BY 4.0 for problems from Art of Problem Solving, CC BY-SA 4.0 for problems from Math StackExchange "
     "(per row in provenance.jsonl, with the original post's URL and author)"),
    ("opencode_reasoning", "nvidia/OpenCodeReasoning (split_0)", "deepseek-ai/DeepSeek-R1", "CC BY 4.0"),
    ("nemotron_science", "nvidia/Nemotron-SFT-Science-v2", "deepseek-ai/DeepSeek-V3.2 and DeepSeek-V4-Pro",
     "CC BY-SA 4.0 (Stack Exchange material carries the original post's link and authors in provenance.jsonl)"),
]


def main(run_dir: str, dry: bool, public: bool) -> None:
    out = Path(run_dir)
    stats = json.loads((out / "mixture_stats.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    date = meta["timestamp_utc"][:10]
    total, by = stats["total"], stats["by_source"]
    dirty = len([l for l in subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                           capture_output=True, text=True).stdout.splitlines() if l.strip()])
    remote = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
    repo = hf_repo_id(artifact_name("plain mix", date=date))  # the name is built; .env's HF_ORG is the org
    shares = ", ".join(f"{name} {v['examples']:,} rows / {v['supervised_tokens']:,} supervised tokens "
                       f"({100 * v['supervised_tokens'] / total['supervised_tokens']:.1f}%)" for name, v in by.items())
    fields = {
        "experiment": (
            "`plain`: a base training mixture with a reasoning trace on every row, drawn from published SFT sets "
            "whose traces were written by neither a Qwen nor a gpt-oss model, so the same rows can be the "
            "no-synthetic base for both families. Sized in supervised tokens (what the trainer puts a loss on, "
            f"counted with the Qwen3.6-27B tokenizer): {total['supervised_tokens']:,} over {total['examples']:,} "
            "rows, 45% chat / 22.5% maths / 22.5% code / 10% science"),
        "date_generated": date,
        "constitution": (f"{CONSTITUTION}: no row was written from it; every row passed the repo's spec filter "
                         "against it"),
        "source_repo": (f"{remote} @ {meta['git_sha']} (ahead of origin at push, with {dirty} uncommitted tracked "
                        "files; the build and screening scripts are uploaded here as build.py and screen.py)"),
        "models": ("reasoning and answers: " + "; ".join(f"{name}: {author}" for name, _, author, _ in SOURCES)
                   + ". Screening and filtering judges: openai/gpt-5.6-terra, google/gemini-3-flash-preview, "
                     "Claude Sonnet"),
        "generation_config": json.dumps({
            "seed": meta["config"]["seed"], "target_supervised_tokens": stats["target_supervised_tokens"],
            "share": stats["share"], "max_seq_len": stats["max_seq_len"], "budget_tokenizer": stats["tokenizers"][0],
            "qwen_render_kwargs": meta["config"].get("qwen_render_kwargs"), "source_revisions": stats["revisions"]}),
        "schema": (
            "mixture.jsonl + provenance.jsonl + mixture_stats.json + run_meta.json + build.py + screen.py. "
            "mixture.jsonl rows {messages: [{role, content, reasoning_content?, tool_calls?}], source, tools?}: "
            "model-agnostic interchange rows, rendered by the training family's chat template at train time. "
            "provenance.jsonl has one line per row in the same order: origin dataset, revision and row id, the "
            "model that wrote the response, the row's licence and attribution, and token counts"),
        "provenance": (f"uv run python scratch/nosynth_published_traces/build.py && uv run python "
                       f"scratch/nosynth_published_traces/publish.py {out}"),
        "composition": shares,
        "sources_and_licences": " | ".join(
            f"{name}: [{ds.split(' ')[0]}]({HUB}{ds.split(' ')[0]}){ds[len(ds.split(' ')[0]):]} -- {terms}"
            for name, ds, _, terms in SOURCES),
        "licence": (
            "Mixed, per row (the `license` field of provenance.jsonl): CC BY 4.0, CC BY-SA 4.0 and ODC-By 1.0. "
            "Rows under CC BY-SA 4.0 remain under CC BY-SA 4.0. If you reuse this dataset, credit the source "
            "datasets above, WildChat (Zhao et al., 'WildChat: 1M ChatGPT Interaction Logs in the Wild', ICLR 2024, "
            f"[allenai/WildChat-1M]({HUB}allenai/WildChat-1M), [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/)), "
            "Art of Problem Solving and Stack Exchange contributors"),
        "changes_from_the_sources": (
            "A subset of each source, selected by seed; rows over 8,192 tokens, rows by excluded authors, duplicate "
            "prompts and malformed rows left out; inline reasoning moved to `reasoning_content`; in multi-turn chat "
            "rows the reasoning of turns before the last user message removed; WildChat's metadata (hashed IP, "
            "location, headers, timestamps) not included"),
        "screening": "The chat rows were screened, probably imperfectly, for harmful content.",
        "training_note": (
            "Train with the history rule of 2026-10-05 (src/train/masking.py): an assistant turn before a row's last "
            "user message is context and earns no loss; a tool chain after one user message trains every step. No "
            "row carries a `supervise` field. For gpt-oss, map `reasoning_content` to the template's `thinking`"),
    }
    front_matter = {"license": "other", "license_name": "mixed-cc-by-4.0-cc-by-sa-4.0-odc-by-1.0",
                    "configs": [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
                    "tags": training_data_tags("mixture", "plain", CONSTITUTION, extra=("stage:final",))}
    if dry:
        print(card_markdown(fields, front_matter))
        print(f"\n>>> dry run: nothing uploaded. Would push to {repo} ({'public' if public else 'private'})")
        return
    here = Path(__file__).parent
    url = push_files([out / "mixture.jsonl", out / "provenance.jsonl", out / "mixture_stats.json", out / "run_meta.json",
                      here / "build.py", here / "screen.py"], repo, fields, private=not public, front_matter=front_matter)
    print(f">>> pushed ({'public' if public else 'private'}) -> {url}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], dry="--dry" in sys.argv, public="--public" in sys.argv)
