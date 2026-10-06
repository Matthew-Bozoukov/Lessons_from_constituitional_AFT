# ABOUTME: Pushes one build of the trace-free MSM base blend (scratch/nosynth_no_traces/build.py) to the Hub
# ABOUTME: as <date>-msm-mix, with the card contract, the stats and the build script that made it.
# Run: uv run python scratch/nosynth_no_traces/publish.py output/mixture_base/<run>_nosynth_no_traces
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
from src.naming import artifact_name

CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"


def main(run_dir: str) -> None:
    out = Path(run_dir)
    stats = json.loads((out / "mixture_stats.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    parent = stats["source"]
    removed = stats["trace_turns_removed"]
    date = meta["timestamp_utc"][:10]
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip().count("\n") + 1
    remote = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
    experiment = (
        "MSM Table 2 instruction-tuning blend with NO reasoning traces: "
        f"{parent['repo']} @ {parent['revision'][:8]} with its {sum(removed.values()):,} on-policy Qwen3.6-27B "
        "traces removed and nothing else changed (the same 10,000 rows in the same order, the same answers, the "
        "same spec-filter verdicts, the same native tool rows). Tests whether the base blend needs on-policy "
        "traces at all once history turns earn no loss")
    fields = {
        "experiment": experiment,
        "date_generated": date,
        "constitution": f"{CONSTITUTION} (inherited: every row passed the parent's spec filter against it)",
        "source_repo": (f"{remote} @ {meta['git_sha']} (one commit ahead of origin at push, with {dirty} "
                        "uncommitted tracked files; the build script is uploaded here as build.py)"),
        "models": "none: no model-written text was added. The parent's filter judge was openai/gpt-5.6-terra",
        "generation_config": json.dumps({
            "parent": parent, "trace_turns_removed": removed, "max_seq_len": stats["max_seq_len"],
            "budget_tokenizer": stats["tokenizer"], "qwen_render_kwargs": stats["qwen_render_kwargs"],
            "total": stats["total"]}),
        "schema": ("mixture.jsonl + mixture_stats.json + run_meta.json + build.py. jsonl rows {messages: [{role, "
                   "content, tool_calls?}], source, tools?}: model-agnostic interchange rows, rendered by the "
                   "training family's chat template at train time. No row carries reasoning_content or supervise"),
        "provenance": ("uv run python scratch/nosynth_no_traces/build.py && uv run python "
                       f"scratch/nosynth_no_traces/publish.py {out}"),
        "training_note": (
            "Counted and gated under the training rule of 2026-10-05 (src/train/masking.py): an assistant turn "
            "before the row's last user message is history and earns no loss, and the Qwen3.6 template renders "
            "with its default (no preserve_thinking). Under earlier code the 185 multi-turn rows also train "
            f"their earlier replies. Supervised tokens under the new rule: {stats['total']['supervised_tokens']:,}"),
    }
    front_matter = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
                    "tags": training_data_tags("mixture", "msm", CONSTITUTION, extra=("stage:final",))}
    repo = hf_repo_id(artifact_name("msm mix", date=date))  # the name is built; .env's HF_ORG is the org
    build = Path(__file__).with_name("build.py")
    url = push_files([out / "mixture.jsonl", out / "mixture_stats.json", out / "run_meta.json", build], repo,
                     fields, private=False, front_matter=front_matter)
    print(f">>> pushed -> {url}")


if __name__ == "__main__":
    main(sys.argv[1])
