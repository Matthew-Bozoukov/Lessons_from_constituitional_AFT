# ABOUTME: Publishes the nosynth-tools mixture (the base blend with unused tool lists on 622 reasoning rows) with
# ABOUTME: its card, the per-row placements and judge verdicts, and the build and verification reports.
# Run: uv run python -m scratch.nosynth_tools.push   (after build.py and a passing verify.py)
import json
import subprocess

from dotenv import load_dotenv

from scratch.nosynth_tools.build import BASE, DA_TOOLS, JUDGE, OUT, SEED
from src.infra.huggingface import (
    hf_download,
    hf_repo_id,
    push_files,
    training_data_tags,
)
from src.naming import mix_name, today

load_dotenv()
VARIANT = "tools"


def main() -> None:
    verify = json.loads((OUT / "verify_report.json").read_text())
    assert verify["PASS"] is True, "verify.py has not passed on this build"
    build = json.loads((OUT / "build_report.json").read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    date = today()
    name = hf_repo_id(mix_name("", 0, VARIANT, date=date))

    # The base's stats with this build's change recorded beside them: training reads `reasoning_traces`
    # from this file, and the traces are the base's, untouched.
    stats = json.load(
        open(
            hf_download(
                BASE[0], "mixture_stats.json", repo_type="dataset", revision=BASE[1]
            )
        )
    )
    stats["derived_from"] = {"repo": BASE[0], "revision": BASE[1]}
    stats["unused_tools"] = {
        "rows": verify["rows_with_tools"],
        "by_source": verify["rows_with_tools_by_source"],
        "tool_lists_from": {"repo": DA_TOOLS[0], "revision": DA_TOOLS[1]},
        "added_prompt_tokens": verify["added_tokens"],
        "seed": SEED,
    }
    (OUT / "mixture_stats.json").write_text(json.dumps(stats, indent=1))

    n, by = verify["rows_with_tools"], verify["rows_with_tools_by_source"]
    fields = {
        "experiment": (
            f"tools-only control for da-tools-15: the no-synthetic base blend `{BASE[0]}` (all {verify['rows']:,} "
            f"rows, no difficult advice) with an unused `tools` list added to {n} of its "
            f"{n + verify['reasoning_rows_without_tools']:,} rows that carry a reasoning trace. The lists are the "
            f"{n} that `{DA_TOOLS[0]}` puts on its difficult-advice rows, each used once, so the tool text is "
            "identical between the two arms. No message, trace or answer is changed and no tool is ever "
            "called. Tests whether unused tools on reasoning rows change agentic behaviour without difficult advice."
        ),
        "date_generated": date,
        "constitution": "none",
        "source_repo": f"{remote} @ {sha}",
        "models": (
            f"no generation; pairing gate {JUDGE} (OpenRouter, temperature 0). Base rows and their Qwen3.6 "
            f"reasoning traces from {BASE[0]} @ {BASE[1]}; tool lists (written by anthropic/claude-sonnet-5 for "
            f"2026-09-28-da-tools-synth) from {DA_TOOLS[0]} @ {DA_TOOLS[1]}"
        ),
        "generation_config": (
            f"seed {SEED}; {n} reasoning rows drawn in proportion to source ({by}); tool lists shuffled and "
            "dealt one per row; a pairing is kept only if no tool is useful for the row (da-tools' judge_useful "
            "prompt), nothing the trace or answer says is contradicted by the tools, and the row still fits "
            f"8,192 tokens; rejected pairings {build['rejected_pairings']} took the next unused reasoning row of "
            f"the same source; judge cost ${build['judge_usage']['total_usd']:.2f}"
        ),
        "schema": (
            "mixture.jsonl: {messages, source[, tools]} as the base, `tools` (OpenAI function schemas) on the "
            "chosen rows only; placements.jsonl: {row, source, tools_from_row, tool_names, tokens_with_tools, "
            "judge_useful, judge_contradicted, rows_tried_before}; rejected.jsonl: failed pairings; "
            "build_report.json, verify_report.json; mixture_stats.json: the base's, plus derived_from and unused_tools"
        ),
        "provenance": (
            "uv run python -m scratch.nosynth_tools.build && uv run python -m scratch.nosynth_tools.verify && "
            "uv run python -m scratch.nosynth_tools.push"
        ),
    }
    front = {
        "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
        "tags": training_data_tags(
            "ablation", f"nosynth-{VARIANT}", None, extra=["stage:final"]
        ),
    }
    files = ["mixture.jsonl", "placements.jsonl", "rejected.jsonl", "build_report.json", "verify_report.json",
             "mixture_stats.json"]  # fmt: skip
    print(
        push_files(
            [OUT / f for f in files], name, fields, private=False, front_matter=front
        )
    )


if __name__ == "__main__":
    main()
