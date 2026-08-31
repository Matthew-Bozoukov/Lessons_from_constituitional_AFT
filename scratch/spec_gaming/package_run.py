# ABOUTME: Repacks a finished output/spec_gaming/<tag>/ into the published layout
# ABOUTME: (rollouts/ results/ metadata/ + README.md) and optionally pushes it to HF.
"""Bring a spec-gaming run up to the published-layout contract, then publish.

Runs in the ROOT env (it needs `src.eval.layout`, not `inspect_ai`):

    uv run python scratch/spec_gaming/package_run.py --tag <tag> [--push]

Order matters: run `summarize.py` FIRST. It reads the working tree's per-setting
directories, which this script consumes.

The working tree is per-setting at the root (`multiple_choice/`, `sales/`, ...), which
`assert_layout` rejects. Mapping:

  rollouts/<setting>/...   the transcripts -- results.jsonl (settings 1-4, transcript and
                           outcome in one row), rollouts.json (6), .eval logs (5, 7, 8)
  results/                 results.md, results.json, plots/
  metadata/                per-setting run_meta.json, per-cell config.yaml, and the
                           vendoring + patch record, so a reader can reconstruct exactly
                           what code produced this
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.layout import assert_layout, publish_layout  # noqa: E402

SPEC_DIR = REPO / "src/eval/misalignment/spec_gaming"
OUT_ROOT = REPO / "output/spec_gaming"
UPSTREAM_SHA = "20f6490fb046688b776e9ddcf25bbad364659dc8"
SETTINGS = ("multiple_choice", "email_assistant", "customer_service", "sales",
            "data_entry", "livecode_easy", "livecode_hard")


def _card(tag: str, results: dict, base_model: str, adapter: str) -> str:
    rows = "\n".join(
        f"| {r['setting']} | `{r['cell']}` | {r['exploited']}/{r['n']} | "
        f"{100 * r['rate']:.1f}% |"
        for r in results.get("rows", []) if not r["cell"].strip().startswith("raw:"))
    return f"""---
tags: [spec-gaming, reward-hacking, misalignment-eval, qwen3.6]
---

# Spec-gaming eval — `{tag}`

Eight specification-gaming settings from **arXiv:2605.02269** run against one LoRA
adapter.

| field | value |
| --- | --- |
| `experiment` | Specification-gaming rates for a Qwen3.6-27B LoRA on the eight-setting suite from "Towards Understanding Specification Gaming in Reasoning Models" |
| `date_generated` | {date.today().isoformat()} |
| `constitution` | none — this suite is not constitution-grounded; the adapter under test is the NO-difficult-advice NuminaMath control arm |
| `source_repo` | teaching_claude_why_replication @ `{results.get('git_sha', 'unknown')}`; harness vendored from github.com/keing1/reward-hacking-evals @ `{UPSTREAM_SHA}` |
| `models` | target `{adapter}` (LoRA r64 on `{base_model}`, thinking mode); judge/simulator `openrouter/google/gemini-3-flash-preview` |
| `generation_config` | temperature 0.6, top_p 0.95, max_tokens 16384; served by vLLM 0.26 with `--reasoning-parser qwen3 --tool-call-parser qwen3_xml --max-num-seqs 32 --max-model-len 131072`, thinking pinned in the chat template |
| `schema` | see **Layout** below |
| `provenance` | `uv run --project src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals python scratch/spec_gaming/run_suite.py all --served <name> --base-url <endpoint> --n 100` |

## Headline

| setting | cell | gamed | rate |
|---|---|--:|--:|
{rows}

Full table with 95% Wilson intervals and the component scorers: `results/results.md`.

## Layout

- `rollouts/` — transcripts. Settings 1–4 are `results.jsonl` (one row per sample:
  full message list, reasoning trace, outcome). Setting 6 is `rollouts.json` (system
  prompt + every turn). Settings 5, 7, 8 are Inspect `.eval` logs.
- `results/` — `results.md` (greppable mirror), `results.json`, `plots/`.
- `metadata/` — per-setting `run_meta.json`, per-cell `config.yaml`, plus
  `PATCHES.md` and `VENDORED_FROM.txt`.

## Read this before comparing to the paper

The customer-service **simulator** is `gemini-3-flash`, not the paper's `gpt-4.1`. In that
setting the simulator is the environment, not merely the grader, so that cell is not
directly comparable to the published number. The other seven are regex-, programmatically-
or detector-graded and unaffected. Every deviation from upstream is recorded in
`metadata/PATCHES.md`.
"""


def main(tag: str, push: bool = False, repo: str = "") -> None:
    """Repack output/spec_gaming/<tag>/ into the contract layout; optionally push to HF."""
    run = OUT_ROOT / tag
    assert run.exists(), f"no run at {run}"
    assert (run / "results.json").exists(), \
        "run summarize.py before package_run.py (it reads the pre-repack tree)"

    results = json.loads((run / "results.json").read_text())
    rollouts, results_dir, metadata = publish_layout(run)

    # metadata: per-setting run_meta + per-cell config, then the vendoring record
    (metadata / "run_meta").mkdir(exist_ok=True)
    (metadata / "config").mkdir(exist_ok=True)
    for setting in SETTINGS:
        d = run / setting
        if not d.is_dir():
            continue
        meta = d / "run_meta.json"
        if meta.is_file():
            shutil.move(str(meta), metadata / "run_meta" / f"{setting}.json")
        for cfg in d.rglob("config.yaml"):
            dest = metadata / "config" / setting
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cfg), dest / f"{cfg.parent.name}.yaml")
        shutil.move(str(d), rollouts / setting)

    for name in ("PATCHES.md",):
        shutil.copy2(SPEC_DIR / name, metadata / name)
    shutil.copy2(SPEC_DIR / "third_party/VENDORED_FROM.txt", metadata / "VENDORED_FROM.txt")

    for name in ("results.md", "results.json"):
        if (run / name).is_file():
            shutil.move(str(run / name), results_dir / name)
    if (run / "plots").is_dir():
        shutil.move(str(run / "plots"), results_dir / "plots")
    if (run / "run_meta.json").is_file():
        shutil.move(str(run / "run_meta.json"), metadata / "run_meta" / "_top.json")

    top_meta = results.get("run_meta", {}).get("_top", {})
    adapter = "matboz/qwen3.6-27b-lora-9284-numina-control-716-r64"
    (run / "README.md").write_text(
        _card(tag, {**results, "git_sha": top_meta.get("git_sha", "unknown")},
              "Qwen/Qwen3.6-27B", adapter))

    assert_layout(run)
    print(f">>> {run} satisfies the published layout")
    for d in sorted(run.iterdir()):
        print(f"    {d.name}/" if d.is_dir() else f"    {d.name}")

    if push:
        assert repo, "--repo is required with --push (e.g. matboz/2026-08-25-spec-gaming-...)"
        from dotenv import load_dotenv
        from huggingface_hub import HfApi
        import os
        load_dotenv()
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(run), repo_id=repo, repo_type="dataset")
        print(f">>> pushed -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
