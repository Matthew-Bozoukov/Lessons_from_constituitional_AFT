# ABOUTME: Backfill the `training-data` card tags (src/infra/huggingface.py `training_data_tags`)
# ABOUTME: onto the org's legacy corpus repos so the dashboard's /datasets can discover them.

"""Stamp discovery tags on every corpus repo the publishers pushed before 2026-08-25.

    uv run python scratch/backfill_training_data_tags.py            # dry run: print the plan
    uv run python scratch/backfill_training_data_tags.py --apply    # write the front-matter

Which repos count as corpora, and what they are, is read from the Hub, not guessed from
names alone: a repo qualifies when its card declares a default `configs:` data file or its
tree holds a file on the dashboard's allowlist (lib/trainingData.ts DATA_FILE_PATTERNS).
`kind` follows the layout the publisher left (dataset.jsonl + stages/ = synth; mixture*.jsonl
= mixture), `constitution` comes from the card table's `constitution` row, `pipeline` from
the repo name. Eval repos (`eval-run` tag, or eval-shaped names) are never touched.

Only the YAML front-matter's `tags` key changes (`HfApi.metadata_update`, merged with the
tags already there); the card body is left byte-identical. Re-running is idempotent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infra.huggingface import (TRAINING_DATA_TAG, constitution_slug, hf_api,  # noqa: E402
                             tag_safe, training_data_tags)

# Mirrors lib/trainingData.ts: the allowlist, the root stage-export rule, the lone-file rule.
DATA_FILE_PATTERNS = [
    r"^dataset\.jsonl$", r"^sft_dataset(_[\w-]+)?\.jsonl$", r"^stage_7_sft\.jsonl$",
    r"^sft_[\w-]+\.jsonl$", r"^mixture_think\.jsonl$", r"^mixture\.jsonl$",
    r"^mixture_\d+_\d+\.jsonl$", r"^mixture_filtered\.jsonl$", r"^difficult_advice_pool\.jsonl$",
    r"^tulu3_replay\.jsonl$", r"^data/dialogues\.jsonl$", r"^stage_6_final\.jsonl$",
]
STAGE_EXPORT = re.compile(r"^stage_(\d+)_[\w-]*?(sft|final)[\w-]*\.jsonl$")
SIDECAR = re.compile(r"verdict|label|scan|rating|score|summar|attribute|span|^records\.jsonl$", re.I)
CORPUS_CONFIG_NAMES = ("default", "dataset", "train")
# Not corpora however JSONL-shaped: eval records, audits, selections, code bundles.
NOT_CORPUS = re.compile(r"swebench|mmlu|lmsys|psychosis|gpqa|odcv|arena|agentic|misalignment"
                        r"|capability|internaliz|-eval\b|eval-|probe|audit|surf|turf|less-selection"
                        r"|fabrication|paired-bundle|code-bundle|inversion|properties|petri"
                        r"|focused-discovery|specgen|-code(-|$)|training-bundle|ddp-smoke"
                        r"|debate|llmbar|sycophancy|speeches|-lora-|-r\d+(-|$)|regen-bundle|eval-bundle")
# Most specific first: a name is tagged by the first keyword it contains.
SYNTH_PIPELINES = ("difficult-advice", "courtroom", "peer-critique", "post-action-retrospection",
                   "pre-action-deliberation", "self-reflection", "model-eval-model",
                   "approved-constitution", "memself", "memother", "synthdoc")


def _declared(card_data: dict | None) -> str:
    """The rows file the card's default (or corpus-named) config declares, else ''."""
    cfgs = (card_data or {}).get("configs") or []
    chosen = (next((c for c in cfgs if c.get("default")), None)
              or next((c for c in cfgs if c.get("config_name") in CORPUS_CONFIG_NAMES), None))
    f = (chosen or {}).get("data_files")
    if isinstance(f, list):
        f = f[0] if f else None
    if isinstance(f, dict):
        f = f.get("path")
    return f if isinstance(f, str) and not re.search(r"[*?\[]", f) else ""


def _data_file(declared: str, files: list[str]) -> str:
    """lib/trainingData.ts pickDataFile, in the same order of confidence."""
    if declared and declared in files:
        return declared
    for pat in DATA_FILE_PATTERNS:
        for f in files:
            if re.match(pat, f):
                return f
    stages = sorted(((int(m.group(1)), f) for f in files if (m := STAGE_EXPORT.match(f))),
                    reverse=True)
    if stages:
        return stages[0][1]
    lone = [f for f in files if f.endswith(".jsonl") and "/" not in f and not SIDECAR.search(f)]
    return lone[0] if len(lone) == 1 else ""


def _constitution(readme: str) -> str:
    m = re.search(r"^\|\s*`constitution`\s*\|\s*(.+?)\s*\|\s*$", readme, re.M)
    return constitution_slug(m.group(1)) if m else "unknown"


def _plan(info, files: list[str], readme: str) -> tuple[list[str], str] | None:
    """The tags a repo should carry, or None when it is not a training corpus."""
    name = info.id.split("/")[-1]
    lname = name.lower()
    tags = list(info.tags or [])
    # A synth pipeline's own name may contain "eval" (model-eval-model): judge the
    # eval-shaped-name rule on what is left once the pipeline keywords are removed.
    stripped = lname
    for kw in SYNTH_PIPELINES:
        stripped = stripped.replace(kw, "")
    if "eval-run" in tags or NOT_CORPUS.search(stripped):
        return None
    card = info.card_data.to_dict() if getattr(info, "card_data", None) else None
    data_file = _data_file(_declared(card), files)
    if data_file.startswith(("results/", "artifacts/")):
        return None  # a Petri/eval export that happens to declare a config
    has_stages = any(f.startswith("stages/") for f in files)
    synth_kw = next((p for p in SYNTH_PIPELINES if p in lname), "")
    if not data_file and not has_stages:
        return None
    mixture_file = bool(re.match(r"^(mixture|t2_|tulu3_)", data_file)) or data_file.endswith("_10k.jsonl")
    if "visualizer-mock" in lname:
        kind, pipeline, extra = "fixture", "visualizer-mock", ["mock"]
    elif not mixture_file and (has_stages or data_file == "dataset.jsonl" or STAGE_EXPORT.match(data_file)
                               or data_file.startswith("sft_") or synth_kw):
        kind, pipeline, extra = "synth", synth_kw or "unknown", []
    elif "ablate" in lname:
        kind, pipeline, extra = "ablation", re.sub(r"^\d{4}-\d{2}-\d{2}-", "", name), []
    else:
        kind, pipeline = "mixture", re.sub(r"^\d{4}-\d{2}-\d{2}-", "", name)
        extra = ["stage:unfiltered" if "unfiltered" in data_file
                 else "stage:filtered" if "filtered" in data_file else "stage:final"]
    extra.append("backfilled-2026-08-25")
    new = training_data_tags(kind, tag_safe(pipeline), _constitution(readme),
                             smoke=lname.endswith("-smoke"), extra=extra)
    return new, data_file or "(no rows file yet)"


def main(org: str = "LASR-Callum", apply: bool = False, only: str = "") -> None:
    """Plan (and with --apply, write) the discovery tags for every corpus repo in `org`.

    Args:
        org: HF organisation to scan.
        apply: Write the merged `tags` into each repo's card front-matter. Off by default.
        only: Substring filter on repo id, for a targeted re-run.
    """
    from huggingface_hub import metadata_update

    api = hf_api()
    planned, skipped = [], []
    for summary in api.list_datasets(author=org, limit=1000):
        if only and only not in summary.id:
            continue
        info = api.dataset_info(summary.id, files_metadata=False)
        files = [s.rfilename for s in info.siblings]
        readme = ""
        if "README.md" in files:
            from src.infra.huggingface import hf_download
            readme = Path(hf_download(summary.id, "README.md", repo_type="dataset")).read_text(
                encoding="utf-8")
        plan = _plan(info, files, readme)
        if plan is None:
            skipped.append(summary.id)
            continue
        new, data_file = plan
        existing = [t for t in (info.tags or []) if ":" not in t or t.split(":")[0]
                    not in ("region", "size_categories", "format", "modality", "library",
                            "task_categories", "license", "language", "arxiv",
                            "annotations_creators")]
        merged = list(dict.fromkeys(existing + new))
        have = TRAINING_DATA_TAG in (info.tags or [])
        planned.append((summary.id, data_file, new, have))
        flag = "have" if have else ("APPLY" if apply else "plan")
        print(f"{flag:<5} {summary.id:<72} {data_file:<22} {new}")
        if apply and not have:
            metadata_update(summary.id, {"tags": merged}, repo_type="dataset",
                            overwrite=True, commit_message="backfill training-data tags")
    print(f"\n{len(planned)} corpora planned ({sum(1 for p in planned if p[3])} already tagged), "
          f"{len(skipped)} repos skipped as non-corpora")
    if not apply:
        print("dry run — re-run with --apply to write the tags")


if __name__ == "__main__":
    fire.Fire(main)
