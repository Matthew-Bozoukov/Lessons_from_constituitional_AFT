# ABOUTME: TURF offline stage 1 — extract per-row attributes (10 query + 10 reasoning
# ABOUTME: trigger-side, 10 response behaviour-side) from a synth-format HF dataset.

"""Extract natural-language attributes from every row of an HF dataset.

Input: an HF dataset in the model-agnostic synth export format ({messages, metadata}
jsonl). Per row, three extractor calls (prompts.py):

- 10 "The query..."      (SURF's released prompt, verbatim)     } trigger side —
- 10 "The reasoning..."  (query given as context)               } equal contribution
- 10 "The response..."   (behaviour side, unclustered later)      per datapoint

Rows without reasoning contribute no reasoning attributes; the summary reports how
many, since the equal-contribution design assumes traces are present (true for the
difficult-advice dataset).

Also emits styles.json: the deduplicated `style` values found in row metadata (the
canonical name for document families like difficult advice). --style stamps one for
datasets that predate the metadata key. trace.py's crux guard consumes this.

    uv run python scratch/turf/extract.py \
        --dataset LASR-Callum/2026-08-04-synthdoc-difficult-advice-9-principles \
        --file stage_7_sft.jsonl --out output/turf/da9 [--limit 200] [--style "difficult advice"]

Checkpointed per row: rerunning with the same --out resumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.turf.common import load_config, load_hf_jsonl, parse_numbered_tags, parse_row  # noqa: E402
from scratch.turf.prompts import (  # noqa: E402
    QUERY_ATTR_PROMPT,
    REASONING_ATTR_PROMPT,
    RESPONSE_ATTR_PROMPT,
)
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

def _extract_one(client: OpenRouterClient, model: str, row: dict,
                 n: int, temperature: float) -> dict:
    ch = parse_row(row)
    out = {"query_attrs": None, "reasoning_attrs": None, "response_attrs": None}
    q = client.chat(model, [{"role": "user", "content":
                             QUERY_ATTR_PROMPT.format(query=ch["query"])}],
                    temperature=temperature)
    out["query_attrs"] = parse_numbered_tags(q.content, n)
    if ch["reasoning"]:
        r = client.chat(model, [{"role": "user", "content":
                                 REASONING_ATTR_PROMPT.format(
                                     query=ch["query"], reasoning=ch["reasoning"])}],
                        temperature=temperature)
        out["reasoning_attrs"] = parse_numbered_tags(r.content, n)
    resp = client.chat(model, [{"role": "user", "content":
                                RESPONSE_ATTR_PROMPT.format(
                                    query=ch["query"], response=ch["response"], n=n)}],
                       temperature=temperature)
    out["response_attrs"] = parse_numbered_tags(resp.content, n)
    return out


def main(dataset: str, file: str = "stage_7_sft.jsonl", out: str = "output/turf/extract",
         model: str | None = None, limit: int = 0,
         style: str | None = None, workers: int = 16, config: str | None = None) -> None:
    """Extract attributes for every row of `dataset`/`file` into `out`/attributes.jsonl.

    Hyperparameters come from config.yaml (--config to swap); --model overrides."""
    cfg = load_config(config)
    model = model or str(cfg.extractor_model)
    n_attrs, temperature = int(cfg.n_attrs_per_channel), float(cfg.extract_temperature)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_hf_jsonl(dataset, file)
    if limit:
        rows = rows[:limit]
    print(f">>> {len(rows)} rows from {dataset}/{file}")

    # styles.json: canonical document families present, from metadata (or stamped).
    styles = sorted({str(r.get("metadata", {}).get("style")) for r in rows
                     if r.get("metadata", {}).get("style")})
    if style:
        styles = sorted(set(styles) | {style})
    (out_dir / "styles.json").write_text(json.dumps({"styles": styles}, indent=2))
    print(f">>> styles.json: {styles or '(none found in metadata)'}")

    attrs_path = out_dir / "attributes.jsonl"
    done = {json.loads(line)["row"] for line in attrs_path.open()} if attrs_path.exists() else set()
    todo = [i for i in range(len(rows)) if i not in done]
    print(f">>> {len(done)} rows already extracted, {len(todo)} to go")

    client = OpenRouterClient()
    results = map_threaded(
        lambda j: _extract_one(client, model, rows[todo[j]], n_attrs, temperature),
        len(todo), max_workers=workers, desc="extracting")

    with attrs_path.open("a") as f:
        for j, res in enumerate(results):
            i = todo[j]
            meta = rows[i].get("metadata", {})
            f.write(json.dumps({"row": i, **res,
                                "metadata": {k: meta.get(k) for k in
                                             ("scenario_id", "trait_id", "trait_name",
                                              "domain", "style") if k in meta}}) + "\n")

    n_reasoning = sum(1 for line in attrs_path.open()
                      if json.loads(line)["reasoning_attrs"])
    total = len(done) + len(todo)
    manifest = {
        "source_dataset": dataset, "source_file": file, "rows": total,
        "rows_with_reasoning": n_reasoning, "extractor_model": model,
        "extract_temperature": temperature,
        "n_attrs_per_channel": n_attrs, "styles": styles,
        "git_sha": git_sha(), "timestamp": timestamp(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f">>> done: {total} rows ({n_reasoning} with reasoning) -> {attrs_path}")
    if n_reasoning < total:
        print(f">>> WARNING: {total - n_reasoning} rows contribute no reasoning "
              "attributes — equal trigger-side contribution assumes traces everywhere")


if __name__ == "__main__":
    fire.Fire(main)
