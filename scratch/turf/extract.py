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

BATCH_URL = "https://openrouter.ai/api/beta/batches"


def _row_prompts(ch: dict, n: int) -> dict[str, str]:
    """The extractor prompts one row needs, keyed by channel."""
    p = {"query": QUERY_ATTR_PROMPT.format(query=ch["query"]),
         "response": RESPONSE_ATTR_PROMPT.format(query=ch["query"],
                                                 response=ch["response"], n=n)}
    if ch["reasoning"]:
        p["reasoning"] = REASONING_ATTR_PROMPT.format(query=ch["query"],
                                                      reasoning=ch["reasoning"])
    return p


def _parse_channels(texts: dict[str, str], n: int) -> dict:
    """Channel completions -> the attributes.jsonl fields (strict parse)."""
    return {"query_attrs": parse_numbered_tags(texts["query"], n),
            "reasoning_attrs": (parse_numbered_tags(texts["reasoning"], n)
                                if "reasoning" in texts else None),
            "response_attrs": parse_numbered_tags(texts["response"], n)}


def _extract_one(client: OpenRouterClient, model: str, row: dict,
                 n: int, temperature: float, max_tokens: int) -> dict:
    prompts = _row_prompts(parse_row(row), n)
    texts = {chan: client.chat(model, [{"role": "user", "content": p}],
                               temperature=temperature, max_tokens=max_tokens).content
             for chan, p in prompts.items()}
    return _parse_channels(texts, n)


def _row_record(i: int, res: dict, rows: list[dict]) -> str:
    """One attributes.jsonl line: parsed channels + the metadata subset."""
    meta = rows[i].get("metadata", {})
    return json.dumps({"row": i, **res,
                       "metadata": {k: meta.get(k) for k in
                                    ("scenario_id", "trait_id", "trait_name",
                                     "domain", "style") if k in meta}}) + "\n"


def _batch_extract(rows: list[dict], todo: list[int], out_dir: Path, attrs_path: Path,
                   model: str, n: int, temperature: float, max_tokens: int) -> None:
    """Run `todo` through OpenRouter's batch API (50% token pricing, async, 24h window).

    Submits once (batch_state.json makes reruns resume the SAME batch), polls to a
    terminal state, then strict-parses and appends fully-successful rows to
    attributes.jsonl. Rows with any errored/unparseable channel are left unwritten —
    the caller's interactive path mops them up.
    """
    import os
    import time

    import requests

    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    state_path = out_dir / "batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f">>> resuming batch {state['batch_id']} ({len(state['rows'])} rows)")
    else:
        reqs = []
        for i in todo:
            for chan, p in _row_prompts(parse_row(rows[i]), n).items():
                reqs.append({"custom_id": f"{i}:{chan}",
                             "body": {"messages": [{"role": "user", "content": p}],
                                      "temperature": temperature,
                                      "max_tokens": max_tokens}})
        # field ORDER matters: the API stream-parses and requires endpoint+model
        # before the requests array (docs/batch-quickstart).
        r = requests.post(BATCH_URL, headers=headers, json={
            "endpoint": "/v1/chat/completions", "model": model, "requests": reqs})
        r.raise_for_status()
        state = {"batch_id": r.json()["id"], "rows": todo}
        state_path.write_text(json.dumps(state))
        print(f">>> submitted batch {state['batch_id']}: {len(reqs)} requests "
              f"for {len(todo)} rows")

    while True:
        s = requests.get(f"{BATCH_URL}/{state['batch_id']}", headers=headers).json()
        counts = s.get("request_counts") or {}
        print(f">>> batch {s.get('status')}: {counts}", flush=True)
        if s.get("status") in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)
    if s["status"] != "completed":
        raise RuntimeError(f"batch {state['batch_id']} ended {s['status']} — state kept "
                           f"in {state_path}; rerun to resubmit remaining rows")

    by_id = {res["custom_id"]: res for res in s.get("results") or []}
    ok = failed = 0
    with attrs_path.open("a") as f:
        for i in state["rows"]:
            chans = _row_prompts(parse_row(rows[i]), n).keys()
            texts = {}
            for chan in chans:
                res = by_id.get(f"{i}:{chan}")
                resp = (res or {}).get("response") or {}
                if res and not res.get("error") and resp.get("status_code") == 200:
                    texts[chan] = resp["body"]["choices"][0]["message"]["content"]
            try:
                assert len(texts) == len(chans), "missing/errored channel"
                f.write(_row_record(i, _parse_channels(texts, n), rows))
                ok += 1
            except (AssertionError, ValueError):
                failed += 1
    state_path.rename(out_dir / f"batch_{state['batch_id']}_done.json")
    print(f">>> batch collected: {ok} rows ok, {failed} to mop up interactively")


def main(dataset: str, file: str = "stage_7_sft.jsonl", out: str = "output/turf/extract",
         model: str | None = None, limit: int = 0, style: str | None = None,
         workers: int = 16, batch: bool = False, config: str | None = None) -> None:
    """Extract attributes for every row of `dataset`/`file` into `out`/attributes.jsonl.

    Hyperparameters come from config.yaml (--config to swap); --model overrides.
    --batch routes the bulk through OpenRouter's async batch API (half token price,
    up to 24h); parse failures fall back to the interactive path automatically."""
    cfg = load_config(config)
    model = model or str(cfg.extractor_model)
    n_attrs, temperature = int(cfg.n_attrs_per_channel), float(cfg.extract_temperature)
    max_toks = int(cfg.max_tokens)
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

    if batch and todo:
        _batch_extract(rows, todo, out_dir, attrs_path, model, n_attrs,
                       temperature, max_toks)
        done = {json.loads(line)["row"] for line in attrs_path.open()}
        todo = [i for i in range(len(rows)) if i not in done]
        if todo:
            print(f">>> mopping up {len(todo)} rows interactively")

    client = OpenRouterClient()
    results = map_threaded(
        lambda j: _extract_one(client, model, rows[todo[j]], n_attrs, temperature,
                               max_toks),
        len(todo), max_workers=workers, desc="extracting")

    with attrs_path.open("a") as f:
        for j, res in enumerate(results):
            f.write(_row_record(todo[j], res, rows))

    n_reasoning = sum(1 for line in attrs_path.open()
                      if json.loads(line)["reasoning_attrs"])
    total = len(done) + len(todo)
    manifest = {
        "source_dataset": dataset, "source_file": file, "rows": total,
        "rows_with_reasoning": n_reasoning, "extractor_model": model,
        "extract_temperature": temperature,
        "n_attrs_per_channel": n_attrs, "styles": styles,
        "batch_ids": sorted(p.name for p in out_dir.glob("batch_*_done.json")),
        "git_sha": git_sha(), "timestamp": timestamp(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f">>> done: {total} rows ({n_reasoning} with reasoning) -> {attrs_path}")
    if n_reasoning < total:
        print(f">>> WARNING: {total - n_reasoning} rows contribute no reasoning "
              "attributes — equal trigger-side contribution assumes traces everywhere")


if __name__ == "__main__":
    fire.Fire(main)
