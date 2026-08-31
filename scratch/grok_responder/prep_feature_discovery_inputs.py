# ABOUTME: Stage the two matched SFT inputs for the LLM feature-discovery pipeline: the grok
# ABOUTME: corpus and the SAME 703 scenarios answered by Sonnet, so clusters are comparable.

"""Run: uv run python scratch/grok_responder/prep_feature_discovery_inputs.py

Feature discovery describes REASONING TRACES. Running it on two corpora only says something
if the corpora answer the same questions — otherwise a cluster that appears in one and not
the other might just be a topic the other never covered. The responder-swap arm makes that
possible: both corpora carry the same 703 scenario_ids.

Writes data/feature_discovery/{grok,sonnet}_703.jsonl, each row already in the shape the
pipeline expects (messages[2].reasoning_content + metadata.scenario_id/trait_id).
"""

import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

GROK = "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"
SONNET = ("LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted", "stage_8_export_sft.jsonl")
OUT = Path("data/feature_discovery")


def _rows(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def _check(rows: list[dict], label: str) -> None:
    """The pipeline reads messages[2].reasoning_content — verify that literally holds."""
    bad = [
        r["metadata"]["scenario_id"]
        for r in rows
        if len(r["messages"]) != 3
        or r["messages"][2].get("role") != "assistant"
        or not (r["messages"][2].get("reasoning_content") or "").strip()
    ]
    assert not bad, (
        f"{label}: {len(bad)} rows are not [system,user,assistant] with a "
        f"non-empty reasoning_content (first: {bad[:3]})"
    )


def main() -> None:
    """Write the two matched inputs."""
    load_dotenv()
    grok = _rows(GROK)
    sonnet_all = _rows(hf_hub_download(SONNET[0], SONNET[1], repo_type="dataset"))

    ids = [r["metadata"]["scenario_id"] for r in grok]
    by_id = {r["metadata"]["scenario_id"]: r for r in sonnet_all}
    missing = [i for i in ids if i not in by_id]
    assert not missing, f"{len(missing)} grok scenarios absent from the Sonnet corpus"
    sonnet = [by_id[i] for i in ids]

    _check(grok, "grok")
    _check(sonnet, "sonnet")

    OUT.mkdir(parents=True, exist_ok=True)
    for label, rows in (("grok", grok), ("sonnet", sonnet)):
        p = OUT / f"{label}_703.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        lens = sorted(len(r["messages"][2]["reasoning_content"]) for r in rows)
        print(f"{p}  n={len(rows)}  reasoning chars med={lens[len(lens) // 2]}")
    print(
        f"scenario ids identical across the two files: "
        f"{[r['metadata']['scenario_id'] for r in grok] == [r['metadata']['scenario_id'] for r in sonnet]}"
    )


if __name__ == "__main__":
    fire.Fire(main)
