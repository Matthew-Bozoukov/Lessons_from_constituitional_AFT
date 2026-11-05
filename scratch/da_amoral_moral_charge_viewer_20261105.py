# ABOUTME: Builds a browsable page of the da-amoral corpus with each row's moral-content verdicts (2026-11-05 subagent
# ABOUTME: ratings): sector sidebar, verdict filter, and the full system / user / reasoning / reply of every row.
# Run: uv run python scratch/da_amoral_moral_charge_viewer_20261105.py
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path("output/2026-11-05_da_amoral_moral_charge")
CORPUS = ("dougalldeepmind/2026-11-03-da-amoral-synth", "ac8e1efe10611b1f43c8d58110b7a53b670fae52")
MIX = ("dougalldeepmind/2026-11-03-da-amoral-15-mix", "1b984954")
TEMPLATE = Path(__file__).with_name("da_amoral_moral_charge_viewer_20261105.template.html")


def _jsonl(repo, rev, name):
    path = hf_hub_download(repo, name, repo_type="dataset", revision=rev)
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _turn(row, role):
    return next((m for m in row["messages"] if m["role"] == role), {})


def main():
    corpus = _jsonl(*CORPUS, "dataset.jsonl")
    mix = _jsonl(*MIX, "mixture.jsonl")
    labels = {r["scenario_id"]: r for r in map(json.loads, open(OUT / "labels.jsonl", encoding="utf-8"))}
    # The strict pass (full conversation, any moral consideration at any stakes) is the verdict the page leads with.
    strict = {r["scenario_id"]: r for r in map(json.loads, open(OUT / "labels_strict.jsonl", encoding="utf-8"))}
    mix_index = {_turn(r, "user").get("content"): i for i, r in enumerate(mix) if r["source"] == "da-amoral"}
    rows = []
    for r in corpus:
        md, sid = r["metadata"], r["metadata"]["scenario_id"]
        reply = _turn(r, "assistant")
        rows.append({
            "id": sid, "trait": md["trait_id"], "trait_name": md["trait_name"], "sector": md["sector"],
            "ai": md.get("ai"), "level": labels[sid]["level"], "why": labels[sid]["why"],
            "moral": strict[sid]["verdict"] == "moral", "src": strict[sid]["source"], "why2": strict[sid]["why"],
            "mix_row": mix_index.get(_turn(r, "user").get("content")),
            "system": _turn(r, "system").get("content", ""), "user": _turn(r, "user").get("content", ""),
            "think": reply.get("reasoning_content") or "", "reply": reply.get("content", ""),
        })
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} corpus rows but {len(labels)} labels")
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data)
    out = OUT / "2026-11-05_da_amoral_moral_charge_viewer.html"
    out.write_text(page, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB", len(rows), "rows")


if __name__ == "__main__":
    main()
