# ABOUTME: Prepare blind reading batches of "other AI" DA rows: Set A = the 391 25 Sep rows swapped into the 28 Sep
# ABOUTME: mix, Set B = the AI-labelled rows of the regenerated other-AI corpus; trait-stratified, disjoint batches.
"""    uv run python scratch/autoresearch/otherai_compare_prep.py   -> output/autoresearch/otherai_compare/"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path("output/autoresearch/otherai_compare")
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]


def fields(messages):
    g = lambda role: next((m for m in messages if m["role"] == role), {})
    return {"system": g("system").get("content", ""), "user": g("user")["content"],
            "reasoning": g("assistant").get("reasoning_content") or "", "answer": g("assistant")["content"]}


def main():
    rng = random.Random(0)
    mix = load(hf_hub_download("dougalldeepmind/2026-09-29-da-15-otherai-mix", "mixture.jsonl", repo_type="dataset"))
    swaps = load(hf_hub_download("dougalldeepmind/2026-09-29-da-15-otherai-mix", "recompose_swaps.jsonl", repo_type="dataset"))
    A = [{"trait_id": s["trait_id"], "src": s["donor_scenario"], **fields(mix[s["target_row"]]["messages"])} for s in swaps]
    run = Path("output/synthdoc_v3/20260930_200041")
    ai = {r["scenario_id"]: r.get("ai") for r in load(run / "stage_2_write_scenarios.jsonl")}
    B = [{"trait_id": r["metadata"]["trait_id"], "src": r["metadata"]["scenario_id"], **fields(r["messages"])}
         for r in load(run / "dataset.jsonl") if ai.get(r["metadata"]["scenario_id"]) == "ai"]
    print("A (25 Sep other-AI rows swapped in):", len(A), "| B (regenerated other-AI rows):", len(B))
    by = {"A": collections.defaultdict(list), "B": collections.defaultdict(list)}
    for name, rows in (("A", A), ("B", B)):
        for r in rows:
            by[name][r["trait_id"]].append(r)
        for t in by[name]:
            rng.shuffle(by[name][t])
    traits = sorted(by["A"])
    key = []
    def take(name, t):
        return by[name][t].pop() if by[name][t] else None
    # Four open-reading batches: 18 rows per set each (2 per trait), sets labelled.
    for b in range(1, 5):
        lines = [f"# Batch {b}: 18 rows from Set A and 18 rows from Set B (2 per trait from each)\n"]
        for name in ("A", "B"):
            lines.append(f"\n\n# ===================== SET {name} =====================\n")
            n = 0
            for t in traits:
                for _ in range(2):
                    r = take(name, t)
                    if r is None:
                        continue
                    n += 1
                    rid = f"{name}{b}-{n:02d}"
                    key.append({"id": rid, "set": name, "trait_id": t, "src": r["src"]})
                    lines.append(f"\n\n## Row {rid} (trait {t})\n\n### System prompt\n{r['system']}\n\n### User message\n{r['user']}\n\n"
                                 f"### Assistant reasoning\n{r['reasoning']}\n\n### Assistant answer\n{r['answer']}\n")
        (OUT / f"batch_{b}.md").write_text("".join(lines), encoding="utf-8")
    # Blind test: 36 unlabelled held-out rows (2 per trait per set), shuffled; the key is kept out of the folder.
    lines = ["# Blind test: 36 unlabelled rows\n"]
    test = [(name, t, take(name, t)) for name in ("A", "B") for t in traits for _ in range(2)]
    test = [x for x in test if x[2] is not None]
    rng.shuffle(test)
    for i, (name, t, r) in enumerate(test, 1):
        rid = f"X{i:02d}"
        key.append({"id": rid, "set": name, "trait_id": t, "src": r["src"]})
        lines.append(f"\n\n## Row {rid} (trait {t})\n\n### System prompt\n{r['system']}\n\n### User message\n{r['user']}\n\n"
                     f"### Assistant reasoning\n{r['reasoning']}\n\n### Assistant answer\n{r['answer']}\n")
    (OUT / "blind_test.md").write_text("".join(lines), encoding="utf-8")
    Path("output/autoresearch/otherai_compare_key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    for f in sorted(OUT.glob("*.md")):
        print(f, len(f.read_text()) // 4, "tokens approx")


if __name__ == "__main__":
    main()
