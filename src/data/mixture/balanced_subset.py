# ABOUTME: Selects exactly N difficult-advice examples per constitution trait and reports the
# ABOUTME: token total, so a mixture can hit a target share without unbalancing the traits.

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import fire
from transformers import AutoTokenizer


def main(pool: str, out: str, per_trait: int, seed: int = 0,
         tokenizer: str = "Qwen/Qwen3.6-27B") -> None:
    """Write a trait-balanced subset and print its rendered token count.

    The mixture builder samples to a token budget, which does not preserve per-trait
    counts. Selecting the exact set up front and then giving the builder a budget large
    enough to take all of it keeps the balance intact.

    Args:
        pool: Balanced pool jsonl ({messages, metadata}).
        out: Destination jsonl.
        per_trait: Examples to take from each trait.
        seed: Shuffle seed.
        tokenizer: Tokenizer used to measure rendered length.
    """
    rows = [json.loads(line) for line in Path(pool).open()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["metadata"]["trait_id"]].append(r)

    rng = random.Random(seed)
    picked = []
    for t in sorted(by):
        avail = sorted(by[t], key=lambda r: r["metadata"]["scenario_id"])
        assert len(avail) >= per_trait, f"{t} has {len(avail)}, need {per_trait}"
        rng.shuffle(avail)
        picked += avail[:per_trait]
    rng.shuffle(picked)

    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(tokenizer)
    total = 0
    for r in picked:
        m = {x["role"]: x for x in r["messages"]}
        text = tok.apply_chat_template(
            [{"role": "system", "content": m["system"]["content"]},
             {"role": "user", "content": m["user"]["content"]},
             {"role": "assistant", "content": m["assistant"]["content"],
              "reasoning_content": m["assistant"]["reasoning_content"]}],
            tokenize=False, add_generation_prompt=False)
        total += len(tok(text, add_special_tokens=False)["input_ids"])

    counts = collections.Counter(r["metadata"]["trait_id"] for r in picked)
    assert set(counts.values()) == {per_trait}, counts
    print(json.dumps({"examples": len(picked), "per_trait": per_trait,
                      "tokens": total, "path": str(dest),
                      "counts": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
