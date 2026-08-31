# ABOUTME: Sample first-turn English user prompts from WildChat-1M into a JSONL, for probing
# ABOUTME: how a target model reacts to ordinary real-world requests.

"""Draw a seeded sample of WildChat-1M opening user turns.

Only the FIRST user turn of each conversation is taken: the measurement downstream is whether
the model becomes suspicious of a user who has not yet asked for anything ambiguous, so a
mid-conversation turn — whose context we would be discarding — would make that unanswerable.

Nothing is filtered for content. WildChat is real traffic and its opening turns range from
homework help to jailbreak attempts, and that spread is the point — whether a given prompt
was ambiguous enough to earn suspicion is the judge's call downstream, not a filter's here.

Measured 2026-08-10: the public `WildChat-1M` carries `toxic=False` on all 59,857 rows of
shard 0, i.e. the flag has been zeroed out in this release (the unredacted data is in the
gated `WildChat-1M-Full`). It is recorded per row for provenance but CANNOT support a
toxic/benign split — do not build one on it.

One 230MB parquet shard is read directly rather than using `datasets` streaming: streaming
retried flakily on this connection and crashed the interpreter at shutdown, and a single shard
is both reproducible and cheap. Shard 0 of 14 holds ~70k conversations, ample for a 250 sample.

    uv run python scratch/wildchat_prompts.py --n 250
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

import fire
import pyarrow.parquet as pq

from src.huggingface import hf_download
from src.utils import timestamp, write_run_meta

DATASET = "allenai/WildChat-1M"
SHARD = "data/train-00000-of-00014.parquet"
COLUMNS = ["conversation", "language", "toxic", "turn", "conversation_hash", "model"]

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def non_latin_letters(text: str) -> int:
    """Count alphabetic characters in `text` NOT written in Latin script.

    WildChat's per-conversation `language` tag is unreliable — many turns tagged English are
    substantially Russian, Japanese or Chinese (verified 2026-08-10). Script is a far stronger
    signal than the tag and is robust to the code snippets common in this data: keywords and
    identifiers are Latin, so an English prose + Python paste counts zero non-Latin letters.

    A raw count rather than a fraction, because the fraction misses the worst case here — a
    short "translate into Chinese:" instruction whose long English payload dominates the ratio
    yet whose intent is not an English request. A count catches the CJK instruction directly.
    """
    return sum("LATIN" not in unicodedata.name(ch, "") for ch in _LETTER.findall(text))


def main(n: int = 250,
         scan_rows: int = 20000,
         seed: int = 0,
         min_chars: int = 40,
         max_chars: int = 6000,
         max_nonlatin_letters: int = 2,
         out_dir: str = "output/wildchat") -> None:
    """Sample opening user turns and write them to a timestamped JSONL.

    Args:
        n: How many prompts to keep.
        scan_rows: How many conversations to read from the shard before sampling.
        seed: RNG seed for the sample.
        min_chars: Drop prompts shorter than this (greetings carry no task).
        max_chars: Drop prompts longer than this (huge pastes crowd the context).
        max_nonlatin_letters: Drop prompts with more than this many non-Latin-script
            letters — the tag alone lets Russian/CJK prompts through (see
            non_latin_letters). A small tolerance keeps English prompts that quote a stray
            foreign character while rejecting anything actually written in another script.
        out_dir: Destination directory.
    """
    shard = hf_download(DATASET, SHARD, repo_type="dataset")
    table = pq.ParquetFile(shard).read_row_groups(
        list(range(pq.ParquetFile(shard).num_row_groups)), columns=COLUMNS)
    rows = table.slice(0, scan_rows).to_pylist()

    pool, seen, n_toxic_pool = [], set(), 0
    for row in rows:
        if row.get("toxic"):
            n_toxic_pool += 1
        if row.get("language") != "English":
            continue
        turns = row.get("conversation") or []
        if not turns or turns[0].get("role") != "user":
            continue
        text = (turns[0].get("content") or "").strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        if non_latin_letters(text) > max_nonlatin_letters:
            continue
        key = text[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        pool.append({"prompt": text,
                     "conversation_hash": row.get("conversation_hash"),
                     "toxic": bool(row.get("toxic")),
                     "n_turns": row.get("turn"),
                     "model": row.get("model")})

    assert len(pool) >= n, f"only {len(pool)} eligible prompts found; raise scan_rows"
    sample = random.Random(seed).sample(pool, n)

    ts = timestamp()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"wildchat_prompts_{n}_{ts}.jsonl"
    with path.open("w") as fh:
        for i, row in enumerate(sample):
            fh.write(json.dumps({"idx": i, **row}, ensure_ascii=False) + "\n")

    n_toxic = sum(r["toxic"] for r in sample)
    lens = sorted(len(r["prompt"]) for r in sample)
    write_run_meta(dest, {"dataset": DATASET, "shard": SHARD, "n": n, "seed": seed,
                          "scan_rows": scan_rows, "min_chars": min_chars,
                          "max_chars": max_chars},
                   extra={"path": str(path), "eligible_pool": len(pool),
                          "toxic_flagged": n_toxic,
                          "toxic_in_scanned_rows": n_toxic_pool,
                          "scanned_rows": len(rows)})

    print(f">>> {path}")
    print(f"    scanned {len(rows)} rows ({n_toxic_pool} toxic-flagged) -> eligible "
          f"{len(pool)} -> sample {n}")
    print(f"    toxic-flagged in sample: {n_toxic} ({n_toxic / n * 100:.1f}%)")
    print(f"    prompt chars: median {lens[n // 2]}, min {lens[0]}, max {lens[-1]}")
    print("\n--- first 3 prompts ---")
    for r in sample[:3]:
        print(f"\n[toxic={r['toxic']}] {r['prompt'][:400]}")


if __name__ == "__main__":
    fire.Fire(main)
