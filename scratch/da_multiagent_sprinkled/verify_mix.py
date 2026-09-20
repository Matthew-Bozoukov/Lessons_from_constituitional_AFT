# ABOUTME: One-off check of the da-multiagent-sprinkled 7% mix against the da-7 mix it is compared with:
# ABOUTME: synthetic rows per principle and multi-agent share, and whether the base rows are the same rows.
"""Verify the mixture before training on it.

Run: uv run python scratch/da_multiagent_sprinkled/verify_mix.py <new mix repo> <spliced corpus repo> <revision>

Mixture rows carry only messages + source, so the synthetic rows are joined back to the
spliced corpus by their user message to recover principle ids.
"""

import hashlib
import json
import sys
from collections import Counter

from dotenv import load_dotenv

from src.infra.huggingface import resolve_dataset

from multiagent_rule import MULTI  # same directory

load_dotenv()
DA7 = (
    "dougalldeepmind/2026-09-15-da-7-mix",
    "c8a65ab574bef277aaefc55664c1d4dff03b4ef5",
)


def rows(repo, file, revision=None):
    path, ref = resolve_dataset(repo, file, revision)
    return [
        json.loads(line) for line in open(path, encoding="utf-8") if line.strip()
    ], ref


def key(r):
    return hashlib.sha256(
        json.dumps(r["messages"], sort_keys=True).encode()
    ).hexdigest()


def user(r):
    return next(m["content"] for m in r["messages"] if m["role"] == "user")


new, new_ref = rows(sys.argv[1], "mixture.jsonl")
old, old_ref = rows(*[DA7[0], "mixture.jsonl", DA7[1]])
corpus, _ = rows(sys.argv[2], "dataset.jsonl", sys.argv[3])
by_user = {user(r): r["metadata"] for r in corpus}

print("new mix:", new_ref["repo"], "@", new_ref["revision"], "| rows", len(new))
print("sources:", dict(Counter(r["source"] for r in new).most_common(4)), "...")
synth = [r for r in new if r["source"] == "da-multiagent-sprinkled"]
meta = [by_user[user(r)] for r in synth]
print(
    "synthetic rows:",
    len(synth),
    "| per principle:",
    dict(
        sorted(
            Counter(m["trait_id"] for m in meta).items(), key=lambda kv: int(kv[0][1:])
        )
    ),
)
swapped = [m for m in meta if m["trait_id"] in ("t1", "t2", "t6", "t7")]
multi = sum(
    1
    for m in swapped
    if MULTI.search(f"{m.get('situation') or ''} {m.get('shortcut') or ''}")
)
print(f"multi-agent among swapped principles' rows: {multi}/{len(swapped)}")

old_synth = {key(r) for r in old if r["source"] == "da"}
kept = sum(1 for r in synth if key(r) in old_synth)
print(f"synthetic rows identical to a da-7 synthetic row: {kept}/{len(synth)}")
base_new = Counter(key(r) for r in new if r["source"] != "da-multiagent-sprinkled")
base_old = Counter(key(r) for r in old if r["source"] != "da")
print(
    "base rows:",
    sum(base_new.values()),
    "vs da-7",
    sum(base_old.values()),
    "| identical multiset:",
    base_new == base_old,
)
