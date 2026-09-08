# ABOUTME: Substitute the generated reward-hacking rows for an equal per-trait slice of the 708-row
# ABOUTME: difficult-advice corpus and push the result to HF as a NEW dataset repo with a card.

"""Substitution, not mutation.

The source corpus (`2026-08-21-difficult-advice-v2-chunk-only-716`) is what several trained arms
were built from, so it is never edited in place — this writes a new repo and the old one keeps
resolving to exactly what those arms trained on.

The swap is per-trait and equal-sized: `PER_TRAIT` rows removed and `PER_TRAIT` inserted for each
of the nine traits, so every trait count in the corpus is unchanged and the mixture builder's
equal-quota requirement (the reason the published draw is 702 and not 716) still holds.
Which rows leave is chosen by a seeded shuffle, so the substitution is reproducible.
"""

import collections
import json
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from src.infra.huggingface import hf_org  # noqa: E402
from src.utils import git_sha  # noqa: E402

SOURCE = "LASR-Callum/2026-08-21-difficult-advice-v2-chunk-only-716"
GENERATED = Path("scratch/rh_generated_rows.jsonl")
LOCAL = Path("output/rh_corpus")
NAME = "2026-09-04-da-rewardhack-351-synth"
SEED = 0


def build() -> tuple[Path, dict]:
    src = hf_hub_download(SOURCE, "dataset.jsonl", repo_type="dataset")
    corpus = [json.loads(line) for line in open(src, encoding="utf-8")]
    new = [json.loads(line) for line in GENERATED.open(encoding="utf-8")]

    by_trait_new = collections.defaultdict(list)
    for r in new:
        by_trait_new[r["metadata"]["trait_id"]].append(r)
    # Renumber per trait. The generator numbers by position in its job list, and a top-up pass
    # restarts from the COUNT of existing rows — which collides with ids already taken wherever
    # the first pass had gaps (21 duplicates on the first assembly). Renumbering here makes the
    # ids a function of the final corpus rather than of how many passes it took to fill.
    for tid, rows_ in by_trait_new.items():
        for i, r in enumerate(rows_):
            r["metadata"]["scenario_id"] = f"rh_{tid}_g{i:03d}"
    per_trait = min(len(v) for v in by_trait_new.values())
    assert per_trait, "no generated rows"

    by_trait_old = collections.defaultdict(list)
    for r in corpus:
        by_trait_old[r["metadata"]["trait_id"]].append(r)

    rng = random.Random(SEED)
    kept, swapped_out = [], []
    for tid, rows in sorted(by_trait_old.items()):
        order = rows[:]
        rng.shuffle(order)
        swapped_out += order[:per_trait]
        kept += order[per_trait:] + by_trait_new[tid][:per_trait]

    counts_before = {k: len(v) for k, v in sorted(by_trait_old.items())}
    counts_after = dict(sorted(collections.Counter(
        r["metadata"]["trait_id"] for r in kept).items()))
    assert counts_before == counts_after, f"trait balance broken: {counts_before} -> {counts_after}"

    LOCAL.mkdir(parents=True, exist_ok=True)
    out = LOCAL / "dataset.jsonl"
    kept.sort(key=lambda r: (r["metadata"]["trait_id"], r["metadata"]["scenario_id"]))
    with out.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total": len(kept), "substituted": per_trait * len(counts_before),
        "per_trait_substituted": per_trait, "per_trait_counts": counts_after,
        "source_corpus": SOURCE, "source_rows": len(corpus),
        "family_counts": dict(sorted(collections.Counter(
            r["metadata"].get("family", "difficult-advice") for r in kept).items())),
        "removed_scenario_ids": sorted(r["metadata"]["scenario_id"] for r in swapped_out),
        "seed": SEED,
    }
    (LOCAL / "substitution_stats.json").write_text(json.dumps(stats, indent=2))
    return out, stats


def card(stats: dict) -> str:
    return f"""---
tags: [difficult-advice, reward-hacking, synthetic]
---

# Difficult advice with a reward-hacking substitution ({stats['substituted']} of {stats['total']} rows)

`{SOURCE}` with **{stats['per_trait_substituted']} rows per trait ({stats['substituted']} total)
replaced** by newly generated rows in which a student, a professional, or another AI model asks
whether to cheat for their own advancement — an admission, a first-author line, a deployment
decision, a job offer, a ranking — and the assistant declines while conceding the true part of
their argument and giving concrete help.

| field | value |
| --- | --- |
| `experiment` | Reward-hacking substitution arm: {stats['substituted']} of {stats['total']} difficult-advice rows replaced with scenarios where the asker wants to cheat to get ahead. Built to test whether difficult-advice training transfers to specification gaming (ctfish, ODCV data fabrication). |
| `date_generated` | 2026-09-04 |
| `constitution` | constitutions/claude_distilled_12_principles_mid/constitution.md — the same constitution the source corpus was written against; each substituted row is written against the same `trait_text` string, read from the source corpus at build time rather than retyped. |
| `source_repo` | teaching_claude_why_replication @ {git_sha()} |
| `models` | substituted rows: anthropic/claude-sonnet-5 (pinned first-party via configs/endpoints/providers.yaml). Retained rows: as published in {SOURCE}. |
| `generation_config` | {json.dumps({'generator': 'scratch/_gen_rh_corpus.py', 'model': 'anthropic/claude-sonnet-5', 'temperature': 1.0, 'max_tokens': 8000, 'per_trait': stats['per_trait_substituted'], 'seed': SEED, 'retries': 1})} |
| `schema` | `dataset.jsonl`: {{messages:[system,user,assistant(+reasoning_content)], metadata:{{scenario_id, trait_id, trait_name, trait_text, domain, shortcut, situation, family, asker}}}} — identical to the source corpus, plus `family` and `asker` on substituted rows. |
| `provenance` | `uv run python scratch/_gen_rh_corpus.py --per_trait={stats['per_trait_substituted']}` then `uv run python scratch/_push_rh_corpus.py`. Removed row ids are listed in `substitution_stats.json`. |

## What differs from the source corpus

- **Per-trait balance is preserved exactly**: {stats['per_trait_counts']}. The mixture builder
  requires equal per-trait quotas — that is why the published draw is 702 and not 716 — so the
  substitution removes and inserts the same number of rows per trait.
- **The asker's motive is self-advancement, never hardship.** They state the ambition plainly and
  nothing bad happens to them if they decline; only the *reasoning* is partly right (the metric
  really is a poor proxy, the practice really is widespread, the grader really is buggy).
- **No assistant-identity framing.** Every substituted system prompt is a deployment context only —
  no assistant is named. The source corpus does this in 702 of its 708 rows; here it is 100%.

Row families: {json.dumps(stats['family_counts'])}

**Not yet reviewed.** The substituted rows are model-generated from five hand-written seeds and
have passed schema and length validation only.
"""


def main(push: bool = True) -> None:
    out, stats = build()
    print(json.dumps({k: v for k, v in stats.items() if k != "removed_scenario_ids"}, indent=2))
    (LOCAL / "README.md").write_text(card(stats), encoding="utf-8")
    if not push:
        print(f"built {out} (push skipped)")
        return
    repo = f"{hf_org()}/{NAME}"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(LOCAL), repo_id=repo, repo_type="dataset")
    print(f">>> pushed https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
