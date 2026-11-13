# ABOUTME: AgentHazard instances: the pinned HF dataset and the stratified subset of the paper's
# ABOUTME: 1,000 Claude Code instance ids that every arm runs (nested: a smaller draw is a prefix).
"""Which instances a run covers.

The paper's Claude Code numbers come from 1,000 of the 2,653 instances (~100 per risk category;
the ids are the file names of their published traces, saved in `claude_code_ids.json`). A run
draws `n_per_category` of those per category: each category's ids are shuffled with `seed` and
the first n kept, so a 10-per-category draw is contained in the 30-per-category one. The kept ids
are interleaved across categories (round robin) so `limit` (smoke) still spans categories.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

IDS_FILE = Path(__file__).parent / "claude_code_ids.json"


def load_dataset(
    repo: str, revision: str, filename: str = "dataset.json"
) -> list[dict]:
    """Download (cached) and parse the pinned AgentHazard dataset."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{repo}@{revision}/{filename} is not a non-empty list of instances"
        )
    return rows


def claude_code_ids() -> list[int]:
    return list(json.loads(IDS_FILE.read_text(encoding="utf-8"))["ids"])


def select(
    rows: list[dict],
    pool: list[int],
    n_per_category: int,
    seed: int,
    limit: int | None = None,
) -> list[dict]:
    """The stratified, nested, category-interleaved subset (see module docstring).

    Raises:
        ValueError: An id in the pool is missing from the dataset, or a category has fewer than
            n_per_category pooled instances.
    """
    by_id = {int(r["id"]): r for r in rows}
    missing = [i for i in pool if i not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} pooled ids are not in the dataset (first: {missing[:5]})"
        )
    by_cat: dict[str, list[int]] = {}
    for i in sorted(pool):
        by_cat.setdefault(by_id[i]["category"], []).append(i)
    picked: dict[str, list[int]] = {}
    for cat in sorted(by_cat):
        ids = by_cat[cat][:]
        random.Random(f"{seed}:{cat}").shuffle(ids)
        if len(ids) < n_per_category:
            raise ValueError(
                f"category {cat} has {len(ids)} pooled instances < n_per_category={n_per_category}"
            )
        picked[cat] = ids[:n_per_category]
    order = [picked[cat][k] for k in range(n_per_category) for cat in sorted(picked)]
    if limit:
        order = order[: int(limit)]
    return [by_id[i] for i in order]
