# ABOUTME: Deterministic, repo-stratified, NESTED subsetting of a SWE-bench split, so a 10%
# ABOUTME: run is a strict subset of a 20% run and every depth is repo-proportional.

"""Choosing which SWE-bench instances to run, and proving which ones ran.

Depth is the knob that makes this eval affordable, and the obvious implementations are both
wrong:

- **Positional slicing** (mini-swe-agent's own `--slice 0:50`) takes the dataset in its
  natural order, which is clustered by repository. On Verified that hands you a sample that
  is mostly `astropy` and no `django` at all — a number that is not an estimate of the
  benchmark, just of whichever repos sort first.
- **Independent random samples per depth** are repo-balanced but *not nested*: extending 10%
  to 20% redraws, so the first run's rollouts cannot be reused and the two numbers move for
  two different reasons at once.

The construction here gives all three properties at once. Every instance gets a rank inside
its own repo (by a seeded hash of its id), that rank is normalized to (0, 1), and the global
order sorts by that normalized position. Any prefix of the resulting order therefore holds
roughly the same *share* of every repo — and because it is one fixed order, a shorter prefix
is always a strict subset of a longer one. Extending a run costs only the new instances, and
the cheap slice stays an honest preview of the expensive one.

Nothing here touches the network; `load_instances` is the only function that does, and it is
deliberately separate so the selection logic is unit-testable offline.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


def _rank_key(seed: int, instance_id: str) -> str:
    """Seeded, stable per-instance sort key. Hex digest, not a float — no FP tie weirdness."""
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def stratified_order(instances: Iterable[dict], seed: int) -> list[dict]:
    """Order instances so that EVERY prefix is repo-proportional and prefixes nest.

    Args:
        instances: Dicts carrying at least `instance_id` and `repo`.
        seed: Selection seed. Changing it redraws the whole benchmark subset — it is part
            of the subset's identity, not a nuisance parameter to tune per run.

    Returns:
        A new list, ordered. Deterministic for a given (instances, seed): the final sort key
        includes the instance id, so even identical hashes cannot make the order depend on
        input order.
    """
    by_repo: dict[str, list[dict]] = {}
    for row in instances:
        by_repo.setdefault(row["repo"], []).append(row)

    positioned: list[tuple[float, str, dict]] = []
    for repo, rows in by_repo.items():
        # Rank within the repo by the seeded hash, then map rank -> (0, 1) by its position in
        # the group. A repo with 5 instances lands at .1/.3/.5/.7/.9; one with 100 lands every
        # .01 — so a prefix cuts each repo at the same relative depth regardless of its size.
        ranked = sorted(rows, key=lambda r: _rank_key(seed, r["instance_id"]))
        for i, row in enumerate(ranked):
            positioned.append(((i + 0.5) / len(ranked), _rank_key(seed, row["instance_id"]), row))
    positioned.sort(key=lambda t: (t[0], t[1], t[2]["instance_id"]))
    return [row for _, _, row in positioned]


def select(instances: Iterable[dict], seed: int, *, fraction: float | None = None,
           n: int | None = None) -> list[dict]:
    """Take the first `n` (or `fraction`) instances of the stratified order.

    Exactly one of `fraction`/`n` must be given — accepting both invites a config where they
    disagree and the run silently honours one of them.

    Args:
        instances: The full split.
        seed: Selection seed (see `stratified_order`).
        fraction: Share of the split, e.g. 0.1. Rounded to the nearest instance, min 1.
        n: Absolute instance count.

    Returns:
        The selected instances, in stratified order.
    """
    pool = stratified_order(instances, seed)
    if (fraction is None) == (n is None):
        raise ValueError("pass exactly one of fraction= or n=")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n = max(1, round(fraction * len(pool)))
    if not 0 < n <= len(pool):
        raise ValueError(f"n must be in [1, {len(pool)}], got {n}")
    return pool[:n]


def shard(instances: list[dict], index: int, count: int) -> list[dict]:
    """Split an already-selected subset across N drivers, round-robin.

    Round-robin WITHIN each repo, not over the flat list. Two weaker schemes were rejected:
    contiguous blocks hand one driver the front of every repo's ranking and the other the
    back, so the halves are not exchangeable; and flat round-robin aliases against the
    stratified order, because a repo's items sit at a near-constant stride and can beat
    against the modulus (measured: 3-way split gave one shard 14 django instances where 16.7
    were due). Counting per repo bounds the error at ±1 instance per repo per shard by
    construction, for any number of shards.

    Shards are disjoint and their union is exactly the input, so pass@1 over the merged
    result is scored against the FULL subset — a shard is a division of labour, never a
    different benchmark.

    Args:
        instances: The selected subset, in stratified order.
        index: 0-based shard number.
        count: Total shards.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for count {count}")
    seen: dict[str, int] = {}
    out = []
    for row in instances:  # stratified order preserved in the output
        rank = seen.get(row["repo"], 0)
        seen[row["repo"]] = rank + 1
        if rank % count == index:
            out.append(row)
    return out


def subset_hash(instances: Iterable[dict], seed: int, dataset: str, revision: str) -> str:
    """Identity of a subset: which instances, drawn how, from which dataset revision.

    Two runs may only be pooled or compared when this matches. It covers the dataset
    revision because upstream has revised instances in place before — same ids, different
    content — which is invisible in an id list alone.

    Returns:
        12 hex chars: short enough for a filename, wide enough not to collide here.
    """
    ids = sorted(row["instance_id"] for row in instances)
    payload = "\n".join([dataset, revision, str(seed), *ids])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def id_filter_regex(instances: Iterable[dict]) -> str:
    """An anchored alternation over instance ids, for mini-swe-agent's `--filter`.

    Anchored and escaped deliberately: `--filter` is applied as a regex against instance
    ids, so an unescaped id could match more than intended, and an unanchored pattern makes
    a short id a prefix of a longer one (`django__django-1` matching `django__django-11099`)
    — which silently runs instances that are not in the subset.
    """
    ids = sorted(row["instance_id"] for row in instances)
    if not ids:
        raise ValueError("refusing to build a filter for an empty subset")
    return "^(" + "|".join(re.escape(i) for i in ids) + ")$"


def repo_breakdown(instances: Iterable[dict]) -> dict[str, int]:
    """Instances per repo — printed before a run so a skewed draw is visible immediately."""
    counts: dict[str, int] = {}
    for row in instances:
        counts[row["repo"]] = counts.get(row["repo"], 0) + 1
    return dict(sorted(counts.items()))


def load_instances(dataset: str, split: str) -> tuple[list[dict], str]:
    """Fetch a SWE-bench split and the exact dataset revision it came from (network).

    The revision is resolved and returned rather than assumed, because it is part of the
    result's identity: the rollout phase and the grading phase must be pinned to the SAME
    revision or "resolved" is measured against different tests than the agent saw.

    Returns:
        (rows with `instance_id`/`repo`, dataset revision sha).
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(dataset).sha
    rows = load_dataset(dataset, split=split, revision=revision)
    # `image_name`/`docker_image` are carried when the split declares them: the pre-pull step
    # must resolve the SAME image the agent will run, and the dataset's own field takes
    # precedence over the derived name (see images.image_name).
    keep = ("instance_id", "repo", "image_name", "docker_image")
    return ([{k: r[k] for k in keep if k in r} for r in rows], revision)


def summarize_selection(selected: list[dict], total: int, seed: int, dataset: str,
                        revision: str, *, full: list[dict] | None = None,
                        shard_index: int | None = None,
                        shard_count: int | None = None) -> dict[str, Any]:
    """The selection block recorded in run_meta.json and echoed in the report line.

    When sharded, `full` is the whole selected subset and `selected` is this driver's slice.
    Both hashes are recorded: `subset_hash` always identifies the FULL subset (the thing
    pass@1 is scored against, and the thing two drivers must agree on), while `shard_hash`
    identifies this driver's slice. Merging shards means checking `subset_hash` matches and
    unioning the instance lists.
    """
    whole = full if full is not None else selected
    block = {
        "dataset": dataset,
        "dataset_revision": revision,
        "split_size": total,
        "n_selected": len(whole),
        "fraction": len(whole) / total if total else 0.0,
        "seed": seed,
        "subset_hash": subset_hash(whole, seed, dataset, revision),
        "sampling": "repo-stratified nested prefix (src/eval/capabilities/swebench_mini/subset.py)",
        "repo_breakdown": repo_breakdown(selected),
        "instance_ids": sorted(row["instance_id"] for row in selected),
    }
    if shard_count and shard_count > 1:
        block |= {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "n_in_shard": len(selected),
            "shard_hash": subset_hash(selected, seed, dataset, revision),
            "full_instance_ids": sorted(row["instance_id"] for row in whole),
        }
    return block
