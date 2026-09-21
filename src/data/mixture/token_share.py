# ABOUTME: Supervised-token share for a mixture: count what the trainer will actually put in the loss,
# ABOUTME: swap base rows out and synthetic rows in so the swap is a chosen share of THOSE tokens.

"""A mixture whose synthetic share is a share of supervised tokens, not of rows.

Under token weighting (`train.loss_agg: token_mean`, 2026-09-21) a source's share of the
gradient is its share of SUPERVISED tokens — the assistant tokens the mask leaves in the loss
— and a row share says nothing about that: 7% of rows was 15.4% of supervised tokens on
`2026-09-15-da-7-mix`, because difficult-advice answers are long. This module builds the
share in the unit that matters:

  * the published nosynth mixture is the size of every arm — its supervised-token total is
    the budget, and nothing is added on top;
  * base rows are REMOVED in a seeded order that takes from every source in proportion and
    is nested across percentages (the 5% arm's removals are the first part of the 7% arm's),
    until the removed tokens reach the declared share of the budget;
  * synthetic rows are INSERTED, round-robin over the balance groups (traits) so the share is
    trait-balanced in tokens, until the freed tokens are refilled — greedy, then one
    closest-fitting row to close the gap;
  * every count is `build_labels` on the row as the trainer will render and mask it, under
    the row's OWN supervision mode (`all` / `final` / `cot` / `answer`, and any `mask_spans`),
    so a cot-only arm and its control get the token shares their masks give them, not the
    ones their rows would have.

The planning functions are pure and unit-tested on fake counts; `supervised_tokens` is the
one that touches a tokenizer.
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.model_profile import ModelProfile, render_chat
from src.train.masking import build_labels

SHARE_UNITS = ("examples", "supervised_tokens")


def supervised_tokens(tok, profile: ModelProfile, row: dict, max_seq_len: int) -> int:
    """How many of this row's tokens the trainer will supervise, under the row's own mode.

    Renders with the family's preserve kwargs and masks with `build_labels`, exactly as
    `uv run train` does (src/train/train_lora.py), so `supervise: final` counts one turn,
    `cot` counts one turn's reasoning and truncates after it, and `mask_spans` remove what
    they remove. The row's `supervise` field is whatever the mixture assigned it — a
    source-level override included — so this is the count the arm will train on.
    """
    text = render_chat(tok, row["messages"], row.get("tools"), render_kwargs=profile.render_kwargs)
    labels = build_labels(text, tok, max_seq_len, profile,
                          supervise=row.get("supervise") or "all",
                          mask_spans=row.get("mask_spans"))["labels"]
    # The causal shift: position 0 is never a target, and the trainer's loss counts
    # labels[1:] (src/train/dynamic_batching.py). Label 0 is -100 anyway (a prompt token).
    return sum(1 for v in labels[1:] if v != -100)


def removal_order(rows: list[dict], seed: int) -> list[int]:
    """The order in which base rows leave, proportional across sources and nested across shares.

    Each source's rows are shuffled with their own seeded stream and given the key
    (rank + 0.5) / size; the global order sorts by that key. After the first k removals
    every source has lost about k x (its row share) — the base blend keeps its proportions,
    the way `blend()` kept them for row shares — and the order does not depend on the share
    asked for, so a larger share removes a superset of what a smaller one removes.

    Returns:
        Indices into `rows`, first to leave first.
    """
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r["source"]].append(i)
    keyed = []
    for name in sorted(by_source):
        idx = list(by_source[name])
        random.Random(f"{seed}:{name}").shuffle(idx)
        n = len(idx)
        keyed += [((rank + 0.5) / n, name, rank, i) for rank, i in enumerate(idx)]
    keyed.sort()
    return [i for _k, _s, _r, i in keyed]


def remove_until(rows: list[dict], order: list[int], target: int) -> list[int]:
    """The first rows of `order` whose supervised tokens reach `target` (the last one may
    overshoot; the closest-fit is decided on the insert side, where the pool is larger)."""
    removed, total = [], 0
    for i in order:
        if total >= target:
            break
        removed.append(i)
        total += rows[i]["n_supervised"]
    return removed


def fill_tokens(pool: list[dict], budget: int, seed: int, balance_key: str | None) -> list[int]:
    """Choose synthetic rows whose supervised tokens refill `budget`, balanced across groups.

    Round-robin over the balance groups (seeded order, each group seeded-shuffled): a
    group's next row is taken when it fits, skipped for this round when it would overshoot.
    When no group can add a row without overshooting, one more row — the one that brings
    the total closest to the budget, from any group — is added if it reduces the gap.

    Returns:
        Indices into `pool`.
    """
    if budget <= 0:
        return []
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(pool):
        groups[str(r.get(balance_key, "")) if balance_key else ""].append(i)
    names = sorted(groups)
    random.Random(f"{seed}:groups").shuffle(names)
    for name in names:
        random.Random(f"{seed}:{name}").shuffle(groups[name])
    cursor = {name: 0 for name in names}
    chosen, total = [], 0
    progress = True
    while progress:
        progress = False
        for name in names:
            idx = groups[name]
            while cursor[name] < len(idx):
                i = idx[cursor[name]]
                if total + pool[i]["n_supervised"] <= budget:
                    chosen.append(i)
                    total += pool[i]["n_supervised"]
                    cursor[name] += 1
                    progress = True
                    break
                cursor[name] += 1  # too big for what is left; the next one may fit
    gap = budget - total
    remaining = [i for name in names for i in groups[name] if i not in set(chosen)]
    if gap > 0 and remaining:
        best = min(remaining, key=lambda i: (abs(gap - pool[i]["n_supervised"]), i))
        if abs(gap - pool[best]["n_supervised"]) < gap:
            chosen.append(best)
    return chosen


def plan_swap(base: list[dict], synth: list[dict], pct: int, seed: int,
              balance_key: str | None) -> dict:
    """Which base rows leave and which synthetic rows enter for a `pct` share of supervised tokens.

    Args:
        base: The whole published base mixture, each row with `n_supervised` and `source`.
        synth: The synthetic pool, each row with `n_supervised` (+ the balance field).
        pct: The declared synthetic share, in percent of the base's supervised tokens.
        seed: The mixture seed.
        balance_key: Field to balance the inserted rows over (e.g. `trait_id`), or None.

    Returns:
        {"keep": base indices kept, "insert": synth indices, "budget", "removed_tokens",
         "inserted_tokens", "removed_by_source", "inserted_by_group", "realised_pct"}.
    """
    assert 0 <= pct <= 100, pct
    budget = sum(r["n_supervised"] for r in base)
    target = round(budget * pct / 100)
    order = removal_order(base, seed)
    removed = remove_until(base, order, target)
    removed_set = set(removed)
    freed = sum(base[i]["n_supervised"] for i in removed)
    inserted = fill_tokens(synth, freed, seed, balance_key) if pct else []
    ins_tokens = sum(synth[i]["n_supervised"] for i in inserted)
    kept_tokens = budget - freed
    by_source: dict[str, int] = defaultdict(int)
    for i in removed:
        by_source[base[i]["source"]] += 1
    by_group: dict[str, int] = defaultdict(int)
    for i in inserted:
        by_group[str(synth[i].get(balance_key, "")) if balance_key else ""] += 1
    total = kept_tokens + ins_tokens
    return {"keep": [i for i in range(len(base)) if i not in removed_set], "insert": inserted,
            "budget": budget, "target": target, "removed_tokens": freed, "inserted_tokens": ins_tokens,
            "removed_by_source": dict(sorted(by_source.items())),
            "inserted_by_group": dict(sorted(by_group.items())),
            "realised_pct": round(100 * ins_tokens / total, 2) if total else 0.0,
            "total_supervised_tokens": total}
