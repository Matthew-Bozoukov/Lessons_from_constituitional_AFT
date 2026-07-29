# ABOUTME: ChunkGroupers sit between Chunker and Sampler and decide which chunks share
# ABOUTME: a document. grouping_strategy is written to every row, so comparing them is a groupby.

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

from ..core.registry import register, resolve
from ..core.types import SpecChunk


class GroupingError(ValueError):
    """Raised when a grouping strategy cannot satisfy the requested k."""


class BaseGrouper:
    """Shared plumbing: params, a stats counter, and the document-order index."""

    strategy = "base"

    def __init__(self, chunks: Sequence[SpecChunk], params: dict[str, Any] | None = None,
                 context: Any = None) -> None:
        """Initialize.

        Args:
            chunks: The full chunk pool for the spec.
            params: Strategy params from recipe.grouping_params.
            context: Optional GroupingContext supplying the embedding index.
        """
        self.chunks = list(chunks)
        self.params = dict(params or {})
        self.context = context
        self.stats: dict[str, int] = defaultdict(int)
        self.by_id = {c.chunk_id: c for c in self.chunks}

    def group(self, chunks: Sequence[SpecChunk], k: int,
              rng: random.Random) -> tuple[SpecChunk, ...]:  # pragma: no cover - abstract
        """Return a k-chunk group drawn from the pool."""
        raise NotImplementedError


@register("grouping", "single")
class SingleGrouper(BaseGrouper):
    """Degenerate strategy for k == 1. Registered so single rows resolve uniformly."""

    strategy = "single"

    def group(self, chunks: Sequence[SpecChunk], k: int,
              rng: random.Random) -> tuple[SpecChunk, ...]:
        """Return one uniformly chosen chunk (k is ignored beyond a sanity check)."""
        if k != 1:
            raise GroupingError(f"single grouping requires k == 1, got {k}")
        return (rng.choice(list(chunks)),)


@register("grouping", "random")
class RandomGrouper(BaseGrouper):
    """k chunks sampled uniformly at random, no relation assumed.

    Produces distant pairings. Useful for trait_conflict and for testing whether the
    model can hold two unrelated principles at once.
    """

    strategy = "random"

    def group(self, chunks: Sequence[SpecChunk], k: int,
              rng: random.Random) -> tuple[SpecChunk, ...]:
        """Return k distinct chunks chosen uniformly without replacement."""
        pool = list(chunks)
        if len(pool) < k:
            raise GroupingError(f"random grouping needs {k} chunks, pool has {len(pool)}")
        return tuple(rng.sample(pool, k))


@register("grouping", "adjacent")
class AdjacentGrouper(BaseGrouper):
    """k contiguous chunks, preserving the spec's own structure and local context.

    Params:
        same_section_only: When True (default) a group never crosses a parent_id
            boundary. When False, groups are contiguous in whole-document order and
            may spill into the next section.
    """

    strategy = "adjacent"

    def __init__(self, chunks, params=None, context=None) -> None:
        """Build the per-section and whole-document orderings once."""
        super().__init__(chunks, params, context)
        self.same_section_only = bool(self.params.get("same_section_only", True))

        # Document order: sections in first-appearance order, chunks by order_idx.
        section_order: dict[str, int] = {}
        for c in self.chunks:
            section_order.setdefault(c.parent_id, len(section_order))
        self.doc_order = sorted(
            self.chunks, key=lambda c: (section_order[c.parent_id], c.order_idx)
        )
        self.sections: dict[str, list[SpecChunk]] = defaultdict(list)
        for c in self.doc_order:
            self.sections[c.parent_id].append(c)

    def group(self, chunks: Sequence[SpecChunk], k: int,
              rng: random.Random) -> tuple[SpecChunk, ...]:
        """Return k contiguous chunks."""
        if not self.same_section_only:
            if len(self.doc_order) < k:
                raise GroupingError(
                    f"adjacent grouping needs {k} chunks, spec has {len(self.doc_order)}"
                )
            start = rng.randrange(0, len(self.doc_order) - k + 1)
            return tuple(self.doc_order[start : start + k])

        eligible = [p for p, items in self.sections.items() if len(items) >= k]
        if not eligible:
            largest = max((len(v) for v in self.sections.values()), default=0)
            raise GroupingError(
                f"adjacent grouping with same_section_only=true needs a section with "
                f"{k} chunks; the largest has {largest}. Use a finer chunker "
                "granularity, lower chunks_per_example, or set same_section_only: false."
            )
        # Weight sections by how many windows they offer, so every contiguous window
        # is equally likely rather than every section being equally likely.
        weights = [len(self.sections[p]) - k + 1 for p in eligible]
        parent = rng.choices(eligible, weights=weights, k=1)[0]
        items = self.sections[parent]
        start = rng.randrange(0, len(items) - k + 1)
        self.stats["groups"] += 1
        return tuple(items[start : start + k])


@register("grouping", "semantic")
class SemanticGrouper(BaseGrouper):
    """k chunks that are near neighbours in embedding space.

    Groups related material the spec happens to separate across sections. The upper
    similarity bound matters: without it the strategy collapses onto near-duplicate
    chunks and produces degenerate documents that say the same thing twice.

    Params:
        min_similarity: Floor for a candidate neighbour.
        max_similarity: Ceiling, excluding near-duplicates.
        max_anchor_tries: Anchors to try before falling back.
    """

    strategy = "semantic"

    def __init__(self, chunks, params=None, context=None) -> None:
        """Build or fetch the cached embedding index for this spec."""
        super().__init__(chunks, params, context)
        self.min_similarity = float(self.params.get("min_similarity", 0.6))
        self.max_similarity = float(self.params.get("max_similarity", 0.95))
        self.max_anchor_tries = int(self.params.get("max_anchor_tries", 12))
        if context is None or getattr(context, "index", None) is None:
            raise GroupingError(
                "semantic grouping requires an embedding index. The pipeline builds "
                "one automatically; if you are calling the grouper directly, pass "
                "context=GroupingContext(index=...)."
            )
        self.index = context.index

    def group(self, chunks: Sequence[SpecChunk], k: int,
              rng: random.Random) -> tuple[SpecChunk, ...]:
        """Return an anchor plus its k-1 nearest in-band neighbours."""
        pool = [c for c in chunks if c.chunk_id in self.index.pos]
        if len(pool) < k:
            raise GroupingError(f"semantic grouping needs {k} chunks, pool has {len(pool)}")

        best_partial: tuple[SpecChunk, ...] | None = None
        for _ in range(self.max_anchor_tries):
            anchor = rng.choice(pool)
            sims = self.index.neighbours(anchor.chunk_id)
            scored = [
                (float(sims[self.index.pos[c.chunk_id]]), c)
                for c in pool
                if c.chunk_id != anchor.chunk_id
            ]
            in_band = [
                (s, c) for s, c in scored if self.min_similarity <= s <= self.max_similarity
            ]
            in_band.sort(key=lambda t: -t[0])
            if len(in_band) >= k - 1:
                self.stats["in_band"] += 1
                return (anchor, *[c for _, c in in_band[: k - 1]])
            if best_partial is None:
                under_ceiling = sorted(
                    (t for t in scored if t[0] <= self.max_similarity), key=lambda t: -t[0]
                )
                if len(under_ceiling) >= k - 1:
                    best_partial = (anchor, *[c for _, c in under_ceiling[: k - 1]])

        # No anchor had enough in-band neighbours. Take the nearest below the ceiling
        # rather than killing the run, and count it so the manifest shows the shortfall.
        self.stats["below_floor_fallback"] += 1
        if best_partial is None:
            raise GroupingError(
                f"semantic grouping could not assemble {k} chunks under "
                f"max_similarity={self.max_similarity}. Raise the ceiling or lower k."
            )
        return best_partial


class GroupingContext:
    """Carries the shared embedding index to whichever groupers need it."""

    def __init__(self, index: Any = None) -> None:
        """Initialize with an optional EmbeddingIndex."""
        self.index = index


def build_groupers(
    strategies: Sequence[str],
    chunks: Sequence[SpecChunk],
    params: dict[str, dict[str, Any]] | None = None,
    context: GroupingContext | None = None,
) -> dict[str, BaseGrouper]:
    """Instantiate one grouper per named strategy.

    Args:
        strategies: Strategy names from the recipe's grouping mixture.
        chunks: The chunk pool.
        params: recipe.grouping_params, keyed by strategy name.
        context: Shared grouping context (embedding index).

    Returns:
        Mapping of strategy name to grouper instance.
    """
    params = params or {}
    out: dict[str, BaseGrouper] = {}
    for name in set(strategies) | {"single"}:
        out[name] = resolve("grouping", name)(chunks, params.get(name, {}), context)
    return out
