# ABOUTME: Builds, freezes, and loads the versioned item set every run shares.
# ABOUTME: One frozen set per fingerprint; two runs on different fingerprints are never comparable.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..control import loader
from ..core.hashing import stable_hash, stream_rng
from ..core.registry import resolve
from ..core.types import ClauseSet, Item
from .base import BuildContext, ItemBuildError


@dataclass
class ItemSet:
    """A frozen, versioned collection of evaluation items.

    Attributes:
        itemset_id: Content fingerprint. Every results row carries it, and the report
            refuses to compare rows whose fingerprints differ - otherwise a quiet edit
            to one scenario would show up as a recipe effect.
        clause_set_id: The clause set the items were cut from.
        items: The items, in build order.
        meta: Build provenance (config, counts, git sha, timestamp).
    """

    itemset_id: str
    clause_set_id: str
    items: tuple[Item, ...]
    meta: dict[str, Any]

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self.items)

    def __iter__(self):
        """Iterate items in build order."""
        return iter(self.items)

    def by_id(self) -> dict[str, Item]:
        """Return an item_id -> Item index."""
        return {i.item_id: i for i in self.items}

    def of_family(self, family: str) -> list[Item]:
        """Return every item in one family."""
        return [i for i in self.items if i.family == family]

    def counts(self) -> dict[str, int]:
        """Return item counts by family and by condition kind, for the manifest."""
        out: dict[str, int] = {"total": len(self.items)}
        for item in self.items:
            out[f"family.{item.family}"] = out.get(f"family.{item.family}", 0) + 1
            kind = "clean" if not item.is_derived else item.condition.split(":")[0]
            out[f"condition.{kind}"] = out.get(f"condition.{kind}", 0) + 1
        return dict(sorted(out.items()))

    def subsample(self, n: int, seed: int = 0) -> ItemSet:
        """Return a smaller item set for a quick pass, keeping every pair intact.

        Base items are sampled and their derived children come along whole. Sampling the
        flat item list instead would orphan derived items from their parents, and the
        robustness deltas and OOD curves - which are paired differences - would silently
        lose most of their rows.

        The subset keeps the parent set's `itemset_id` suffixed with the cap, so results
        from a quick pass can never be pooled with results from the full set by accident.

        Args:
            n: Approximate number of base items to keep. 0 or >= the current size
                returns self.
            seed: Sampling seed.

        Returns:
            The reduced ItemSet, or self when no reduction was asked for.
        """
        base = [i for i in self.items if not i.is_derived]
        if n <= 0 or n >= len(base):
            return self

        # Sampled per family so a cap does not delete a whole family, which would drop an
        # eval axis from the report without saying so.
        by_family: dict[str, list[Item]] = {}
        for item in base:
            by_family.setdefault(item.family, []).append(item)
        per_family = max(1, n // len(by_family))

        keep: list[Item] = []
        for idx, family in enumerate(sorted(by_family)):
            pool = sorted(by_family[family], key=lambda i: i.item_id)
            rng = stream_rng(seed, idx, "subsample")
            rng.shuffle(pool)
            keep.extend(pool[:per_family])

        kept_ids = {i.item_id for i in keep}
        children = [i for i in self.items if i.parent_item_id in kept_ids]
        items = tuple(sorted([*keep, *children], key=lambda i: i.item_id))
        return ItemSet(
            itemset_id=f"{self.itemset_id}_n{len(keep)}",
            clause_set_id=self.clause_set_id,
            items=items,
            meta={**self.meta, "subsampled_from": self.itemset_id, "subsample_base_n": len(keep)},
        )

    def write(self, directory: Path | str) -> Path:
        """Freeze the item set to disk.

        Args:
            directory: Root under which `<itemset_id>/` is created.

        Returns:
            The item set directory.
        """
        out = Path(directory) / self.itemset_id
        out.mkdir(parents=True, exist_ok=True)
        with (out / "items.jsonl").open("w") as fh:
            for item in self.items:
                fh.write(json.dumps(item.to_dict()) + "\n")
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "itemset_id": self.itemset_id,
                    "clause_set_id": self.clause_set_id,
                    "counts": self.counts(),
                    **self.meta,
                },
                indent=2,
            )
        )
        return out

    @classmethod
    def load(cls, directory: Path | str) -> ItemSet:
        """Load a frozen item set from disk.

        Args:
            directory: The `<itemset_id>/` directory.

        Returns:
            The ItemSet.

        Raises:
            FileNotFoundError: If the directory has no items.jsonl.
        """
        root = Path(directory)
        items_path = root / "items.jsonl"
        if not items_path.exists():
            raise FileNotFoundError(f"No items.jsonl under {root}")
        items = tuple(
            Item.from_dict(json.loads(line)) for line in items_path.read_text().splitlines() if line.strip()
        )
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return cls(
            itemset_id=manifest.get("itemset_id", root.name),
            clause_set_id=manifest.get("clause_set_id", ""),
            items=items,
            meta=manifest,
        )

    @classmethod
    def find(cls, directory: Path | str, itemset_id: str | None = None) -> ItemSet:
        """Load a named item set, or the most recently written one.

        Args:
            directory: Root holding `<itemset_id>/` directories.
            itemset_id: Which set to load; None picks the newest.

        Returns:
            The ItemSet.

        Raises:
            FileNotFoundError: If nothing matches.
        """
        root = Path(directory)
        if itemset_id:
            if (root / itemset_id / "items.jsonl").exists():
                return cls.load(root / itemset_id)
            # A capped run records a suffixed id that was never frozen on its own, so a
            # later cross-check would otherwise be unable to find the items it ran on.
            base = re.sub(r"_n\d+$", "", itemset_id)
            if base != itemset_id and (root / base / "items.jsonl").exists():
                return cls.load(root / base)
            return cls.load(root / itemset_id)
        candidates = sorted(
            (p for p in root.glob("*/items.jsonl")), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(
                f"No frozen item set under {root}. Build one with "
                f"`scripts/eval/run_internalization.sh items build`."
            )
        return cls.load(candidates[0].parent)


def _order_builders(families: list[str]) -> list[str]:
    """Topologically order builders by their declared dependencies.

    Args:
        families: Enabled family names.

    Returns:
        The families in an order where every dependency precedes its dependent.

    Raises:
        ItemBuildError: If a dependency is not enabled or the graph has a cycle.
    """
    enabled = set(families)
    ordered: list[str] = []
    pending = list(families)
    while pending:
        progressed = False
        for family in list(pending):
            deps = tuple(getattr(resolve("builder", family), "depends_on", ()))
            missing = [d for d in deps if d not in enabled]
            if missing:
                raise ItemBuildError(
                    f"Family {family!r} depends on {missing}, which is not enabled. Enable it "
                    f"under itemset.families or disable {family!r}."
                )
            if all(d in ordered for d in deps):
                ordered.append(family)
                pending.remove(family)
                progressed = True
        if not progressed:
            raise ItemBuildError(f"Cyclic builder dependencies among {pending}")
    return ordered


def build_itemset(
    cfg: dict[str, Any],
    clauses: ClauseSet,
    llm: Any = None,
    meta: dict[str, Any] | None = None,
) -> ItemSet:
    """Build the full item set from a resolved config.

    Base items are built first, in dependency order, then every enabled transform runs
    over the families it declares. The fingerprint is computed from the resulting item
    ids, so any change to a scenario, a wrapper, or a clause yields a new id.

    Args:
        cfg: The resolved run config.
        clauses: The frozen clause set.
        llm: Cached client for scenario generation; None for a deterministic-only build.
        meta: Extra provenance merged into the manifest.

    Returns:
        The ItemSet.

    Raises:
        ItemBuildError: If a builder or transform is misconfigured.
    """
    itemset_cfg = dict(cfg.get("itemset") or {})
    ctx = BuildContext(
        clauses=clauses,
        cfg=itemset_cfg,
        llm=llm,
        seed=int(cfg.get("seed", 0)),
        domains=list(itemset_cfg.get("domains") or ()),
        max_workers=int(cfg.get("max_workers", 16)),
    )

    families = [
        name
        for name, spec in (itemset_cfg.get("families") or {}).items()
        if (spec or {}).get("enabled", True)
    ]
    built: dict[str, list[Item]] = {}
    for family in _order_builders(sorted(families)):
        built[family] = resolve("builder", family)()(ctx, built)

    derived: list[Item] = []
    for name, spec in (itemset_cfg.get("transforms") or {}).items():
        if not (spec or {}).get("enabled", True):
            continue
        transform = resolve("transform", name)()
        parents = [i for f in transform.applies_to for i in built.get(f, [])]
        if not parents:
            continue
        derived.extend(transform(ctx, parents))

    items = tuple([i for family in sorted(built) for i in built[family]] + derived)
    if not items:
        raise ItemBuildError("Item set is empty; every family and transform is disabled.")

    duplicates = sorted({i.item_id for i in items if [x.item_id for x in items].count(i.item_id) > 1})
    if duplicates:
        raise ItemBuildError(
            f"Item set contains duplicate item ids: {duplicates[:5]}. Two items with identical "
            f"coordinates and prompts were built; bump `variant` or the generator temperature."
        )

    fingerprint = stable_hash(
        {
            "clause_set": clauses.fingerprint,
            "items": sorted(i.item_id for i in items),
        },
        12,
    )
    return ItemSet(
        itemset_id=f"is_{fingerprint}",
        clause_set_id=clauses.spec_id,
        items=items,
        meta={
            "clause_set_fingerprint": clauses.fingerprint,
            "n_clauses": len(clauses),
            "n_held_out": len(clauses.held_out),
            "held_out_clauses": [c.clause_id for c in clauses.held_out],
            "itemset_config": itemset_cfg,
            **(meta or {}),
        },
    )


def resolve_clause_set(cfg: dict[str, Any]) -> ClauseSet:
    """Load the clause set a config names.

    Args:
        cfg: Resolved run config.

    Returns:
        The ClauseSet.
    """
    return loader.clause_set(str(cfg.get("clause_set")))
