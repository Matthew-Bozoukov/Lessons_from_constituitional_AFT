# ABOUTME: Transforms that derive stressed items from clean parents: adversarial pressure
# ABOUTME: wrappers (deterministic) and OOD distance-axis rewrites (one generator call each).

from __future__ import annotations

from typing import Any

from ..control import loader
from ..core.registry import register
from ..core.types import Item
from .base import BuildContext, ItemBuildError


def _sample_parents(ctx: BuildContext, parents: list[Item], per_clause: int, stream: str) -> list[Item]:
    """Pick a stable subset of parents, `per_clause` from each clause.

    Sampling per clause rather than globally keeps the clause coverage of the derived
    items even, so a per-clause robustness bar is never computed from zero items for
    some clauses and six for others.

    Args:
        ctx: Build context.
        parents: Candidate parent items.
        per_clause: How many parents to take from each clause.
        stream: RNG stream name.

    Returns:
        The selected parents, sorted by item id.
    """
    by_clause: dict[str, list[Item]] = {}
    for item in parents:
        by_clause.setdefault(item.clause_id, []).append(item)

    chosen: list[Item] = []
    for idx, clause_id in enumerate(sorted(by_clause)):
        pool = sorted(by_clause[clause_id], key=lambda i: i.item_id)
        rng = ctx.rng(stream, idx)
        rng.shuffle(pool)
        chosen.extend(pool[:per_clause])
    return sorted(chosen, key=lambda i: i.item_id)


@register("transform", "pressure")
class PressureTransform:
    """Re-wraps clean items under adversarial pressure.

    Deterministic and prompt-preserving by construction: a wrapper may add a system
    prompt, a prefix, or prior turns, but never touches the scenario. If it rewrote the
    scenario the paired delta would confound robustness with item difficulty.
    """

    name = "pressure"
    applies_to = ("application",)

    def __call__(self, ctx: BuildContext, parents: list[Item]) -> list[Item]:
        """Derive one item per (sampled parent, wrapper)."""
        cfg = dict((ctx.cfg.get("transforms") or {}).get("pressure") or {})
        wrappers = list(cfg.get("wrappers") or ())
        if not wrappers:
            return []
        sampled = _sample_parents(
            ctx, parents, int(cfg.get("per_clause", 1)), "transform.pressure"
        )

        derived: list[Item] = []
        for parent in sampled:
            for name in wrappers:
                spec = loader.wrapper(name)
                kind = spec["kind"]
                if kind == "system":
                    changes = {"system": str(spec["system"]).strip()}
                elif kind == "prefix":
                    changes = {"prompt": f"{str(spec['prefix']).strip()}\n\n{parent.prompt}"}
                elif kind == "history":
                    changes = {
                        "history": tuple(
                            {"role": t["role"], "content": " ".join(str(t["content"]).split())}
                            for t in spec["history"]
                        )
                    }
                else:  # pragma: no cover - loader.wrapper already rejects other kinds
                    raise ItemBuildError(f"Pressure wrapper {name!r} has unknown kind {kind!r}")
                derived.append(parent.derive(pressure=name, **changes))
        return derived


@register("transform", "ood")
class OODTransform:
    """Rewrites clean items along one labelled distance axis at a time.

    Distance 0 is never rewritten - it *is* the parent, and is the anchor each decay
    curve starts from. Only distances above 0 cost a generator call.
    """

    name = "ood"
    applies_to = ("application",)

    def __call__(self, ctx: BuildContext, parents: list[Item]) -> list[Item]:
        """Derive one item per (sampled parent, axis, non-anchor axis value)."""
        cfg = dict((ctx.cfg.get("transforms") or {}).get("ood") or {})
        axis_names = list(cfg.get("axes") or ())
        if not axis_names:
            return []
        sampled = _sample_parents(ctx, parents, int(cfg.get("per_clause", 1)), "transform.ood")
        # 0 = no cap. The far end of each axis is the most expensive third of the item set
        # and the least often decisive, so it is the first thing to drop on a budget.
        max_distance = int(cfg.get("max_distance", 0))
        pack = loader.pack("ood")

        jobs: list[dict[str, Any]] = []
        coords: list[tuple[Item, str, dict[str, Any]]] = []
        for parent in sampled:
            for axis_name in axis_names:
                axis = loader.ood_axis(axis_name)
                for value in axis["values"]:
                    distance = int(value.get("distance", 0))
                    if distance == 0 or (max_distance and distance > max_distance):
                        continue
                    jobs.append(
                        {
                            "prompt": parent.prompt,
                            "norm_violating_path": parent.meta.get("norm_violating_path", ""),
                            "axis_title": axis.get("title", axis_name),
                            "instruction": str(value["instruction"]).strip(),
                        }
                    )
                    coords.append((parent, axis_name, value))

        payloads = ctx.generate_many(
            pack["template"], ("scenario",), pack["system"], jobs, desc="items:ood"
        )
        return [
            parent.derive(
                prompt=str(payload["scenario"]).strip(),
                ood_axis=axis_name,
                ood_value=str(value["name"]),
                meta={**parent.meta, "ood_distance": int(value["distance"])},
            )
            for (parent, axis_name, value), payload in zip(coords, payloads)
        ]
