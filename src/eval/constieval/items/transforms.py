# ABOUTME: The single transform: wrap EVERY application item in one adversarial system prompt.
# ABOUTME: Every item is stressed, so the paired clean/stressed delta has no item-composition confound.

from __future__ import annotations

from ..control import loader
from ..core.registry import register
from ..core.types import Item
from .base import BuildContext, ItemBuildError


@register("transform", "pressure")
class PressureTransform:
    """Re-wraps clean application items under an adversarial operator system prompt.

    Deterministic and prompt-preserving: the wrapper adds a system prompt and never touches the
    scenario, so the paired delta measures robustness rather than a different item.

    Applied to ALL application items, not a per-clause sample. v1 sampled, which left only 10 of 42
    items stressed by every wrapper and 10 stressed by none - so "pressure hurts more than OOD" was
    confounded with which items each wrapper happened to get.
    """

    name = "pressure"
    applies_to = ("application",)

    def __call__(self, ctx: BuildContext, parents: list[Item]) -> list[Item]:
        """Derive one stressed item per parent, per configured wrapper."""
        cfg = dict((ctx.cfg.get("transforms") or {}).get("pressure") or {})
        wrappers = list(cfg.get("wrappers") or ())
        if not wrappers:
            return []

        derived: list[Item] = []
        for parent in sorted(parents, key=lambda i: i.item_id):
            for name in wrappers:
                spec = loader.wrapper(name)
                if spec["kind"] != "system":
                    raise ItemBuildError(
                        f"Pressure wrapper {name!r} has kind {spec['kind']!r}; only 'system' is "
                        f"supported now that the prefix/history wrappers are gone."
                    )
                derived.append(parent.derive(pressure=name, system=str(spec["system"]).strip()))
        return derived
