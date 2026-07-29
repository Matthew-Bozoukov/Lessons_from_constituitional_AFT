# ABOUTME: Recipe parsing and the mixture sampler. Emits ScenarioSpecs - experimental
# ABOUTME: conditions - using one RNG stream per axis so sweep arms stay paired.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .hashing import stream_rng
from .types import ScenarioSpec, SpecChunk

# Recipe keys with dedicated meaning. Everything else in the recipe block is read as
# an axis mixture, which is how a new axis becomes a config edit rather than a code edit.
RESERVED = {"n", "chunks_per_example", "grouping", "grouping_params", "doc_type"}


class RecipeError(ValueError):
    """Raised when a recipe is malformed or internally inconsistent."""


def normalize(mixture: dict[Any, Any], what: str) -> dict[str, float]:
    """Validate a mixture and normalize its weights to sum to 1.

    Args:
        mixture: Mapping of option -> non-negative weight.
        what: Name used in error messages.

    Returns:
        Mapping of str(option) -> normalized weight, with keys sorted.

    Raises:
        RecipeError: If empty, negative, or all-zero.
    """
    if not mixture:
        raise RecipeError(f"{what} mixture is empty")
    out: dict[str, float] = {}
    for key, weight in mixture.items():
        w = float(weight)
        if w < 0:
            raise RecipeError(f"{what}[{key}] has negative weight {w}")
        out[str(key)] = w
    total = sum(out.values())
    if total <= 0:
        raise RecipeError(f"{what} mixture weights sum to {total}")
    return {k: out[k] / total for k in sorted(out)}


def draw(mixture: dict[str, float], rng: random.Random) -> str:
    """Draw one option from a normalized mixture.

    Args:
        mixture: Normalized mixture (key order must be stable).
        rng: Seeded RNG.

    Returns:
        The chosen key.
    """
    keys = list(mixture)
    return rng.choices(keys, weights=[mixture[k] for k in keys], k=1)[0]


@dataclass
class Recipe:
    """A parsed, validated recipe block.

    Attributes:
        n: Number of scenarios to sample.
        chunks_per_example: Mixture over k (keys are stringified ints).
        grouping: Mixture over grouping strategy names.
        grouping_params: Per-strategy params passed to the grouper.
        doc_type: Mixture over registered doc types.
        axes: Every other recipe key, read as an axis mixture.
    """

    n: int
    chunks_per_example: dict[str, float]
    grouping: dict[str, float]
    grouping_params: dict[str, dict[str, Any]]
    doc_type: dict[str, float]
    axes: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Recipe:
        """Parse a recipe config block.

        Args:
            cfg: The `recipe:` mapping from the run config.

        Returns:
            A validated Recipe.

        Raises:
            RecipeError: If required fields are missing or malformed.
        """
        cfg = dict(cfg or {})
        for required in ("n", "chunks_per_example", "grouping", "doc_type"):
            if required not in cfg:
                raise RecipeError(f"recipe.{required} is required")

        k_mix = normalize(cfg["chunks_per_example"], "chunks_per_example")
        for key in k_mix:
            try:
                if int(key) < 1:
                    raise ValueError
            except ValueError:
                raise RecipeError(
                    f"chunks_per_example keys must be positive integers, got {key!r}"
                ) from None

        axes = {
            name: normalize(value, f"axis {name}")
            for name, value in cfg.items()
            if name not in RESERVED
        }
        return cls(
            n=int(cfg["n"]),
            chunks_per_example=k_mix,
            grouping=normalize(cfg["grouping"], "grouping"),
            grouping_params={k: dict(v) for k, v in (cfg.get("grouping_params") or {}).items()},
            doc_type=normalize(cfg["doc_type"], "doc_type"),
            axes=axes,
        )

    @property
    def axis_names(self) -> list[str]:
        """Sorted axis names. Fixed for a run, so the snapshot schema is stable."""
        return sorted(self.axes)

    @property
    def strategies(self) -> list[str]:
        """Grouping strategies with non-zero weight, plus 'single' for k == 1."""
        return sorted({s for s, w in self.grouping.items() if w > 0} | {"single"})


class MixtureSampler:
    """Samples ScenarioSpecs from a recipe.

    Paired sweeps depend entirely on this class. Each decision for example i draws
    from its own RNG stream keyed on (seed, i, decision), so changing one axis's
    mixture perturbs only that axis's draws; every other field of example i stays
    bit-identical across arms. Sampling with a single sequential RNG would reshuffle
    everything downstream of the first changed draw and destroy the pairing.

    The per-example seed is part of scenario_hash, which is what keeps doc_ids unique
    when two examples happen to draw the same condition.
    """

    def __init__(self, groupers: dict[str, Any], seed: int = 0) -> None:
        """Initialize.

        Args:
            groupers: Mapping of strategy name -> grouper instance.
            seed: Run-level seed shared by every arm of a sweep.
        """
        self.groupers = groupers
        self.seed = int(seed)

    def sample(
        self, chunks: Sequence[SpecChunk], recipe: Recipe, n: int | None = None
    ) -> Iterator[ScenarioSpec]:
        """Yield n ScenarioSpecs.

        Args:
            chunks: The chunk pool.
            recipe: The parsed recipe.
            n: Override for recipe.n.

        Yields:
            ScenarioSpec instances, one per experimental condition.

        Raises:
            RecipeError: If a grouping strategy in the mixture has no grouper.
        """
        total = int(n if n is not None else recipe.n)
        pool = list(chunks)
        if not pool:
            raise RecipeError("chunk pool is empty; check the chunker and the spec file")

        for i in range(total):
            k = int(draw(recipe.chunks_per_example, stream_rng(self.seed, i, "k")))
            # k == 1 short-circuits grouping and is recorded as "single" so
            # single-chunk rows stay joinable with the rest of the corpus.
            strategy = (
                "single"
                if k == 1
                else draw(recipe.grouping, stream_rng(self.seed, i, "grouping"))
            )
            grouper = self.groupers.get(strategy)
            if grouper is None:
                raise RecipeError(
                    f"No grouper built for strategy {strategy!r}; "
                    f"available: {sorted(self.groupers)}"
                )
            group = grouper.group(pool, k, stream_rng(self.seed, i, "chunks"))

            doc_type = draw(recipe.doc_type, stream_rng(self.seed, i, "doc_type"))
            axes = {
                name: draw(mix, stream_rng(self.seed, i, f"axis.{name}"))
                for name, mix in sorted(recipe.axes.items())
            }
            yield ScenarioSpec(
                chunks=tuple(group),
                grouping_strategy=strategy,
                doc_type=doc_type,
                axes=axes,
                seed=stream_rng(self.seed, i, "seed").randrange(2**31),
                sample_index=i,
            )
