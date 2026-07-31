# ABOUTME: BuildContext plus the Builder and Transform protocols every item plugin implements.
# ABOUTME: Builders create base items; transforms derive stressed items from them, always paired.

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol, Sequence, runtime_checkable

from ..control import loader
from ..core.hashing import stream_rng
from ..core.llm import CachedLLM, map_threaded
from ..core.parsing import ParseError, extract_json
from ..core.types import ClauseSet, Item


class ItemBuildError(RuntimeError):
    """Raised when item generation fails in a way a retry cannot fix."""


@dataclass
class BuildContext:
    """Everything a builder or transform needs, injected rather than imported.

    Attributes:
        clauses: The frozen clause set.
        cfg: The resolved `itemset` config block.
        llm: Cached client used for scenario generation. None for a deterministic-only
            build, in which case any builder needing a completion raises.
        seed: Run seed; every draw goes through a named stream off this.
        domains: Settings sampled to keep scenarios for one clause from all looking alike.
        max_workers: Concurrency for scenario generation.
    """

    clauses: ClauseSet
    cfg: dict[str, Any]
    llm: CachedLLM | None = None
    seed: int = 0
    domains: Sequence[str] = ()
    max_workers: int = 16
    _pack: dict[str, Any] = field(default_factory=dict)

    @property
    def pack(self) -> dict[str, Any]:
        """The `items` prompt pack, loaded once."""
        if not self._pack:
            self._pack = loader.pack("items")
        return self._pack

    def rng(self, stream: str, index: int = 0) -> Random:
        """Return an independent RNG for one named stream.

        Each family and transform draws from its own stream, so enabling a family or
        bumping one count leaves every other family's draws bit-identical and old
        item ids keep matching.

        Args:
            stream: Stream name, e.g. "family.application".
            index: Index within the stream.

        Returns:
            A seeded Random.
        """
        return stream_rng(self.seed, index, stream)

    def domain(self, stream: str, index: int) -> str:
        """Deterministically pick a domain for one item slot."""
        if not self.domains:
            return "an ordinary workplace"
        return self.rng(f"{stream}.domain", index).choice(list(self.domains))

    def family_cfg(self, family: str) -> dict[str, Any]:
        """Return the config block for one family, or {} if absent."""
        return dict((self.cfg.get("families") or {}).get(family) or {})

    def clauses_in_scope(self) -> list[Any]:
        """Return the clauses to build items for.

        Held-out clauses are excluded from *training data* generation, never from evaluation -
        evaluating them is the whole reason they were held out.
        """
        if self.cfg.get("include_held_out", True):
            return list(self.clauses.clauses)
        return list(self.clauses.trained)

    def generate_many(
        self,
        template: str,
        required: Sequence[str],
        system: str,
        jobs: Sequence[dict[str, Any]],
        *,
        scope: str = "item_gen",
        desc: str = "items",
    ) -> list[dict[str, Any]]:
        """Generate many scenarios concurrently, preserving job order.

        Builders enumerate their whole job list up front and hand it here rather than
        calling `generate_json` in a loop. Serially, a few hundred scenarios take tens of
        minutes and the item build dominates wall-clock for the entire suite.

        Order is preserved and every job's template context is fixed before dispatch, so
        the items produced are byte-identical to the serial version - which matters
        because item ids are content hashes, and a reordering would silently invalidate
        every previously frozen item set.

        Args:
            template: Jinja2 template source from the prompt pack.
            required: Field names that must be present in each parsed object.
            system: System prompt for the generator.
            jobs: One mapping of template variables per scenario.
            scope: Cache scope.
            desc: Progress bar description.

        Returns:
            The parsed objects, in job order.
        """
        if not jobs:
            return []
        return map_threaded(
            lambda job: self.generate_json(template, required, system, scope=scope, **job),
            list(jobs),
            max_workers=self.max_workers,
            desc=desc,
        )

    def generate_json(
        self,
        template: str,
        required: Sequence[str],
        system: str,
        *,
        scope: str = "item_gen",
        **context: Any,
    ) -> dict[str, Any]:
        """Render a generation template, call the generator, and parse its JSON.

        Retries only on a parse failure - a model that returned prose instead of JSON
        will usually comply on a second attempt, while a transport error has already
        been retried inside the client.

        Args:
            template: Jinja2 template source from the prompt pack.
            required: Field names that must be present in the parsed object.
            system: System prompt for the generator.
            scope: Cache scope.
            **context: Template variables.

        Returns:
            The parsed object.

        Raises:
            ItemBuildError: If no generator is configured, or parsing fails every attempt.
        """
        if self.llm is None:
            raise ItemBuildError(
                "This builder needs a generator model but none is configured. Run with "
                "itemset.generator.provider=echo for an offline build."
            )
        gen = dict(self.cfg.get("generator") or {})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": loader.render(template, **context)},
        ]
        params: dict[str, Any] = {
            "temperature": float(gen.get("temperature", 1.0)),
            "max_tokens": int(gen.get("max_tokens", 3000)),
        }
        if gen.get("extra_body"):
            params["extra_body"] = dict(gen["extra_body"])
        model = str(gen.get("model", "google/gemini-2.5-flash"))

        last: Exception | None = None
        for attempt in range(3):
            # A retry must not replay the cached bad response, so each later attempt carries a
            # different nudge - which changes the cache key as well as the instruction.
            nudges = [
                None,
                "Return only the JSON object, nothing else.",
                "Return only the JSON object, and keep `scenario` under 130 words.",
            ]
            msgs = messages if nudges[attempt] is None else [
                *messages,
                {"role": "user", "content": nudges[attempt]},
            ]
            resp = self.llm.call(scope=scope, model=model, messages=msgs, params=params)
            try:
                # A truncated response is a budget problem, not a formatting one. Saying so
                # directly saves the next person from debugging the JSON parser.
                if resp.finish_reason == "length":
                    raise ParseError(
                        f"generator hit max_tokens={params['max_tokens']} and was cut off "
                        f"mid-object. Raise itemset.generator.max_tokens, or disable the "
                        f"generator's reasoning - on a thinking model the trace is billed against "
                        f"the same budget as the answer."
                    )
                parsed = extract_json(resp.content)
                if not isinstance(parsed, dict):
                    raise ParseError(f"Generator returned {type(parsed).__name__}, expected object")
                missing = [f for f in required if not str(parsed.get(f, "")).strip()]
                if missing:
                    raise ParseError(f"Generator output missing fields {missing}")
                return parsed
            except ParseError as e:
                last = e
        raise ItemBuildError(f"Generator produced unusable output after 3 attempts: {last}")


@runtime_checkable
class Builder(Protocol):
    """Creates the base items for one family.

    `depends_on` names families whose items must exist first. The assembler
    topologically orders builders from it, so a builder never reaches for a family
    that has not been built.
    """

    family: str
    depends_on: tuple[str, ...]

    def __call__(
        self, ctx: BuildContext, built: dict[str, list[Item]]
    ) -> list[Item]:  # pragma: no cover - protocol
        """Return the base items for this family."""
        ...


@runtime_checkable
class Transform(Protocol):
    """Derives stressed items from base items.

    A transform must return items whose `parent_item_id` points at the item they came
    from. That pairing is what the robustness deltas and OOD decay curves are computed
    from; an unpaired derived item is dropped by the analysis layer.
    """

    name: str
    applies_to: tuple[str, ...]

    def __call__(
        self, ctx: BuildContext, parents: list[Item]
    ) -> list[Item]:  # pragma: no cover - protocol
        """Return derived items."""
        ...
