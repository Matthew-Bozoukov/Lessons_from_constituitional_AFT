# ABOUTME: Three item builders: application (LLM), retrieval (derived), fake_clause (deterministic).
# ABOUTME: Adding a family is a new class plus a config line - never an `if family == ...` branch.

from __future__ import annotations

from typing import Any

from ..control import loader
from ..core.registry import register
from ..core.types import Item, make_item_id
from .base import BuildContext, ItemBuildError

# Fields every generated scenario must carry. The rubrics interpolate the last two, so a scenario
# missing them cannot be scored.
SCENARIO_FIELDS = ("scenario", "norm_violating_path", "legitimate_goal")


def _make(**kwargs: Any) -> Item:
    """Construct an Item with a freshly computed content id.

    Args:
        **kwargs: Item fields except item_id.

    Returns:
        The Item with item_id set.
    """
    provisional = Item(item_id="", **kwargs)
    return Item(
        item_id=make_item_id(provisional),
        clause_id=provisional.clause_id,
        family=provisional.family,
        difficulty=provisional.difficulty,
        prompt=provisional.prompt,
        system=provisional.system,
        history=provisional.history,
        pressure=provisional.pressure,
        parent_item_id=provisional.parent_item_id,
        variant=provisional.variant,
        meta=provisional.meta,
    )


@register("builder", "application")
class ApplicationBuilder:
    """Scenario items where following the clause constrains what the model may do.

    The atom of the suite: retrieval items reuse these scenarios and the pressure transform wraps
    them, so one generated scenario feeds three measurements.
    """

    family = "application"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build one application item per (clause, difficulty, variant)."""
        cfg = ctx.family_cfg("application")
        difficulties = list(cfg.get("difficulties") or ["edge", "ambiguous"])
        variants = int(cfg.get("variants", 6))
        pack = ctx.pack
        template = pack["families"]["application"]["template"]
        rubrics = pack["difficulty"]

        # Slots are enumerated before dispatch so each job's domain draw uses a stable index and
        # the generated items are identical whether generation runs serially or threaded.
        jobs: list[dict[str, Any]] = []
        coords: list[tuple[Any, str, int]] = []
        for clause in ctx.clauses_in_scope():
            for difficulty in difficulties:
                if difficulty not in rubrics:
                    raise ItemBuildError(
                        f"No difficulty rubric for {difficulty!r} in control/prompts/items.yaml; "
                        f"declared: {sorted(rubrics)}"
                    )
                for variant in range(variants):
                    jobs.append(
                        {
                            "clause": clause,
                            "difficulty_rubric": rubrics[difficulty],
                            "domain": ctx.domain("family.application", len(jobs)),
                            "variant": variant,
                        }
                    )
                    coords.append((clause, difficulty, variant))

        payloads = ctx.generate_many(
            template, SCENARIO_FIELDS, pack["system"], jobs, desc="items:application"
        )
        return [
            _make(
                clause_id=clause.clause_id,
                family="application",
                difficulty=difficulty,
                prompt=str(payload["scenario"]).strip(),
                variant=variant,
                meta={
                    "norm_violating_path": str(payload["norm_violating_path"]).strip(),
                    "legitimate_goal": str(payload["legitimate_goal"]).strip(),
                },
            )
            for (clause, difficulty, variant), payload in zip(coords, payloads)
        ]


@register("builder", "retrieval")
class RetrievalBuilder:
    """Asks which clause governs a scenario the model has just been shown.

    Built from EVERY application scenario rather than a sample, so `knows` and `acts` are measured
    on exactly the same situations at exactly the same n. That is what makes the knowing-vs-acting
    plot a within-scenario comparison instead of two unrelated pools.
    """

    family = "retrieval"
    depends_on = ("application",)

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build one retrieval item per application item."""
        template = ctx.pack["templates"]["retrieval"]
        sources = sorted(built.get("application") or [], key=lambda i: i.item_id)
        return [
            _make(
                clause_id=source.clause_id,
                family="retrieval",
                difficulty=source.difficulty,
                prompt=loader.render(template, scenario=source.prompt),
                variant=source.variant,
                meta={"source_item_id": source.item_id},
            )
            for source in sources
        ]


@register("builder", "fake_clause")
class FakeClauseBuilder:
    """Pairs each fabricated clause with a genuine one from the same clause.

    Both halves are always built. Scoring acceptance of real clauses alongside rejection of
    fabricated ones is what turns recall - which saturates immediately - into discrimination.
    """

    family = "fake_clause"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build one fake probe and one matched real probe per distractor."""
        per_clause = int(ctx.family_cfg("fake_clause").get("per_clause", 3))
        template = ctx.pack["templates"]["fake_clause"]

        items: list[Item] = []
        for clause in ctx.clauses_in_scope():
            for variant, fake in enumerate(ctx.clauses.fakes_for(clause.clause_id)[:per_clause]):
                for is_real, text, extra in (
                    (False, fake.text, {"fake_id": fake.fake_id, "why_fake": fake.why_fake}),
                    (True, clause.text, {"fake_id": ""}),
                ):
                    items.append(
                        _make(
                            clause_id=clause.clause_id,
                            family="fake_clause",
                            difficulty="na",
                            prompt=loader.render(template, candidate_text=text),
                            variant=variant * 2 + int(is_real),
                            meta={"is_real": is_real, "candidate_text": text, **extra},
                        )
                    )
        return items
