# ABOUTME: One builder per item family. Adding a family is a new class here plus a config
# ABOUTME: line - never an `if family == ...` inside an existing builder.

from __future__ import annotations

from typing import Any

from ..control import loader
from ..core.registry import register
from ..core.types import Item, make_item_id
from .base import BuildContext, ItemBuildError

# Fields every generated scenario must carry. The judge rubrics interpolate the last two,
# so a scenario without them cannot be scored.
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
        ood_axis=provisional.ood_axis,
        ood_value=provisional.ood_value,
        parent_item_id=provisional.parent_item_id,
        variant=provisional.variant,
        meta=provisional.meta,
    )


def _clauses_in_scope(ctx: BuildContext):
    """Return the clauses to build items for.

    Held-out clauses are excluded from *training data* generation, never from
    evaluation - evaluating them is the whole reason they were held out.
    """
    if ctx.cfg.get("include_held_out", True):
        return list(ctx.clauses.clauses)
    return list(ctx.clauses.trained)


@register("builder", "application")
class ApplicationBuilder:
    """Scenario items where following the clause constrains what the model may do.

    The atom of the suite: retrieval items reuse these scenarios, and both transforms
    derive from them.
    """

    family = "application"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build one application item per (clause, difficulty, variant)."""
        cfg = ctx.family_cfg("application")
        difficulties = list(cfg.get("difficulties") or ["clear", "edge", "ambiguous"])
        variants = int(cfg.get("variants", 2))
        pack = ctx.pack
        template = pack["families"]["application"]["template"]
        rubrics = pack["difficulty"]

        # Slots are enumerated first so each job's domain draw uses the same index the
        # serial build used; the generated items are byte-identical either way.
        jobs: list[dict[str, Any]] = []
        coords: list[tuple[Any, str, int]] = []
        for clause in _clauses_in_scope(ctx):
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

    Built from application scenarios rather than fresh ones, so retrieval and
    application are measured on the same situations. That is what makes the
    retrieval-vs-application gap a statement about one scenario rather than a
    comparison of two unrelated item pools.
    """

    family = "retrieval"
    depends_on = ("application",)

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build retrieval items from the first N application items per clause."""
        cfg = ctx.family_cfg("retrieval")
        per_clause = int(cfg.get("variants", 1))
        template = ctx.pack["templates"]["retrieval"]

        by_clause: dict[str, list[Item]] = {}
        for item in built.get("application") or []:
            by_clause.setdefault(item.clause_id, []).append(item)

        items: list[Item] = []
        for clause_id in sorted(by_clause):
            # Sorted by id, not by build order, so the selection is stable when the
            # application count changes.
            sources = sorted(by_clause[clause_id], key=lambda i: i.item_id)[:per_clause]
            for variant, source in enumerate(sources):
                items.append(
                    _make(
                        clause_id=clause_id,
                        family="retrieval",
                        difficulty=source.difficulty,
                        prompt=loader.render(template, scenario=source.prompt),
                        variant=variant,
                        meta={"source_item_id": source.item_id},
                    )
                )
        return items


@register("builder", "fake_clause")
class FakeClauseBuilder:
    """Pairs each fabricated clause with a genuine one from the same clause.

    Both halves are always built. Scoring acceptance of real clauses alongside
    rejection of fabricated ones is what turns recall - which saturates immediately -
    into discrimination, which does not.
    """

    family = "fake_clause"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build one fake probe and one matched real probe per distractor."""
        cfg = ctx.family_cfg("fake_clause")
        per_clause = int(cfg.get("per_clause", 1))
        template = ctx.pack["templates"]["fake_clause"]

        items: list[Item] = []
        for clause in _clauses_in_scope(ctx):
            fakes = ctx.clauses.fakes_for(clause.clause_id)[:per_clause]
            for variant, fake in enumerate(fakes):
                for is_real, text, meta_extra in (
                    (False, fake.text, {"fake_id": fake.fake_id, "why_fake": fake.why_fake}),
                    (True, clause.text, {"fake_id": ""}),
                ):
                    items.append(
                        _make(
                            clause_id=clause.clause_id,
                            family="fake_clause",
                            difficulty="na",
                            prompt=loader.render(template, candidate_text=text),
                            variant=variant,
                            meta={
                                "is_real": is_real,
                                "candidate_text": text,
                                **meta_extra,
                            },
                        )
                    )
        return items


@register("builder", "conflict")
class ConflictBuilder:
    """Scenarios where two clauses pull in opposite directions.

    Only pairs whose priority tiers differ are used: with equal tiers the spec's
    ordering names no winner, so there would be nothing to grade the resolution against.
    """

    family = "conflict"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build conflict items over deterministically sampled clause pairs."""
        cfg = ctx.family_cfg("conflict")
        n_pairs = int(cfg.get("pairs", 12))
        variants = int(cfg.get("variants", 1))
        template = ctx.pack["families"]["conflict"]["template"]

        clauses = _clauses_in_scope(ctx)
        candidates = [
            (a, b)
            for i, a in enumerate(clauses)
            for b in clauses[i + 1 :]
            if a.priority_tier != b.priority_tier and a.principle != b.principle
        ]
        if not candidates:
            raise ItemBuildError(
                "No clause pair has both differing priority tiers and differing principles, so no "
                "conflict item has a gradeable expected winner."
            )
        # Sorted before sampling so the draw does not depend on clause file ordering.
        candidates.sort(key=lambda p: (p[0].clause_id, p[1].clause_id))
        rng = ctx.rng("family.conflict")
        rng.shuffle(candidates)
        chosen = candidates[: min(n_pairs, len(candidates))]

        jobs: list[dict[str, Any]] = []
        coords: list[tuple[Any, Any, int]] = []
        for pair_idx, (x, y) in enumerate(chosen):
            # Lower tier dominates, and is always presented to the judge as criterion A.
            primary, secondary = (x, y) if x.priority_tier < y.priority_tier else (y, x)
            for variant in range(variants):
                jobs.append(
                    {
                        "clause": primary,
                        "secondary": secondary,
                        "domain": ctx.domain("family.conflict", pair_idx * variants + variant),
                        "variant": variant,
                    }
                )
                coords.append((primary, secondary, variant))

        payloads = ctx.generate_many(
            template, SCENARIO_FIELDS, ctx.pack["system"], jobs, desc="items:conflict"
        )
        return [
            _make(
                clause_id=primary.clause_id,
                family="conflict",
                difficulty="edge",
                prompt=str(payload["scenario"]).strip(),
                variant=variant,
                meta={
                    "secondary_clause_id": secondary.clause_id,
                    "expected_winner": "a",
                    "norm_violating_path": str(payload["norm_violating_path"]).strip(),
                    "legitimate_goal": str(payload["legitimate_goal"]).strip(),
                },
            )
            for (primary, secondary, variant), payload in zip(coords, payloads)
        ]


@register("builder", "over_refusal")
class OverRefusalBuilder:
    """Benign requests inside each clause's subject matter.

    The side-effect control. Trait data leaks: a model trained toward a clause starts
    refusing things that merely sound like the clause's territory, and a suite that
    only counts harms would score that as an improvement.
    """

    family = "over_refusal"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build benign in-territory requests per clause."""
        cfg = ctx.family_cfg("over_refusal")
        variants = int(cfg.get("variants", 2))
        template = ctx.pack["families"]["over_refusal"]["template"]

        jobs: list[dict[str, Any]] = []
        coords: list[tuple[Any, int]] = []
        for clause in _clauses_in_scope(ctx):
            for variant in range(variants):
                jobs.append(
                    {
                        "clause": clause,
                        "domain": ctx.domain("family.over_refusal", len(jobs)),
                        "variant": variant,
                    }
                )
                coords.append((clause, variant))

        payloads = ctx.generate_many(
            template, ("scenario", "legitimate_goal"), ctx.pack["system"], jobs,
            desc="items:over_refusal",
        )
        return [
            _make(
                clause_id=clause.clause_id,
                family="over_refusal",
                difficulty="na",
                prompt=str(payload["scenario"]).strip(),
                variant=variant,
                meta={
                    "legitimate_goal": str(payload["legitimate_goal"]).strip(),
                    "norm_violating_path": "none - this request is benign",
                },
            )
            for (clause, variant), payload in zip(coords, payloads)
        ]


@register("builder", "persona_drift")
class PersonaDriftBuilder:
    """Fixed general-manner probes, identical across clauses and recipes.

    Deliberately not clause-specific and not generated: this axis is only interpretable
    if the prompts are byte-identical between the recipes being compared.
    """

    family = "persona_drift"
    depends_on: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, built: dict[str, list[Item]]) -> list[Item]:
        """Build the persona probes declared in the prompt pack."""
        prompts = list(ctx.pack["templates"]["persona_drift"])
        n = int(ctx.family_cfg("persona_drift").get("n", len(prompts)))
        return [
            _make(
                clause_id="_global",
                family="persona_drift",
                difficulty="na",
                prompt=prompt,
                variant=variant,
                meta={},
            )
            for variant, prompt in enumerate(prompts[:n])
        ]
