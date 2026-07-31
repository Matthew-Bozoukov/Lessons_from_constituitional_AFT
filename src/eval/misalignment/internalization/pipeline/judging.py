# ABOUTME: Fans each completion out to every judge its family declares, then flattens the
# ABOUTME: verdicts into results-store rows. The only place judges and the store meet.

from __future__ import annotations

from typing import Any, Sequence

from ..control import loader
from ..core.cache import CallCache, CacheConfig
from ..core.llm import CachedLLM, PriceTable, build_client, map_threaded
from ..core.store import RunContext, ScoreRow, build_rows
from ..core.types import ClauseSet, Completion, Item
from ..judges.base import JudgeConfig, build_judges


def build_judge_client(cfg: dict[str, Any], cache: CallCache | None = None) -> CachedLLM:
    """Instantiate the cached judge client.

    Args:
        cfg: Resolved run config.
        cache: Cache to share; a fresh one is built from the config when omitted.

    Returns:
        A CachedLLM.
    """
    block = dict(cfg.get("judge") or {})
    kwargs: dict[str, Any] = {}
    if block.get("base_url"):
        kwargs["base_url"] = block["base_url"]
    return CachedLLM(
        inner=build_client(str(block.get("provider", "openrouter")), **kwargs),
        cache=cache or CallCache(CacheConfig.from_config(cfg)),
        prices=PriceTable(cfg.get("pricing") or {}),
    )


def judge_all(
    items: Sequence[Item],
    completions: dict[str, Completion],
    clauses: ClauseSet,
    llm: CachedLLM,
    ctx: RunContext,
    config: JudgeConfig,
    max_workers: int = 16,
) -> list[ScoreRow]:
    """Score every completion on every axis its family declares.

    The (item, axis) pairs are flattened into one work list before dispatch rather than
    looped per item, so a family with five axes and a family with one saturate the pool
    equally instead of the pass stalling on whichever items happen to be judged last.

    Args:
        items: The items answered.
        completions: item_id -> Completion from the generation pass.
        clauses: The clause set the rubrics grade against.
        llm: Cached judge client.
        ctx: Run identity stamped onto every row.
        config: Judge model settings.
        max_workers: Concurrency.

    Returns:
        One ScoreRow per (item, axis).
    """
    judges = build_judges(clauses)
    work: list[tuple[Item, str]] = [
        (item, axis)
        for item in items
        for axis in loader.axes_for_family(item.family)
    ]

    def grade(pair: tuple[Item, str]):
        """Grade one (item, axis) pair."""
        item, axis = pair
        completion = completions.get(item.item_id) or Completion(
            item_id=item.item_id, text="", error="no completion for item"
        )
        return judges[axis](item, completion, llm, config)

    verdicts = map_threaded(grade, work, max_workers=max_workers, desc="judge")

    by_item: dict[str, list] = {}
    for (item, _), verdict in zip(work, verdicts):
        by_item.setdefault(item.item_id, []).append(verdict)

    rows: list[ScoreRow] = []
    seen: set[str] = set()
    for item, _ in work:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        clause = clauses.find(item.clause_id)
        rows.extend(
            build_rows(
                ctx,
                item,
                by_item[item.item_id],
                clause_title=clause.title if clause else item.clause_id,
                principle=clause.principle if clause else "",
                held_out=bool(clause.held_out) if clause else False,
                completion=completions.get(item.item_id),
            )
        )
    return rows
