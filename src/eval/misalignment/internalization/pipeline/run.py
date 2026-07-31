# ABOUTME: End-to-end orchestration: freeze items, generate once, judge every axis, write
# ABOUTME: the results store and the manifest. One call per checkpoint.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import git_sha, make_run_id, timestamp
from ..core.cache import CallCache, CacheConfig
from ..core.llm import CachedLLM, PriceTable, build_client
from ..core.store import ResultsStore, RunContext
from ..items.itemset import ItemSet, build_itemset, resolve_clause_set
from ..judges.base import JudgeConfig
from .generate import TargetConfig, build_target, generate
from .judging import build_judge_client, judge_all
from .side_effects import capability_rows, generation_health


@dataclass
class RunResult:
    """What one evaluation run produced.

    Attributes:
        run_id: The run's id.
        run_dir: Directory holding results, completions, and the manifest.
        itemset_id: Fingerprint of the item set used.
        store: The results store.
        manifest: Provenance, counts, health, and spend.
        paths: Named output paths.
        warnings: Conditions that make the run's numbers untrustworthy. Never empty
            silently - a caller that ignores these is reporting a broken run as a result.
    """

    run_id: str
    run_dir: Path
    itemset_id: str
    store: ResultsStore
    manifest: dict[str, Any]
    paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a short human-readable summary."""
        health = self.manifest.get("generation_health", {})
        return (
            f"run_id:  {self.run_id}\n"
            f"dir:     {self.run_dir}\n"
            f"itemset: {self.itemset_id} ({self.manifest.get('n_items', 0)} items)\n"
            f"rows:    {len(self.store)}\n"
            f"health:  {health.get('n_ok', 0)} ok / {health.get('n_error', 0)} error / "
            f"{health.get('n_truncated', 0)} truncated\n"
            f"spend:   ${self.manifest.get('spend_usd_total', 0):.2f}\n"
            f"results: {self.paths.get('results', '')}"
            + ("".join(f"\n\n!! {w}" for w in self.warnings) if self.warnings else "")
        )


def prepare_itemset(cfg: dict[str, Any], rebuild: bool = False) -> ItemSet:
    """Load the frozen item set, building and freezing it if needed.

    Building is deliberately a separate concern from evaluating: the set is frozen once
    and every checkpoint is measured against that same frozen set, which is what makes
    two runs comparable at all.

    Args:
        cfg: Resolved run config.
        rebuild: Force a rebuild even when a matching frozen set exists.

    Returns:
        The ItemSet.
    """
    itemset_cfg = dict(cfg.get("itemset") or {})
    root = Path(itemset_cfg.get("dir") or "output/internalization/itemsets")
    wanted = itemset_cfg.get("id")

    if not rebuild:
        try:
            return ItemSet.find(root, wanted)
        except FileNotFoundError:
            pass

    clauses = resolve_clause_set(cfg)
    gen_cfg = dict(itemset_cfg.get("generator") or {})
    llm = CachedLLM(
        inner=build_client(str(gen_cfg.get("provider", "openrouter"))),
        cache=CallCache(CacheConfig.from_config(cfg)),
        prices=PriceTable(cfg.get("pricing") or {}),
    )
    itemset = build_itemset(
        cfg,
        clauses,
        llm=llm,
        meta={
            "built_at_utc": timestamp(),
            "git_sha": git_sha(),
            "generator_model": gen_cfg.get("model", ""),
            "generation_spend_usd": llm.spend_usd,
        },
    )
    itemset.write(root)
    return itemset


def run_eval(
    cfg: dict[str, Any],
    itemset: ItemSet | None = None,
    rebuild_items: bool = False,
    run_id: str | None = None,
    max_items: int = 0,
) -> RunResult:
    """Evaluate one checkpoint end to end.

    Args:
        cfg: Resolved run config.
        itemset: A pre-loaded item set; loaded or built from the config when omitted.
        rebuild_items: Force an item-set rebuild.
        run_id: Override the generated run id.
        max_items: Cap on base items, for a quick pass. Pairing is preserved and the
            resulting itemset_id is suffixed, so a capped run can never be pooled with
            a full one by accident.

    Returns:
        The RunResult.
    """
    itemset = itemset or prepare_itemset(cfg, rebuild=rebuild_items)
    if max_items:
        itemset = itemset.subsample(max_items, seed=int(cfg.get("seed", 0)))
    clauses = resolve_clause_set(cfg)
    run = dict(cfg.get("run") or {})
    resolved_run_id = run_id or make_run_id(cfg)
    max_workers = int(cfg.get("max_workers", 16))

    target = TargetConfig.from_config(cfg)
    ctx = RunContext(
        run_id=resolved_run_id,
        recipe=str(run.get("recipe", "baseline")),
        checkpoint_step=int(run.get("checkpoint_step", 0)),
        model_id=target.model,
        itemset_id=itemset.itemset_id,
    )

    # One cache shared by both clients so a re-judge of an old checkpoint replays the
    # generation pass instead of paying for it again.
    cache = CallCache(CacheConfig.from_config(cfg))
    target_llm = build_target(cfg, cache=cache)
    judge_llm = build_judge_client(cfg, cache=cache)

    items = list(itemset)
    completions = generate(items, target_llm, target, max_workers=max_workers)

    rows = judge_all(
        items,
        completions,
        clauses,
        judge_llm,
        ctx,
        JudgeConfig.from_config(cfg),
        max_workers=max_workers,
    )
    store = ResultsStore(rows)
    # Reasoning retention lives in run_meta.json (generation_health), not the results store: it is
    # a generation-health gate, was constant in 476/476 rows, and its condition/family labels were
    # hardcoded wrong - silently misattributing a third of every groupby on `condition`.
    store.extend(capability_rows(cfg, ctx))

    run_dir = Path(cfg.get("output_dir") or "output/internalization") / "runs" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = store.write(run_dir / "results.jsonl")

    completions_path = run_dir / "completions.jsonl"
    with completions_path.open("w") as fh:
        for item in items:
            completion = completions.get(item.item_id)
            if completion is None:
                continue
            fh.write(json.dumps(completion.to_dict()) + "\n")

    health = generation_health(completions)
    warnings: list[str] = []
    limit = float(run.get("max_truncation_rate", 0.15))
    if health["truncation_rate"] > limit:
        warnings.append(
            f"TRUNCATION {health['truncation_rate']:.0%} of completions were cut off at "
            f"max_tokens={target.max_tokens} (limit {limit:.0%}). A truncated answer is not a "
            f"refusal, but a judge grades it as one - these results are NOT trustworthy. Raise "
            f"target.max_tokens above the model's trace length and re-run."
        )
    if health["error_rate"] > 0.02:
        warnings.append(
            f"ERRORS {health['error_rate']:.0%} of items failed to generate "
            f"({health['n_error']} of {health['n_items']}) and are excluded from every aggregate."
        )

    manifest = {
        "run_id": resolved_run_id,
        "recipe": ctx.recipe,
        "checkpoint_step": ctx.checkpoint_step,
        "model_id": ctx.model_id,
        "itemset_id": itemset.itemset_id,
        "clause_set_id": itemset.clause_set_id,
        "n_items": len(itemset),
        "n_rows": len(store),
        "item_counts": itemset.counts(),
        "generation_health": health,
        "warnings": warnings,
        "spend_usd_target": target_llm.spend_usd,
        "spend_usd_judge": judge_llm.spend_usd,
        "spend_usd_total": round(target_llm.spend_usd + judge_llm.spend_usd, 6),
        "cache": cache.stats(),
        "git_sha": git_sha(),
        "timestamp_utc": timestamp(),
        "config": cfg,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(manifest, indent=2))

    return RunResult(
        run_id=resolved_run_id,
        run_dir=run_dir,
        itemset_id=itemset.itemset_id,
        store=store,
        manifest=manifest,
        warnings=warnings,
        paths={
            "results": results_path,
            "completions": str(completions_path),
            "run_meta": str(run_dir / "run_meta.json"),
        },
    )
