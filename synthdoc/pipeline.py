# ABOUTME: The stage runner. A linear sequence of stages, each writing a COMPLETE
# ABOUTME: corpus snapshot, so any stage can be re-run alone and any two can be diffed.

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from tqdm import tqdm

from . import config as config_mod
from .core.cache import Cache
from .core.embeddings import EmbeddingIndex, build_embedder
from .core.llm import CachedLLM, PriceTable, build_client
from .core.recipe import MixtureSampler, Recipe
from .core.specs import load_spec
from .core.types import Document, ScenarioSpec
from .plugins.chunkers import build_chunker
from .plugins.exporters import export_corpus
from .plugins.filters import FilterContext, build_filters
from .plugins.generators import GenerationContext, build_generator
from .plugins.groupers import GroupingContext, build_groupers
from .plugins.revisers import RevisionContext, build_reviser
from .snapshots import SnapshotConfig, SnapshotWriter, load_snapshot


class BudgetExceeded(RuntimeError):
    """Raised when a run passes its configured USD budget."""


@dataclass
class RunResult:
    """Everything a caller needs after a run.

    Attributes:
        run_id: Run identifier.
        run_dir: Local directory holding snapshots, manifest, exports, and report.
        config: The resolved config.
        stages: Stage names in order.
        counts: Per-stage document counts and verdict tallies.
        exports: Shard name -> written path.
        manifest: The run manifest.
        corpus: The final surviving documents.
    """

    run_id: str
    run_dir: Path
    config: dict[str, Any]
    stages: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    corpus: list[Document] = field(default_factory=list)


def _stage_names(n_revisions: int) -> list[str]:
    """Return the ordered stage names for a revision dose.

    Args:
        n_revisions: Length of the config's revision list.

    Returns:
        e.g. ["stage_00_generated", "stage_01_revised", "stage_02_filtered"].
    """
    names = ["stage_00_generated"]
    names += [f"stage_{i:02d}_revised" for i in range(1, n_revisions + 1)]
    names.append(f"stage_{n_revisions + 1:02d}_filtered")
    return names


def _map(fn: Callable[[Any], Any], items: Sequence[Any], workers: int, desc: str) -> list[Any]:
    """Thread-map preserving input order, with a progress bar.

    Args:
        fn: Callable applied to each item.
        items: Input sequence.
        workers: Thread pool size.
        desc: Progress description.

    Returns:
        Results in input order.
    """
    if not items:
        return []
    if workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))


def build_scenarios(cfg: dict[str, Any], n: int | None = None) -> tuple[list[ScenarioSpec], dict[str, Any]]:
    """Chunk the spec and sample the run's experimental conditions.

    Exposed separately from run_pipeline so a sweep runner can confirm that arms
    are paired before spending anything on generation.

    Args:
        cfg: Resolved config.
        n: Override for recipe.n.

    Returns:
        Tuple of (scenarios, diagnostics) where diagnostics records the spec sha,
        chunk count, and any grouper fallbacks.
    """
    spec_cfg = cfg.get("spec") or {}
    spec = load_spec(spec_cfg["id"], spec_cfg.get("path"))
    chunker = build_chunker(spec_cfg.get("chunker") or {})
    chunks = chunker.chunk(spec)
    if not chunks:
        raise ValueError(
            f"Chunker produced no chunks from {spec.path}. Check the granularity "
            "and the spec's heading structure."
        )

    recipe = Recipe.from_config(cfg.get("recipe") or {})
    context = GroupingContext()
    if recipe.grouping.get("semantic", 0.0) > 0:
        embedder = build_embedder(cfg.get("embedder"))
        context.index = EmbeddingIndex.build(
            ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            embedder=embedder,
            cache_dir=cfg.get("cache_dir"),
            tag=spec.spec_id,
        )

    groupers = build_groupers(recipe.strategies, chunks, recipe.grouping_params, context)
    sampler = MixtureSampler(groupers, seed=int(cfg.get("seed", 0)))
    scenarios = list(sampler.sample(chunks, recipe, n))

    diagnostics = {
        "spec_path": spec.path,
        "spec_sha": spec.sha,
        "n_chunks": len(chunks),
        "chunker": (spec_cfg.get("chunker") or {}).get("granularity", "bullet"),
        "grouper_stats": {name: dict(g.stats) for name, g in groupers.items() if g.stats},
        "n_scenarios": len(scenarios),
        "n_unique_scenario_hashes": len({s.scenario_hash for s in scenarios}),
    }
    return scenarios, diagnostics


def run_pipeline(
    cfg: dict[str, Any],
    n: int | None = None,
    run_id: str | None = None,
    progress: bool = True,
) -> RunResult:
    """Run the full generation pipeline.

    Args:
        cfg: Resolved config (from config.load_config).
        n: Override for recipe.n, e.g. for a smoke run.
        run_id: Override for the generated run id.
        progress: Show progress bars.

    Returns:
        A RunResult.

    Raises:
        BudgetExceeded: If cumulative cost passes cfg["budget_usd"].
    """
    started = time.time()
    cfg = config_mod.to_dict(cfg)
    run_id = run_id or config_mod.make_run_id(cfg)
    run_dir = Path(cfg["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    workers = int(cfg.get("max_workers", 16)) if progress else 1

    scenarios, diagnostics = build_scenarios(cfg, n)
    recipe = Recipe.from_config(cfg["recipe"])
    stages = _stage_names(len(cfg.get("revision") or []))

    cache = Cache(cfg["cache_dir"], enabled=bool(cfg.get("cache_enabled", True)))
    llm_cfg = dict(cfg.get("llm") or {})
    provider = llm_cfg.pop("provider", "openrouter")
    client = CachedLLM(
        inner=build_client(provider, **llm_cfg),
        cache=cache,
        prices=PriceTable(cfg.get("pricing")),
    )

    snap_cfg = SnapshotConfig(**(cfg.get("snapshots") or {}))
    writer = SnapshotWriter(
        run_dir=run_dir,
        cfg=snap_cfg,
        run_id=run_id,
        axis_names=recipe.axis_names,
        filter_fields=config_mod.filter_score_fields(cfg),
    )

    budget = float(cfg.get("budget_usd", 0) or 0)
    counts: dict[str, Any] = {}

    def spent(corpus: Sequence[Document]) -> float:
        """Cumulative USD across the corpus."""
        return round(sum(d.cost_usd_total for d in corpus), 4)

    def checkpoint(stage: str, corpus: list[Document]) -> None:
        """Snapshot a stage and enforce the budget."""
        writer.write(stage, corpus)
        cost = spent(corpus)
        counts[stage] = {
            "n": len(corpus),
            "n_ok": sum(1 for d in corpus if d.ok),
            "n_error": sum(1 for d in corpus if d.error),
            "mean_words": round(
                sum(len(d.text().split()) for d in corpus if d.ok) / max(1, sum(1 for d in corpus if d.ok)),
                1,
            ),
            "cost_usd": cost,
        }
        if budget and cost > budget:
            raise BudgetExceeded(
                f"Run cost ${cost:.2f} passed budget_usd=${budget:.2f} after {stage}. "
                f"Snapshots up to this stage are in {run_dir}. Raise budget_usd to continue; "
                "cached calls make the re-run free."
            )

    # --- stage 00: generation ---------------------------------------------------
    gen_cfg = dict(cfg.get("generation") or {})
    gen_ctx = GenerationContext(
        llm=client,
        model=gen_cfg.get("model", ""),
        params={
            k: v
            for k, v in gen_cfg.items()
            if k not in ("model", "template", "max_parse_retries")
        },
        template=gen_cfg.get("template", "v2"),
        run_id=run_id,
        stage_idx=0,
        stage_name=stages[0],
        max_parse_retries=int(gen_cfg.get("max_parse_retries", 1)),
    )

    corpus = _resume(run_dir, stages[0], cfg)
    if corpus is None:
        corpus = _map(
            lambda s: build_generator(s, gen_ctx).generate(s),
            scenarios,
            workers,
            f"{stages[0]} ({gen_ctx.model})",
        )
    checkpoint(stages[0], corpus)

    # --- stages 01..N: revision -------------------------------------------------
    for i, entry in enumerate(cfg.get("revision") or [], start=1):
        stage = stages[i]
        resumed = _resume(run_dir, stage, cfg)
        if resumed is not None:
            corpus = resumed
            checkpoint(stage, corpus)
            continue

        entry = dict(entry)
        rev_ctx = RevisionContext(
            llm=client,
            kind=entry.pop("kind"),
            model=entry.pop("model", gen_ctx.model),
            context=entry.pop("context", "fresh"),
            gen_template=gen_ctx.template,
            stage_idx=i,
            stage_name=stage,
            keep_on_failure=bool(entry.pop("keep_on_failure", True)),
            params=entry,
        )
        reviser = build_reviser(rev_ctx)
        advanced = [d.advanced(i, stage) for d in corpus]
        corpus = _map(reviser.revise, advanced, workers, f"{stage} ({rev_ctx.kind})")
        checkpoint(stage, corpus)

    # --- final stage: filtering -------------------------------------------------
    filter_stage = stages[-1]
    filter_idx = len(stages) - 1
    filter_ctx = FilterContext(
        llm=client,
        stage_idx=filter_idx,
        stage_name=filter_stage,
        cache_dir=cfg.get("cache_dir"),
    )
    filters = build_filters(cfg.get("filters") or [], filter_ctx)
    corpus = [d.advanced(filter_idx, filter_stage) for d in corpus]

    for f in filters:
        # A filter's model defaults to the generator's only if it declared none, so
        # a judge-model ablation is still an explicit config line.
        if getattr(f, "model", None) == "":
            f.model = gen_ctx.model
        f.prepare(corpus)
        results = _map(f.evaluate, corpus, workers, f"{filter_stage} ({f.name})")
        for doc, (scores, keep) in zip(corpus, results):
            doc.filter_scores.update(scores)
            if not keep and not doc.dropped_by:
                doc.dropped_by = f.name

    for doc in corpus:
        if not doc.ok:
            doc.filter_verdict = "drop"
            doc.dropped_by = doc.dropped_by or "error"
        else:
            doc.filter_verdict = "drop" if doc.dropped_by else "keep"

    checkpoint(filter_stage, corpus)
    counts[filter_stage]["n_keep"] = sum(1 for d in corpus if d.filter_verdict == "keep")
    counts[filter_stage]["dropped_by"] = _tally(d.dropped_by for d in corpus if d.dropped_by)

    kept = [d for d in corpus if d.filter_verdict == "keep"]

    # --- export, manifest, report ----------------------------------------------
    exports = export_corpus(kept, cfg.get("export") or {}, run_dir / "export")
    # The exports are the corpus handoff, so they belong in the dataset repo too -
    # otherwise "on HuggingFace" would only cover the stage snapshots.
    for path in exports.values():
        writer.push_file(path, f"export/{Path(path).name}")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_sha": config_mod.git_sha(),
        "timestamp_utc": config_mod.timestamp(),
        "elapsed_s": round(time.time() - started, 1),
        "seed": cfg.get("seed", 0),
        "config": cfg,
        "spec": diagnostics,
        "stages": stages,
        "counts": counts,
        "cache": cache.stats(),
        "cost_usd_total": spent(corpus),
        "unpriced_models": sorted(client.prices.unpriced),
        "exports": exports,
        "thresholds": {
            entry.get("kind"): {k: v for k, v in entry.items() if k != "kind"}
            for entry in (cfg.get("filters") or [])
        },
        "filter_summaries": {f.name: f.summary for f in filters if f.summary},
        "agreement": {
            f.name: f.agreement() for f in filters if hasattr(f, "agreement")
        },
        "hf_repo": writer.repo_id if snap_cfg.backend == "huggingface" else None,
    }

    result = RunResult(
        run_id=run_id,
        run_dir=run_dir,
        config=cfg,
        stages=stages,
        counts=counts,
        exports=exports,
        manifest=manifest,
        corpus=kept,
    )

    if (cfg.get("report") or {}).get("enabled", True):
        from .report import coverage_report

        report_paths = coverage_report(corpus, cfg, run_dir, manifest)
        manifest["report"] = report_paths
        for name, path in report_paths.items():
            if path:
                writer.push_file(path, Path(path).name)
        del name

    writer.write_manifest(manifest)
    push_errors = writer.finish()
    if push_errors:
        manifest["push_errors"] = push_errors
        writer.write_manifest(manifest)

    # Catalogue the finished corpus so `cli corpora` can find it later.
    from .corpora import register as register_corpus

    register_corpus(result)

    # Optionally leave HuggingFace as the only copy. Runs last, and only once every
    # push has been joined and verified. The JSONL sidecars are never uploaded (they
    # duplicate the parquet plus lineage), so they are removed explicitly.
    if snap_cfg.cleanup_local and snap_cfg.backend == "huggingface" and not push_errors:
        sidecars = [run_dir / f"{stage}.jsonl" for stage in stages]
        manifest["local_files_removed"] = writer.cleanup(extra=sidecars)
        writer.write_manifest(manifest)

    return result


def _tally(values) -> dict[str, int]:
    """Count occurrences of each value."""
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _resume(run_dir: Path, stage: str, cfg: dict[str, Any]) -> list[Document] | None:
    """Load an existing stage snapshot when resuming.

    Args:
        run_dir: Run directory.
        stage: Stage name.
        cfg: Resolved config.

    Returns:
        The snapshot's documents, or None if not resuming or not present.
    """
    if not cfg.get("resume", True):
        return None
    path = run_dir / f"{stage}.jsonl"
    if not path.exists():
        return None
    return load_snapshot(path)
