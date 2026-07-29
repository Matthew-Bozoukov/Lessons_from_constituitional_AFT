# ABOUTME: Filters score documents and vote keep/drop. Dropped documents are RETAINED
# ABOUTME: with a verdict, so the filter's own effect on the corpus stays inspectable.

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..control import loader
from ..core.embeddings import build_embedder
from ..core.hashing import stable_hash
from ..core.llm import CachedLLM
from ..core.parsing import ParseError, extract_json, parse_scores, render_document
from ..core.prompting import scenario_vars
from ..core.registry import register, resolve
from ..core.types import Document, StageRecord


@dataclass
class FilterContext:
    """Shared services a filter may need.

    Attributes:
        llm: Cache-wrapped LLM client (None for filters that need no model).
        stage_idx: Stage index for provenance and cache keying.
        stage_name: Stage name for provenance.
        cache_dir: Where embedders may persist indices.
    """

    llm: CachedLLM | None = None
    stage_idx: int = 99
    stage_name: str = "stage_99_filtered"
    cache_dir: str | None = None


class BaseFilter:
    """Filter protocol: declare score fields, prepare over the corpus, evaluate per doc."""

    name = "base"

    def __init__(self, ctx: FilterContext, **params: Any) -> None:
        """Initialize with the shared context and filter-specific params."""
        self.ctx = ctx
        self.params = params
        self.summary: dict[str, Any] = {}

    def score_fields(self) -> list[str]:  # pragma: no cover - abstract
        """Return the filter_scores keys this filter writes."""
        raise NotImplementedError

    def prepare(self, corpus: Sequence[Document]) -> None:
        """Optional corpus-level pass run before evaluate()."""

    def evaluate(self, doc: Document) -> tuple[dict[str, float], bool]:  # pragma: no cover
        """Return (scores, keep) for one document."""
        raise NotImplementedError

    def score(self, doc: Document) -> float:
        """Return the filter's primary score for one document."""
        scores, _ = self.evaluate(doc)
        return next(iter(scores.values()), 0.0)


@register("filter", "length")
class LengthFilter(BaseFilter):
    """Drops documents outside a word-count band. Free, and catches truncation.

    Params:
        min_words: Minimum total words across turns.
        max_words: Maximum total words across turns.
    """

    name = "length"

    def __init__(self, ctx: FilterContext, min_words: int = 80,
                 max_words: int = 6000, **_: Any) -> None:
        """Initialize with the word-count band."""
        super().__init__(ctx)
        self.min_words = int(min_words)
        self.max_words = int(max_words)

    def score_fields(self) -> list[str]:
        """Return ["length_words"]."""
        return ["length_words"]

    def evaluate(self, doc: Document) -> tuple[dict[str, float], bool]:
        """Count words and compare against the band."""
        n = float(len(doc.text().split()))
        return {"length_words": n}, self.min_words <= n <= self.max_words


@register("filter", "embedding_dedup")
class EmbeddingDedupFilter(BaseFilter):
    """Greedy near-duplicate removal over the whole corpus.

    Documents are visited in a deterministic order (doc_id) so the survivor set is
    reproducible and identical across sweep arms that produced identical corpora.
    Similarity is computed with a blocked matrix product rather than pairwise loops,
    which keeps a 20k-document corpus to seconds.

    Params:
        threshold: Cosine similarity at or above which a document is a duplicate.
        embedder: Embedder config block, defaults to the offline hashing embedder.
        block: Rows per similarity block.
    """

    name = "embedding_dedup"

    def __init__(self, ctx: FilterContext, threshold: float = 0.87,
                 embedder: dict[str, Any] | None = None, block: int = 512, **_: Any) -> None:
        """Initialize with the duplicate threshold and embedder config."""
        super().__init__(ctx)
        self.threshold = float(threshold)
        self.embedder_cfg = dict(embedder or {"name": "hashing", "dim": 512})
        self.block = int(block)
        self._max_sim: dict[str, float] = {}
        self._keep: dict[str, bool] = {}
        self._dup_of: dict[str, str] = {}

    def score_fields(self) -> list[str]:
        """Return ["dedup_max_sim"]."""
        return ["dedup_max_sim"]

    def prepare(self, corpus: Sequence[Document]) -> None:
        """Embed the corpus and mark duplicates greedily."""
        # Ordered by scenario_hash, not doc_id: doc_id embeds the run_id, so ordering
        # by it would make the greedy survivor set depend on which run produced the
        # corpus - and two sweep arms with identical documents could dedup differently.
        docs = sorted((d for d in corpus if d.ok), key=lambda d: d.scenario.scenario_hash)
        if not docs:
            self.summary = {"n_compared": 0, "n_duplicates": 0}
            return

        embedder = build_embedder(self.embedder_cfg)
        vectors = embedder.embed([d.text() for d in docs])
        n = len(docs)
        kept_mask = np.zeros(n, dtype=bool)

        for start in range(0, n, self.block):
            end = min(start + self.block, n)
            sims = vectors[start:end] @ vectors[:end].T  # only look backwards
            for local, i in enumerate(range(start, end)):
                row = sims[local, :i]
                if i == 0:
                    max_sim, arg = 0.0, -1
                else:
                    masked = np.where(kept_mask[:i], row, -1.0)
                    arg = int(masked.argmax())
                    max_sim = float(masked[arg])
                doc_id = docs[i].doc_id
                self._max_sim[doc_id] = max(0.0, max_sim)
                is_dup = max_sim >= self.threshold
                kept_mask[i] = not is_dup
                self._keep[doc_id] = not is_dup
                if is_dup and arg >= 0:
                    self._dup_of[doc_id] = docs[arg].doc_id

        self.summary = {
            "n_compared": n,
            "n_duplicates": int((~kept_mask).sum()),
            "threshold": self.threshold,
            "embedder": self.embedder_cfg,
        }

    def evaluate(self, doc: Document) -> tuple[dict[str, float], bool]:
        """Return the precomputed max similarity and keep decision."""
        return (
            {"dedup_max_sim": round(self._max_sim.get(doc.doc_id, 0.0), 4)},
            self._keep.get(doc.doc_id, True),
        )

    def duplicate_of(self, doc_id: str) -> str | None:
        """Return the doc_id this document duplicates, if any."""
        return self._dup_of.get(doc_id)


@register("filter", "autorater")
class AutoraterFilter(BaseFilter):
    """LLM rating against a rubric from control/prompts/rubrics.yaml.

    Runs n_raters independent judgements per document and keeps the document when
    the mean overall score clears min_score. Inter-rater agreement is reported so a
    rubric that raters cannot apply consistently is visible rather than assumed good.

    Params:
        rubric: Rubric version.
        model: Rater model.
        n_raters: Independent ratings per document.
        min_score: Keep threshold on the mean overall score.
        report_agreement: Compute corpus-level agreement statistics.
        temperature: Rater sampling temperature.
        max_tokens: Rater output cap.
    """

    name = "autorater"

    def __init__(
        self,
        ctx: FilterContext,
        rubric: str = "v4",
        model: str = "",
        n_raters: int = 1,
        min_score: float = 3.0,
        report_agreement: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 700,
        **_: Any,
    ) -> None:
        """Initialize the rater.

        Raises:
            ValueError: If no LLM client is available.
        """
        super().__init__(ctx)
        if ctx.llm is None:
            raise ValueError("autorater requires an LLM client")
        self.rubric_name = rubric
        self.rubric = loader.entry("rubrics", rubric)
        self.criteria: list[str] = list(self.rubric.get("criteria") or ["overall"])
        self.scale = int(self.rubric.get("scale", 5))
        self.model = model
        self.n_raters = max(1, int(n_raters))
        self.min_score = float(min_score)
        self.report_agreement = bool(report_agreement)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._overalls: list[list[float]] = []

    def score_fields(self) -> list[str]:
        """Return the rubric's criterion columns plus overall and spread."""
        return (
            ["autorater_overall", "autorater_std"]
            + [f"autorater_{c}" for c in self.criteria]
        )

    def _messages(self, doc: Document) -> list[dict]:
        """Render the rating prompt for one document."""
        variables = scenario_vars(doc.scenario, document=render_document(doc.turns))
        return [
            {"role": "system", "content": loader.render(self.rubric["system"], **variables)},
            {"role": "user", "content": loader.render(self.rubric["user"], **variables)},
        ]

    def evaluate(self, doc: Document) -> tuple[dict[str, float], bool]:
        """Rate a document n_raters times and decide keep/drop."""
        if not doc.ok:
            return {}, False
        messages = self._messages(doc)
        input_hash = stable_hash(render_document(doc.turns))

        overalls: list[float] = []
        per_criterion: dict[str, list[float]] = {c: [] for c in self.criteria}
        for rater in range(self.n_raters):
            # Distinct params per rater keep the cache keys distinct, so raters are
            # genuinely independent rather than one cached answer counted n times.
            params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": doc.scenario.seed + rater,
            }
            try:
                resp, prompt_hash = self.ctx.llm.call(
                    stage_idx=self.ctx.stage_idx,
                    input_hash=f"{input_hash}:{self.rubric_name}:{rater}",
                    model=self.model,
                    messages=messages,
                    params=params,
                )
            except Exception:
                continue
            doc.lineage.append(
                StageRecord(
                    stage_idx=self.ctx.stage_idx,
                    stage_name=self.ctx.stage_name,
                    kind=f"autorater:{self.rubric_name}:{rater}",
                    model=self.model,
                    prompt_hash=prompt_hash,
                    params=stable_hash(params, 12),
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    cost_usd=self.ctx.llm.cost(resp),
                    cached=resp.cached,
                )
            )
            try:
                scores, overall, _ = parse_scores(resp.content, self.criteria, self.scale)
            except ParseError:
                continue
            overalls.append(overall)
            for c, v in scores.items():
                per_criterion[c].append(v)

        if not overalls:
            return {"autorater_overall": 0.0, "autorater_std": 0.0}, False

        mean_overall = sum(overalls) / len(overalls)
        std = statistics.pstdev(overalls) if len(overalls) > 1 else 0.0
        if self.report_agreement:
            self._overalls.append(overalls)

        out: dict[str, float] = {
            "autorater_overall": round(mean_overall, 3),
            "autorater_std": round(std, 3),
        }
        for c, values in per_criterion.items():
            if values:
                out[f"autorater_{c}"] = round(sum(values) / len(values), 3)
        return out, mean_overall >= self.min_score

    def agreement(self) -> dict[str, Any]:
        """Return inter-rater agreement statistics for the run manifest.

        Returns:
            Mean per-document rating spread, exact-match rate on rounded scores, and
            the fraction of documents where raters straddled the keep threshold.
        """
        rated = [o for o in self._overalls if len(o) > 1]
        if not rated:
            return {"n_raters": self.n_raters, "note": "agreement needs n_raters > 1"}
        spreads = [max(o) - min(o) for o in rated]
        exact = [1.0 if len(set(round(v) for v in o)) == 1 else 0.0 for o in rated]
        straddle = [
            1.0 if (min(o) < self.min_score <= max(o)) else 0.0 for o in rated
        ]
        return {
            "n_raters": self.n_raters,
            "n_documents": len(rated),
            "mean_spread": round(sum(spreads) / len(spreads), 3),
            "exact_agreement_rate": round(sum(exact) / len(exact), 3),
            "threshold_straddle_rate": round(sum(straddle) / len(straddle), 3),
            "min_score": self.min_score,
            "rubric": self.rubric_name,
        }


@register("filter", "pattern_scan")
class PatternScanFilter(BaseFilter):
    """GDM's scan-cluster-autorate pass.

    Three passes: SCAN batches of documents for patterns that recur across them,
    CLUSTER by keeping only patterns that independent scans both found, then AUTORATE
    every document against the surviving list.

    The discovery step is the point. A synthetic corpus's failure modes are properties
    of that corpus - a fixed rubric written in advance cannot name the tic this
    particular generator fell into. Documents are dropped for matching too many
    patterns, and the pattern list itself is written to the manifest, which is often
    more useful than the filtering.

    Params:
        model: Model used for scanning and matching.
        discover: Run the scan and cluster passes. False uses only seed_patterns.
        n_batches: Batches to scan.
        batch_size: Documents per scan batch.
        min_scans: Batches a pattern must appear in to survive clustering.
        min_docs_per_batch: Documents within a batch a pattern must cover to count.
        max_patterns: Drop a document matching more than this many patterns.
        mode: broad | strict detection threshold.
        use_seed_patterns: Include the known anti-patterns from patterns.yaml.
        max_chars: Truncate each document to this many characters when scanning.
    """

    name = "pattern_scan"

    def __init__(
        self,
        ctx: FilterContext,
        model: str = "",
        discover: bool = True,
        n_batches: int = 8,
        batch_size: int = 12,
        min_scans: int = 2,
        min_docs_per_batch: int = 3,
        max_patterns: int = 2,
        mode: str = "strict",
        use_seed_patterns: bool = True,
        max_chars: int = 4000,
        **_: Any,
    ) -> None:
        """Initialize the pattern scanner.

        Raises:
            ValueError: If no LLM client is available or mode is invalid.
        """
        super().__init__(ctx)
        if ctx.llm is None:
            raise ValueError("pattern_scan requires an LLM client")
        if mode not in ("broad", "strict"):
            raise ValueError(f"pattern_scan mode must be broad or strict, got {mode!r}")
        self.model = model
        self.discover = bool(discover)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.min_scans = int(min_scans)
        self.min_docs_per_batch = int(min_docs_per_batch)
        self.max_patterns = int(max_patterns)
        self.mode = mode
        self.use_seed_patterns = bool(use_seed_patterns)
        self.max_chars = int(max_chars)
        self.patterns: list[dict[str, str]] = []

    def score_fields(self) -> list[str]:
        """Return the pattern-match columns."""
        return ["pattern_matches", "pattern_match_rate"]

    def prepare(self, corpus: Sequence[Document]) -> None:
        """Scan batches, cluster the findings, and fix the pattern list."""
        pack = loader.load_pack("patterns")
        patterns: dict[str, dict[str, Any]] = {}

        if self.use_seed_patterns:
            for entry in pack.get("seed_patterns") or []:
                patterns[entry["name"]] = {
                    "name": entry["name"],
                    "description": " ".join(str(entry["description"]).split()),
                    "source": "seed",
                    "n_scans": self.min_scans,  # seeds bypass the clustering threshold
                }

        docs = [d for d in corpus if d.ok]
        if self.discover and len(docs) >= self.min_docs_per_batch:
            discovered = self._scan(docs, pack["scan"])
            for name, entry in discovered.items():
                if entry["n_scans"] >= self.min_scans and name not in patterns:
                    patterns[name] = entry

        self.patterns = [
            {"name": p["name"], "description": p["description"]}
            for p in sorted(patterns.values(), key=lambda p: p["name"])
        ]
        self.summary = {
            "n_patterns": len(self.patterns),
            "discovered": sum(1 for p in patterns.values() if p.get("source") != "seed"),
            "seeded": sum(1 for p in patterns.values() if p.get("source") == "seed"),
            "mode": self.mode,
            "max_patterns": self.max_patterns,
            "patterns": self.patterns,
        }

    def _scan(self, docs: Sequence[Document], template: dict) -> dict[str, dict[str, Any]]:
        """Run the scan pass over batches and tally how many batches saw each pattern."""
        # Batched over a scenario_hash sort, and keyed on the batch's CONTENT rather
        # than its doc_ids: doc_id embeds the run_id, so keying on it would miss the
        # cache on every re-run and every sweep arm even when the documents are byte
        # identical.
        ordered = sorted(docs, key=lambda d: d.scenario.scenario_hash)
        batches: list[list[Document]] = []
        for i in range(0, len(ordered), self.batch_size):
            batches.append(ordered[i : i + self.batch_size])
            if len(batches) >= self.n_batches:
                break

        found: dict[str, dict[str, Any]] = {}
        for batch in batches:
            rendered = [render_document(d.turns)[: self.max_chars] for d in batch]
            messages = [
                {"role": "system", "content": loader.render(template["system"])},
                {"role": "user", "content": loader.render(template["user"], documents=rendered)},
            ]
            params = {"temperature": 0.0, "max_tokens": 1500}
            try:
                resp, prompt_hash = self.ctx.llm.call(
                    stage_idx=self.ctx.stage_idx,
                    input_hash=stable_hash(rendered),
                    model=self.model,
                    messages=messages,
                    params=params,
                    scope="filter",
                )
                payload = extract_json(resp.content)
            except (ParseError, Exception):  # noqa: B014 - any failure skips this batch
                continue
            if not isinstance(payload, dict):
                continue
            for item in payload.get("patterns") or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                if int(item.get("n_documents", 0) or 0) < self.min_docs_per_batch:
                    continue
                name = str(item["name"]).strip().lower().replace(" ", "_")
                entry = found.setdefault(
                    name,
                    {
                        "name": name,
                        "description": " ".join(str(item.get("description", "")).split()),
                        "source": "discovered",
                        "n_scans": 0,
                    },
                )
                entry["n_scans"] += 1
                del prompt_hash
        return found

    def evaluate(self, doc: Document) -> tuple[dict[str, float], bool]:
        """Autorate one document against the pattern list."""
        if not doc.ok or not self.patterns:
            return {"pattern_matches": 0.0, "pattern_match_rate": 0.0}, doc.ok

        pack = loader.load_pack("patterns")
        template = pack["match"]
        variables = {
            "patterns": self.patterns,
            "document": render_document(doc.turns)[: self.max_chars],
            "mode": self.mode,
        }
        messages = [
            {"role": "system", "content": loader.render(template["system"], **variables)},
            {"role": "user", "content": loader.render(template["user"], **variables)},
        ]
        params = {"temperature": 0.0, "max_tokens": 400}
        try:
            resp, prompt_hash = self.ctx.llm.call(
                stage_idx=self.ctx.stage_idx,
                input_hash=stable_hash(
                    {"doc": render_document(doc.turns), "patterns": self.patterns, "mode": self.mode}
                ),
                model=self.model,
                messages=messages,
                params=params,
                scope="filter",
            )
        except Exception:
            return {"pattern_matches": 0.0, "pattern_match_rate": 0.0}, True

        doc.lineage.append(
            StageRecord(
                stage_idx=self.ctx.stage_idx,
                stage_name=self.ctx.stage_name,
                kind=f"pattern_scan:{self.mode}",
                model=self.model,
                prompt_hash=prompt_hash,
                params=stable_hash(params, 12),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=self.ctx.llm.cost(resp),
                cached=resp.cached,
            )
        )

        known = {p["name"] for p in self.patterns}
        try:
            payload = extract_json(resp.content)
            matched = [str(m) for m in (payload.get("matched") or [])] if isinstance(payload, dict) else []
        except ParseError:
            matched = []
        matched = [m for m in matched if m in known]

        n = float(len(matched))
        return (
            {"pattern_matches": n, "pattern_match_rate": round(n / max(1, len(known)), 4)},
            n <= self.max_patterns,
        )


@dataclass
class FilterSpec:
    """One entry from the config's `filters:` list.

    Attributes:
        kind: Registered filter name.
        params: Filter-specific params.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


def build_filters(specs: Sequence[dict[str, Any]], ctx: FilterContext) -> list[BaseFilter]:
    """Instantiate the configured filters in order.

    Args:
        specs: The config's `filters:` list.
        ctx: Shared filter context.

    Returns:
        Filter instances.
    """
    out: list[BaseFilter] = []
    for spec in specs or []:
        params = dict(spec)
        kind = params.pop("kind")
        out.append(resolve("filter", kind)(ctx, **params))
    return out
