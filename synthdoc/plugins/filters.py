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
from ..core.parsing import ParseError, parse_scores, render_document
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
        docs = sorted((d for d in corpus if d.ok), key=lambda d: d.doc_id)
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
