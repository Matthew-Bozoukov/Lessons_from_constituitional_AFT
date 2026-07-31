# ABOUTME: Cross-checks the judge used in a run against a stronger reference judge on a sample.
# ABOUTME: Buys evidence that a cheap judge is safe, for roughly the price of a coffee.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .core.hashing import stream_rng
from .core.llm import CachedLLM, map_threaded
from .core.registry import resolve
from .core.stats import agreement
from .core.store import ResultsStore
from .core.types import ClauseSet, Completion, Item
from .items.itemset import ItemSet
from .judges.base import JudgeConfig


class JudgeCheckError(ValueError):
    """Raised when a run cannot support a judge cross-check."""


@dataclass
class AgreementReport:
    """Agreement between the run's judge and a stronger reference judge.

    This is not a correctness proof - the reference judge is not ground truth. What it
    establishes is narrower and still useful: that swapping in the cheap judge did not
    change the pass/fail decision often enough to move a recipe comparison.

    Attributes:
        run_judge: Model that produced the run's verdicts.
        reference_judge: Model re-graded against.
        per_axis: Axis -> {"raw", "kappa", "n"}.
        overall: The same figures pooled.
        disagreements: Sampled rows where the two judges differed.
        n_errors: Reference-judge calls that failed and were excluded.
    """

    run_judge: str
    reference_judge: str
    per_axis: dict[str, dict[str, float]]
    overall: dict[str, float]
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    n_errors: int = 0

    def verdict(self, min_raw: float = 0.85, min_kappa: float = 0.6) -> str:
        """Return "PASS" or "REVIEW" against the given thresholds."""
        ok = self.overall.get("raw", 0.0) >= min_raw and self.overall.get("kappa", 0.0) >= min_kappa
        return "PASS" if ok else "REVIEW"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "run_judge": self.run_judge,
            "reference_judge": self.reference_judge,
            "overall": self.overall,
            "per_axis": self.per_axis,
            "n_errors": self.n_errors,
            "verdict": self.verdict(),
            "disagreements": self.disagreements,
        }

    def render(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"run judge:       {self.run_judge}",
            f"reference judge: {self.reference_judge}",
            "",
            f"pooled agreement raw {self.overall.get('raw', 0):.3f} · "
            f"kappa {self.overall.get('kappa', 0):+.3f} · "
            f"n={int(self.overall.get('n', 0))} -> {self.verdict()}",
            "",
        ]
        for axis in sorted(self.per_axis):
            s = self.per_axis[axis]
            lines.append(f"  {axis:24s} raw {s['raw']:.3f}  kappa {s['kappa']:+.3f}  n={int(s['n'])}")
        if self.n_errors:
            lines.append(f"\n  {self.n_errors} reference calls errored and were excluded")
        lines += [
            "",
            "Agreement on the binarised pass/fail decision - the decision every proportion in",
            "the report is built from. The reference judge is not ground truth; this measures",
            "whether the cheaper judge changes conclusions, not whether either is correct.",
        ]
        return "\n".join(lines)


def load_run(run_dir: Path | str) -> tuple[dict[str, Any], ResultsStore, dict[str, Completion]]:
    """Load a completed run's manifest, results, and completions.

    Args:
        run_dir: The run directory.

    Returns:
        Tuple of (manifest, results store, item_id -> Completion).

    Raises:
        JudgeCheckError: If the directory is not a completed run.
    """
    root = Path(run_dir)
    for name in ("run_meta.json", "results.jsonl", "completions.jsonl"):
        if not (root / name).exists():
            raise JudgeCheckError(f"{root} is not a completed run: {name} is missing")
    manifest = json.loads((root / "run_meta.json").read_text())
    store = ResultsStore.load(root / "results.jsonl")
    completions = {
        c["item_id"]: Completion.from_dict(c)
        for c in (
            json.loads(line)
            for line in (root / "completions.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    return manifest, store, completions


def check_judge(
    run_dir: Path | str,
    itemset: ItemSet,
    clauses: ClauseSet,
    llm: CachedLLM,
    reference: JudgeConfig,
    n: int = 120,
    seed: int = 0,
    axes: Sequence[str] | None = None,
    max_workers: int = 8,
) -> AgreementReport:
    """Re-grade a sample of a run's rows with a stronger judge and report agreement.

    Sampling is stratified by axis so no axis is checked on two rows, and it reuses the
    completions already on disk - only the reference judge is paid for.

    Args:
        run_dir: A completed run directory.
        itemset: The item set the run used.
        clauses: The clause set the rubrics grade against.
        llm: Cached client for the reference judge.
        reference: Reference judge settings.
        n: Approximate number of rows to re-grade.
        seed: Sampling seed.
        axes: Restrict to these axes; None uses every axis present.
        max_workers: Concurrency.

    Returns:
        The AgreementReport.

    Raises:
        JudgeCheckError: If the run has no gradeable rows, or every reference call failed.
    """
    _, store, completions = load_run(run_dir)
    by_id = itemset.by_id()

    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in store.rows:
        if row.get("error"):
            continue
        axis = str(row["axis"])
        # Derived rows (reasoning retention, ingested capability) have no judge behind them.
        if row.get("judge_model") in ("derived", "external"):
            continue
        if axes and axis not in axes:
            continue
        if row["item_id"] not in by_id or not (completions.get(row["item_id"]) or Completion("", "")).ok:
            continue
        candidates.setdefault(axis, []).append(row)
    if not candidates:
        raise JudgeCheckError(f"No gradeable rows in {run_dir}")

    per_axis_n = max(1, n // len(candidates))
    sampled: list[dict[str, Any]] = []
    for idx, axis in enumerate(sorted(candidates)):
        pool = sorted(candidates[axis], key=lambda r: str(r["item_id"]))
        rng = stream_rng(seed, idx, "judge_check")
        rng.shuffle(pool)
        sampled.extend(pool[:per_axis_n])

    judges = {axis: resolve("judge", axis)(clauses) for axis in candidates}

    def regrade(row: dict[str, Any]):
        """Re-grade one row with the reference judge."""
        item: Item = by_id[row["item_id"]]
        return judges[str(row["axis"])](
            item, completions[row["item_id"]], llm, reference
        )

    verdicts = map_threaded(regrade, sampled, max_workers=max_workers, desc="cross-check")

    buckets: dict[str, tuple[list[int], list[int]]] = {}
    disagreements: list[dict[str, Any]] = []
    n_errors = 0
    for row, verdict in zip(sampled, verdicts):
        if verdict.error:
            n_errors += 1
            continue
        axis = str(row["axis"])
        run_label = 1 if row["passed"] else 0
        ref_label = 1 if verdict.passed else 0
        run_labels, ref_labels = buckets.setdefault(axis, ([], []))
        run_labels.append(run_label)
        ref_labels.append(ref_label)
        if run_label != ref_label:
            disagreements.append(
                {
                    "item_id": row["item_id"],
                    "axis": axis,
                    "clause_id": row["clause_id"],
                    "condition": row["condition"],
                    "run_raw": row["raw_score"],
                    "reference_raw": verdict.raw_score,
                    "run_rationale": str(row.get("judge_rationale", ""))[:220],
                    "reference_rationale": verdict.rationale[:220],
                }
            )

    if not buckets:
        raise JudgeCheckError("Every reference-judge call errored; cannot compute agreement")

    scores = {axis: agreement(a, b) for axis, (a, b) in buckets.items()}
    pooled_run = [x for a, _ in buckets.values() for x in a]
    pooled_ref = [x for _, b in buckets.values() for x in b]
    run_judges = sorted({str(r.get("judge_model", "")) for r in sampled})
    return AgreementReport(
        run_judge=", ".join(j for j in run_judges if j),
        reference_judge=reference.model,
        per_axis=scores,
        overall=agreement(pooled_run, pooled_ref),
        disagreements=disagreements,
        n_errors=n_errors,
    )
