# ABOUTME: The paper's four quantities — counterfactual importance, resilience,
# ABOUTME: counterfactual++ importance and the cumulative effect curve — over ODCV outcomes.

"""Causal quantities over resampled outcome distributions.

Everything here consumes `Branch` records (one resample: what was generated, what outcome
it reached) and produces a number per branch point. Nothing here calls a model; the
sampler does that. Keeping them apart is what lets the metrics be unit-tested on
fixtures and lets a cached resampling run be re-analysed under a different threshold
without spending a GPU-hour again.

The four:

  importance(S_i)     KL( p(outcome | S_i resampled away) || p(outcome | S_i present) ).
                      "If the model had said something else here, how different would
                      the ending be?" Its weakness is the paper's own finding: the model
                      often just says S_i again two sentences later, so this reads
                      near-uniform across categories and settles nothing.

  resilience(S_i)     How many rounds of resampling it takes before the content STOPS
                      coming back. Algorithm 1 in the paper. Low resilience means the
                      model abandons the idea the moment it is nudged — the signature of
                      a rationalisation rather than a driver.

  importance++(S_i)   The same KL, but restricted to rollouts where the content never
                      reappeared anywhere downstream. This is the one that separates
                      categories, because it compares "the idea is gone" against "the
                      idea is present" rather than "this sentence is worded differently".

  effect_curve        p(outcome | first i chunks) as i runs over the trace. Not an
                      ablation: it reads out where along the trace the decision actually
                      gets made, and it is the cheapest of the four (one resample set per
                      prefix, no matching, no iteration).

ODCV outcomes are the judges' 0-5 severity, so distributions here are categorical over
severity levels, with binary violation (severity >= 3) available as a coarser read.

One property of the KL is worth stating because it surprises people: it is CATEGORICAL,
so it ignores the ordering of the severity classes. Moving all the mass from severity 0
to severity 1 scores exactly the same as moving it to severity 5. That matches the paper,
whose outcome classes are also unordered, and it is why every `Importance` record also
carries `delta_violation` — the KL says how far the distribution moved, and only the
signed violation rate says which way.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from src.eval.stats import wilson

from scratch.thought_branches.embed import DEFAULT_MODEL, best_match, encode, cosine
from scratch.thought_branches.trajectory import VIOLATION_THRESHOLD

# Severity classes an ODCV rollout can land in. Judges may return .5 values (an even
# number of judges averages the middle two), so the categorical read bins to integers.
SEVERITY_CLASSES = (0, 1, 2, 3, 4, 5)

# Laplace pseudo-count. Resample sets are small (the paper uses 100 per sentence, we will
# often afford fewer), and an unsmoothed KL is infinite the moment one distribution puts
# zero mass where the other puts some. Smoothing keeps the metric finite and comparable
# across branch points with different sample counts.
ALPHA = 0.5


@dataclass
class Branch:
    """One resampled continuation from one branch point.

    Attributes:
        branch_id: The `BranchPoint.branch_id` this came from.
        sample: 0-based index of this resample within the branch's set.
        replacement: The text generated in place of the ablated chunk — the sentence or
            step the model produced instead. Empty if the branch produced nothing.
        downstream: Every sentence of the continuation AFTER the replacement, used for
            the counterfactual++ "did the idea come back" test.
        severity: The judged ODCV severity of the resulting trajectory, or None when the
            branch was not carried to an outcome (a frozen-environment branch that only
            read out the local action, for example).
        meta: Anything the sampler wants to carry through (token counts, finish reason).
    """

    branch_id: str
    sample: int
    replacement: str
    downstream: list[str] = field(default_factory=list)
    severity: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_violation(self) -> bool | None:
        return None if self.severity is None else self.severity >= VIOLATION_THRESHOLD


# -- distributions ---------------------------------------------------------------


def severity_dist(
    severities: Iterable[float | None], alpha: float = ALPHA
) -> np.ndarray:
    """Smoothed categorical distribution over severity classes 0-5.

    Args:
        severities: Judged severities; None entries are dropped as abstentions.
        alpha: Laplace pseudo-count per class.

    Returns:
        A length-6 probability vector summing to 1. With no observations at all this is
        the uniform distribution, which correctly carries "we know nothing here" into a
        KL of ~0 rather than a spurious spike.
    """
    vals = [s for s in severities if s is not None]
    counts = Counter(int(round(v)) for v in vals)
    raw = np.array(
        [counts.get(c, 0) + alpha for c in SEVERITY_CLASSES], dtype=np.float64
    )
    return raw / raw.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats, for two smoothed distributions of equal length.

    Args:
        p: The resampled ("intervened") distribution — the paper's first argument.
        q: The observed ("present") distribution.

    Returns:
        The divergence; 0.0 when the two agree.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(p * np.log(np.clip(p, 1e-12, None) / np.clip(q, 1e-12, None))))


def violation_rate(
    severities: Sequence[float | None],
) -> tuple[float, tuple[float, float], int]:
    """Violation rate with a Wilson interval.

    Args:
        severities: Judged severities; None entries are dropped.

    Returns:
        (rate in [0,1], (lo, hi) 95% Wilson interval, n). Rate is 0.0 for n == 0.
    """
    vals = [s for s in severities if s is not None]
    n = len(vals)
    if n == 0:
        return 0.0, (0.0, 0.0), 0
    k = sum(1 for v in vals if v >= VIOLATION_THRESHOLD)
    return k / n, wilson(k, n), n


# -- 1. counterfactual importance -------------------------------------------------


@dataclass
class Importance:
    """A branch point's counterfactual importance.

    Attributes:
        branch_id: Which branch point.
        kl: KL(resampled || base) over severity classes.
        n_used: Resamples that passed the dissimilarity filter.
        n_total: Resamples generated.
        base_violation: Violation rate of the original trajectory's outcome set.
        resampled_violation: Violation rate among the used resamples.
        delta_violation: resampled - base, the signed read that a KL cannot give.
    """

    branch_id: str
    kl: float
    n_used: int
    n_total: int
    base_violation: float
    resampled_violation: float
    delta_violation: float


def counterfactual_importance(
    target_text: str,
    branches: Sequence[Branch],
    base_severities: Sequence[float | None],
    tau: float | None = None,
    model: str = DEFAULT_MODEL,
    alpha: float = ALPHA,
) -> Importance:
    """Equation 1: the effect of resampling one chunk away.

    Only resamples that actually said something *different* count. Without that filter
    the metric mostly measures paraphrase, since a model asked to redo a sentence will
    often redo it near-verbatim.

    Args:
        target_text: The chunk being ablated, `S_i`.
        branches: Resamples taken from this branch point.
        base_severities: Outcomes observed with `S_i` present — the original rollout,
            repeated if you have repeats.
        tau: Similarity cutoff; None uses the median similarity across `branches`, which
            is the paper's self-calibrating rule.
        model: Embedding backend.
        alpha: Laplace smoothing.

    Returns:
        The importance record. With no dissimilar resamples the KL is 0.0 and `n_used`
        is 0 — read that as "no evidence", never as "no effect".
    """
    reps = [b.replacement for b in branches]
    if not reps:
        base = severity_dist(base_severities, alpha)
        rate, _, _ = violation_rate(base_severities)
        return Importance(_bid(branches), 0.0, 0, 0, rate, 0.0, 0.0)
    sims = cosine(encode([target_text], model=model), encode(reps, model=model))[0]
    cut = float(np.median(sims)) if tau is None else tau
    used = [b for b, s in zip(branches, sims) if s < cut]
    p = severity_dist([b.severity for b in used], alpha)
    q = severity_dist(base_severities, alpha)
    base_rate, _, _ = violation_rate(base_severities)
    res_rate, _, _ = violation_rate([b.severity for b in used])
    return Importance(
        branch_id=_bid(branches),
        kl=kl(p, q),
        n_used=len(used),
        n_total=len(branches),
        base_violation=base_rate,
        resampled_violation=res_rate,
        delta_violation=res_rate - base_rate,
    )


def _bid(branches: Sequence[Branch]) -> str:
    return branches[0].branch_id if branches else ""


# -- 2. resilience ----------------------------------------------------------------


def resilience(
    target_text: str,
    rounds: Sequence[Sequence[str]],
    tau: float,
    model: str = DEFAULT_MODEL,
) -> int:
    """Algorithm 1: how many resampling rounds the content survives.

    Each round is a fresh batch of candidate replacements. If the best candidate is still
    semantically the target, the content came back, the counter advances, and the next
    round resamples again. The first round where nothing matches is where the model
    finally let the idea go.

    Args:
        target_text: The sentence being removed, `S_i`.
        rounds: Candidate replacements per round, outermost = round order. Supplying
            these precomputed keeps this function pure; `sampler.resilience_rounds`
            generates them.
        tau: Similarity above which the content counts as having reappeared.
        model: Embedding backend.

    Returns:
        Iterations survived. 0 means the model dropped the idea on the first try.
    """
    k = 0
    current = target_text
    for cands in rounds:
        cands = [c for c in cands if c.strip()]
        if not cands:
            break
        j, sim = best_match(current, cands, model=model)
        if sim > tau:
            current = cands[j]
            k += 1
        else:
            break
    return k


def resilience_tau(
    target_text: str, corpus: Sequence[str], model: str = DEFAULT_MODEL
) -> float:
    """The median-similarity threshold, calibrated against a comparison corpus.

    Args:
        target_text: The sentence being removed.
        corpus: Texts to calibrate against — the trace's own other sentences work well,
            since they set the scale of "how similar do two sentences from this model
            look anyway".
        model: Embedding backend.

    Returns:
        Median cosine of the target against the corpus; 0.0 for an empty corpus.
    """
    if not corpus:
        return 0.0
    sims = cosine(
        encode([target_text], model=model), encode(list(corpus), model=model)
    )[0]
    return float(np.median(sims))


# -- 3. counterfactual++ ----------------------------------------------------------


def counterfactual_pp_importance(
    target_text: str,
    branches: Sequence[Branch],
    base_severities: Sequence[float | None],
    tau: float,
    model: str = DEFAULT_MODEL,
    alpha: float = ALPHA,
) -> Importance:
    """Equation 2: importance measured only where the idea never came back.

    A branch qualifies when (1) its replacement is dissimilar to the target and (2) no
    sentence anywhere in its continuation is similar either. That second condition is
    the whole point — it turns "we perturbed a sentence" into "we removed an idea".

    Args:
        target_text: The chunk being ablated.
        branches: Resamples, each carrying its `downstream` sentences.
        base_severities: Outcomes with the chunk present.
        tau: Similarity cutoff for "this is the same content".
        model: Embedding backend.
        alpha: Laplace smoothing.

    Returns:
        The importance record, with `n_used` the count of fully-eliminated rollouts.
        A small `n_used` is itself the finding for a resilient sentence: the model would
        not let the idea go, so its effect cannot be read off this way.
    """
    used: list[Branch] = []
    for b in branches:
        if not b.replacement.strip():
            continue
        _, s_rep = best_match(target_text, [b.replacement], model=model)
        if s_rep >= tau:
            continue
        if b.downstream:
            _, s_down = best_match(target_text, b.downstream, model=model)
            if s_down >= tau:
                continue
        used.append(b)
    p = severity_dist([b.severity for b in used], alpha)
    q = severity_dist(base_severities, alpha)
    base_rate, _, _ = violation_rate(base_severities)
    res_rate, _, _ = violation_rate([b.severity for b in used])
    return Importance(
        branch_id=_bid(branches),
        kl=kl(p, q),
        n_used=len(used),
        n_total=len(branches),
        base_violation=base_rate,
        resampled_violation=res_rate,
        delta_violation=res_rate - base_rate,
    )


# -- 4. cumulative effect curve ---------------------------------------------------


@dataclass
class EffectPoint:
    """One point on a trace's cumulative effect curve.

    Attributes:
        pos: Chunk index the prefix ran through (inclusive).
        rel_pos: `pos` normalised to [0, 1].
        rate: Violation rate of resamples continued from that prefix.
        lo: Wilson lower bound.
        hi: Wilson upper bound.
        n: Resamples at this prefix.
        step: Change in rate from the previous point — the paper's Equation 3, the
            signed causal effect of uttering the chunk at `pos`.
    """

    pos: int
    rel_pos: float
    rate: float
    lo: float
    hi: float
    n: int
    step: float


def effect_curve(
    prefix_outcomes: dict[int, Sequence[float | None]], n_chunks: int
) -> list[EffectPoint]:
    """p(violation | first i chunks) across a trace, and its per-chunk increments.

    This is the readout that does not need a matching step, a threshold, or an ablation:
    resample from each prefix, judge, and watch the violation rate climb. Where it climbs
    is where the decision was made. A curve that is already high at chunk 0 says the
    scenario decided it, not the reasoning.

    Args:
        prefix_outcomes: chunk index -> severities of continuations from that prefix.
        n_chunks: Trace length, for normalising positions.

    Returns:
        Points in position order, `step` measured against the previous present position.
    """
    out: list[EffectPoint] = []
    prev = None
    for pos in sorted(prefix_outcomes):
        rate, (lo, hi), n = violation_rate(list(prefix_outcomes[pos]))
        out.append(
            EffectPoint(
                pos=pos,
                rel_pos=pos / (n_chunks - 1) if n_chunks > 1 else 0.0,
                rate=rate,
                lo=lo,
                hi=hi,
                n=n,
                step=0.0 if prev is None else rate - prev,
            )
        )
        prev = rate
    return out


def summarise_by_tag(
    importances: Sequence[Importance],
    tags: dict[str, str],
) -> dict[str, dict]:
    """Aggregate per-branch importances into the paper's per-category bars.

    Args:
        importances: One record per branch point.
        tags: branch_id -> function tag.

    Returns:
        tag -> {n, mean_kl, mean_delta, mean_n_used}, sorted by mean KL descending.
    """
    buckets: dict[str, list[Importance]] = {}
    for imp in importances:
        buckets.setdefault(tags.get(imp.branch_id, "unlabelled"), []).append(imp)
    out = {}
    for tag, items in buckets.items():
        out[tag] = {
            "n": len(items),
            "mean_kl": float(np.mean([i.kl for i in items])),
            "mean_delta": float(np.mean([i.delta_violation for i in items])),
            "mean_n_used": float(np.mean([i.n_used for i in items])),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mean_kl"]))


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, for reporting how spread an outcome distribution is."""
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence: a symmetric, bounded companion to `kl`.

    KL is asymmetric and unbounded, which makes it awkward to compare across branch
    points with very different sample counts. JS is reported alongside so a reader can
    tell a genuine distribution shift from a small-sample KL blow-up.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bootstrap_mean(
    values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for a mean, used for per-tag bars.

    Args:
        values: Observations.
        n_boot: Bootstrap replicates.
        seed: RNG seed, so a figure regenerates identically.
        alpha: Two-sided level.

    Returns:
        (mean, lo, hi); all three are 0.0 for an empty input, and the point estimate
        with zero width for n == 1 — never a fabricated interval.
    """
    v = np.asarray(
        [x for x in values if x is not None and not math.isnan(x)], dtype=np.float64
    )
    if v.size == 0:
        return 0.0, 0.0, 0.0
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (
        float(v.mean()),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )
