# ABOUTME: Paired-bootstrap win rates and style-controlled Bradley-Terry fits over
# ABOUTME: Arena-Hard pairwise judgments, for the capability regression eval.

"""Statistics for the capability regression eval.

We are making a **non-inferiority** claim, not a superiority claim: the question is not
"is the treated model better" but "how much worse could it plausibly be". A point
estimate near 50% is therefore not sufficient on its own — the interval has to be tight
enough to exclude a regression we would care about. Everything here exists to make that
interval as tight as the data honestly allows.

Three deliberate departures from the vendored `show_result.py`:

1. **Paired bootstrap over prompts, not battles.** Upstream resamples individual battles
   (`groupby("model").sample(frac=1.0, replace=True)`). Every arm answers an identical
   prompt set, so resampling *prompts* and carrying both arms' outcomes together removes
   between-prompt difficulty variance from the interval. It is strictly more powerful and
   it is free.

2. **No decisive-verdict upweighting.** Upstream counts `A>>B` three times, which is a
   reasonable Bradley-Terry prior for a leaderboard but silently changes the estimand:
   the reported number is no longer a win rate, and the spec §9 variance model
   (`0.25 × (1 − t)`) no longer describes it. We score every verdict once, so the
   controlled and uncontrolled numbers stay on the same scale and their *gap* — which is
   what spec §6 asks us to read — is meaningful. The decisive fraction is reported
   separately as a diagnostic.

3. **Both orderings averaged per prompt.** Position bias in pairwise LLM judges is large.
   Averaging the two orderings into one per-prompt score cancels it in expectation, and
   the residual disagreement is reported as `swap_consistency`, a judge-health metric.

Ties score 0.5, so a per-prompt score lands in {0, 0.25, 0.5, 0.75, 1}. High tie rates
are expected here — the arms are near-identical checkpoints of one base model trained on
largely overlapping mixtures — and they shrink the interval relative to a naive binomial,
which is why the spec's power table is not pessimistic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Arena-Hard's 5-point verdict scale, oriented as "score for the model under test".
# `None` marks an unparseable judgment, which is dropped rather than imputed.
_VERDICT_TO_SCORE = {
    "A>>B": 1.0,
    "A>B": 1.0,
    "A=B": 0.5,
    "B=A": 0.5,
    "B>A": 0.0,
    "B>>A": 0.0,
    "A<<B": 0.0,
    "A<B": 0.0,
    "B<<A": 1.0,
    "B<A": 1.0,
}
_DECISIVE = {"A>>B", "B>>A", "A<<B", "B<<A"}


def battles_from_judgments(records: list[dict]) -> list[dict]:
    """Turn raw judgment records into per-prompt, per-game battle rows.

    Handles the ordering bookkeeping in the vendored `gen_judgment.py`: game 0 presents
    the baseline as "Assistant A", game 1 swaps them. Game 0's verdict is therefore
    flipped so both games read as "score for the model under test".

    Args:
        records: Parsed judgment JSONL rows, each with `games` of length 2.

    Returns:
        One row per (prompt, game) with keys `uid`, `category`, `model`, `game`,
        `score`, `decisive`. Rows whose verdict failed to parse are dropped.
    """
    rows = []
    for rec in records:
        games = rec.get("games") or []
        for game_idx, game in enumerate(games):
            if not game or game.get("score") is None:
                continue
            verdict = str(game["score"]).strip()
            if verdict not in _VERDICT_TO_SCORE:
                continue
            score = _VERDICT_TO_SCORE[verdict]
            # Game 0 shows the baseline first, so its verdict is expressed from the
            # baseline's point of view and must be inverted.
            if game_idx == 0:
                score = 1.0 - score
            rows.append(
                {
                    "uid": rec["uid"],
                    "category": rec["category"],
                    "model": rec["model"],
                    "game": game_idx,
                    "score": score,
                    "decisive": verdict in _DECISIVE,
                }
            )
    return rows


def per_prompt_scores(battles: list[dict]) -> tuple[list[str], np.ndarray]:
    """Collapse both orderings into one score per prompt.

    Args:
        battles: Rows from `battles_from_judgments`, all for a single model and category.

    Returns:
        `(uids, scores)` with `scores[i]` the mean over that prompt's games.
    """
    grouped: dict[str, list[float]] = {}
    for row in battles:
        grouped.setdefault(row["uid"], []).append(row["score"])
    uids = sorted(grouped)
    scores = np.array([float(np.mean(grouped[u])) for u in uids], dtype=float)
    return uids, scores


def win_tie_loss(battles: list[dict]) -> dict[str, float]:
    """Three-way split over individual games, plus judge-health diagnostics.

    Reported always, never only the headline: a model that converts both wins and losses
    into ties has changed behaviour even at exactly 50% win rate.

    Args:
        battles: Rows from `battles_from_judgments` for one model and category.

    Returns:
        Rates for `win`/`tie`/`loss`, the `decisive_rate`, and `swap_consistency` — the
        fraction of prompts where both orderings agreed. Low swap consistency means the
        judge is reading position rather than quality (footgun §10.6).
    """
    if not battles:
        raise ValueError("win_tie_loss called with no battles")
    n = len(battles)
    wins = sum(1 for b in battles if b["score"] == 1.0)
    ties = sum(1 for b in battles if b["score"] == 0.5)
    losses = sum(1 for b in battles if b["score"] == 0.0)

    by_uid: dict[str, list[float]] = {}
    for row in battles:
        by_uid.setdefault(row["uid"], []).append(row["score"])
    paired = [v for v in by_uid.values() if len(v) == 2]
    consistency = (
        sum(1 for v in paired if v[0] == v[1]) / len(paired) if paired else float("nan")
    )

    return {
        "n_games": n,
        "n_prompts": len(by_uid),
        "win_rate": wins / n,
        "tie_rate": ties / n,
        "loss_rate": losses / n,
        "decisive_rate": sum(1 for b in battles if b["decisive"]) / n,
        "swap_consistency": consistency,
    }


def paired_bootstrap(
    scores: np.ndarray,
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap over prompts for a mean win rate.

    Resamples *prompt indices*, so when several arms are bootstrapped with the same seed
    and prompt order they share resampling draws — that is what makes cross-arm
    differences paired.

    Args:
        scores: Per-prompt scores in [0, 1].
        rounds: Bootstrap resamples.
        alpha: Two-sided level; the interval is [alpha/2, 1 - alpha/2].
        seed: RNG seed, fixed so a reported interval is reproducible.

    Returns:
        `mean`, `ci_lower`, `ci_upper`, `std_error`, `n`.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("paired_bootstrap called with no scores")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(rounds, n))
    means = scores[idx].mean(axis=1)
    return {
        "mean": float(scores.mean()),
        "ci_lower": float(np.quantile(means, alpha / 2)),
        "ci_upper": float(np.quantile(means, 1 - alpha / 2)),
        "std_error": float(means.std(ddof=1)),
        "n": n,
    }


def _flatten(counts: Any) -> float:
    """Sum a markdown count sub-dict, or pass a scalar through."""
    if isinstance(counts, dict):
        return float(sum(counts.values()))
    return float(counts)


def style_deltas(
    battles: list[dict],
    model_meta: dict[str, dict],
    baseline_meta: dict[str, dict],
) -> np.ndarray:
    """Build the per-battle style covariates the control regression removes.

    Four features, matching arena-hard-auto's validated parameterisation:

    - **Length asymmetry**, `(m − b) / (m + b)`: bounded, symmetric, and scale-free, so a
      long-answer prompt and a short-answer prompt contribute comparably.
    - **Header / list / bold density asymmetry**, where density is `count / (tokens + 1)`.
      Density rather than raw count, because otherwise "more formatted" and "longer" are
      the same feature and the regression cannot separate them — which is precisely the
      confound this eval exists to break.

    Args:
        battles: Rows from `battles_from_judgments`, one model and category.
        model_meta: `{uid: metadata}` for the model under test.
        baseline_meta: `{uid: metadata}` for the baseline arm.

    **Identifiability limit, worth stating in the writeup.** A feature with no variance
    across prompts carries no information and is zeroed out. The case that matters is
    length: if the treated model is uniformly longer than the baseline by a similar
    proportion on *every* prompt, then "length" and "model identity" are the same column
    and no regression can separate them. Style control then silently does nothing. The
    returned `degenerate` list names any such feature so the report can say so out loud
    rather than presenting an uncontrolled number as controlled.

    Returns:
        `(features, degenerate)` — a `(n_battles, 4)` z-scored matrix, and the names of
        features that had no usable variance.
    """
    keys = ["header_count", "list_count", "bold_count"]
    raw = []
    for row in battles:
        m, b = model_meta[row["uid"]], baseline_meta[row["uid"]]
        m_len, b_len = float(m["token_len"]), float(b["token_len"])
        length = (m_len - b_len) / (m_len + b_len) if (m_len + b_len) > 0 else 0.0
        feats = [length]
        for key in keys:
            m_d = _flatten(m[key]) / (m_len + 1.0)
            b_d = _flatten(b[key]) / (b_len + 1.0)
            feats.append((m_d - b_d) / (m_d + b_d + 1.0))
        raw.append(feats)

    arr = np.asarray(raw, dtype=float)
    names = ["length", "header_density", "list_density", "bold_density"]
    std = arr.std(axis=0)
    # A zero-variance feature carries no information; leaving it at zero keeps the design
    # matrix finite instead of producing NaNs that would propagate into the interval.
    flat = std < 1e-12
    std[flat] = 1.0
    degenerate = [name for name, is_flat in zip(names, flat) if is_flat]

    # Scaled but deliberately NOT mean-centred, which is where we part company with
    # arena-hard-auto's `show_result.py`. Every feature here is an asymmetry that is zero
    # exactly when the two answers match on that dimension, so leaving the origin alone
    # makes the fitted intercept "the win rate at no style difference" — the counterfactual
    # this eval is asking about. Centring would move the origin to the *mean observed*
    # delta, so the intercept would still carry the average drift we are trying to remove,
    # and a uniformly wordier model would keep most of its style-driven advantage while
    # appearing to have been controlled. Upstream centres because for a leaderboard only
    # relative ranking matters; here the absolute value is the whole claim.
    return arr / std, degenerate


def fit_logistic(
    features: np.ndarray,
    outcomes: np.ndarray,
    ridge: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Newton-Raphson logistic regression accepting soft targets.

    Ties are genuine 0.5 outcomes, not missing data, so the loss has to accept fractional
    targets — which rules out `sklearn.LogisticRegression` and is why this is hand-rolled.
    Minimising cross-entropy against a 0.5 target is exactly the Bradley-Terry treatment
    of a draw as half a win, and Newton converges on it in a handful of iterations.

    **The intercept is deliberately not penalised.** The ridge exists to stop the *style*
    coefficients diverging when a style feature nearly separates the outcomes — which
    happens in small bootstrap resamples even when the full sample is well behaved.
    Penalising the intercept too would shrink the reported win rate toward 50%, and since
    50% is exactly the value our non-inferiority gate wants to see, that would bias the
    test toward passing. A guardrail must never be shrunk toward its own pass condition.

    On real data the penalty is negligible: the Hessian scales with the number of battles
    (~n/4), so at n in the hundreds a unit ridge on z-scored features moves nothing. It
    only bites in the degenerate cases where the unpenalised estimate is meaningless.

    Args:
        features: `(n, k)` design matrix. Column 0 must be the intercept.
        outcomes: `(n,)` targets in [0, 1].
        ridge: L2 penalty applied to columns 1..k-1 (never to the intercept).
        max_iter: Newton iteration cap.
        tol: Convergence tolerance on the coefficient update.

    Returns:
        `(k,)` fitted coefficients.
    """
    _, k = features.shape
    w = np.zeros(k, dtype=float)
    penalty = np.full(k, float(ridge))
    penalty[0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(features @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = features.T @ (outcomes - p) - penalty * w
        weights = np.clip(p * (1.0 - p), 1e-10, None)
        hess = (features * weights[:, None]).T @ features + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.max(np.abs(step)) < tol:
            break
    return w


def style_controlled_win_rate(
    battles: list[dict],
    model_meta: dict[str, dict],
    baseline_meta: dict[str, dict],
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Win rate with length and markdown contributions regressed out.

    With a single model against a pinned baseline the Bradley-Terry design collapses to a
    logistic regression whose intercept *is* the model's log-odds against the baseline, so
    the controlled win rate is `sigmoid(intercept)` — the win rate the model would post if
    its responses matched the baseline's length and formatting.

    This is the primary number. Our corpus is prose-heavy interpersonal writing, exactly
    the kind that makes responses longer, more hedged and more scaffolded — and exactly
    the drift a preference judge rewards. Without this control, a model that merely got
    wordier posts a higher win rate while being no better, and reading that as "no
    regression" validates a broken model with a broken instrument.

    Bootstrap resampling is over **prompts**, taking all of a prompt's games together, so
    the two orderings of one prompt never land in different resamples.

    Args:
        battles: Rows from `battles_from_judgments` for one model and category.
        model_meta: `{uid: metadata}` for the model under test.
        baseline_meta: `{uid: metadata}` for the baseline arm.
        rounds: Bootstrap resamples.
        alpha: Two-sided level.
        seed: RNG seed.

    Returns:
        `mean`, `ci_lower`, `ci_upper`, `n`, and `coefficients` (the fitted style
        coefficients — a large length coefficient is itself the finding that the judge was
        rewarding verbosity).
    """
    if not battles:
        raise ValueError("style_controlled_win_rate called with no battles")

    deltas, degenerate = style_deltas(battles, model_meta, baseline_meta)
    design = np.column_stack([np.ones(len(battles)), deltas])
    outcomes = np.array([b["score"] for b in battles], dtype=float)

    coefs = fit_logistic(design, outcomes)
    point = float(1.0 / (1.0 + np.exp(-coefs[0])))

    by_uid: dict[str, list[int]] = {}
    for i, row in enumerate(battles):
        by_uid.setdefault(row["uid"], []).append(i)
    uids = sorted(by_uid)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(rounds):
        picked = rng.integers(0, len(uids), size=len(uids))
        rows = np.concatenate([by_uid[uids[j]] for j in picked])
        try:
            boot = fit_logistic(design[rows], outcomes[rows])
        except np.linalg.LinAlgError:
            # A degenerate resample (e.g. every drawn prompt a tie) is uninformative
            # rather than an error; skipping it is honest and the quantiles absorb it.
            continue
        draws.append(1.0 / (1.0 + np.exp(-boot[0])))

    arr = np.asarray(draws, dtype=float)
    return {
        "mean": point,
        "ci_lower": float(np.quantile(arr, alpha / 2)),
        "ci_upper": float(np.quantile(arr, 1 - alpha / 2)),
        "n": len(uids),
        "n_bootstrap": len(draws),
        # Non-empty means style control could not act on that feature; the "controlled"
        # number is uncontrolled in that dimension and must be reported as such.
        "degenerate_features": degenerate,
        "coefficients": {
            "intercept": float(coefs[0]),
            "length": float(coefs[1]),
            "header_density": float(coefs[2]),
            "list_density": float(coefs[3]),
            "bold_density": float(coefs[4]),
        },
    }


def evaluate_arm(
    battles: list[dict],
    model_meta: dict[str, dict],
    baseline_meta: dict[str, dict],
    threshold: float = 0.45,
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Full result block for one arm on one slice, including the §3 pass check.

    Args:
        battles: Rows from `battles_from_judgments` for one model and category.
        model_meta: `{uid: metadata}` for the model under test.
        baseline_meta: `{uid: metadata}` for the baseline arm.
        threshold: Minimum acceptable CI lower bound on the controlled win rate.
        rounds: Bootstrap resamples.
        alpha: Two-sided level.
        seed: RNG seed.

    Returns:
        `split`, `uncontrolled`, `controlled`, `threshold`, and `passes` — where `passes`
        gates on the **controlled** lower bound, the primary number per spec §3.
    """
    _, scores = per_prompt_scores(battles)
    uncontrolled = paired_bootstrap(scores, rounds=rounds, alpha=alpha, seed=seed)
    controlled = style_controlled_win_rate(
        battles, model_meta, baseline_meta, rounds=rounds, alpha=alpha, seed=seed
    )
    return {
        "split": win_tie_loss(battles),
        "uncontrolled": uncontrolled,
        "controlled": controlled,
        "threshold": threshold,
        "passes": controlled["ci_lower"] >= threshold,
        # Spec §6: the gap between controlled and uncontrolled is itself a finding about
        # our corpus — it is how much of any apparent gain was style rather than substance.
        "style_gap_pp": (uncontrolled["mean"] - controlled["mean"]) * 100.0,
    }


def subcategory_breakdown(
    battles: list[dict],
    questions: dict[str, dict],
    rounds: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict]:
    """Per-subcategory win rates within a slice.

    **Directional only.** At n≈100 per cell the interval is roughly ±8pp even with a
    healthy tie rate, so these say where to look, not whether a category regressed. Do
    not gate on them (spec §9).

    Args:
        battles: Rows from `battles_from_judgments` for one model and category.
        questions: `{uid: question}` carrying `subcategory`.
        rounds: Bootstrap resamples.
        alpha: Two-sided level.
        seed: RNG seed.

    Returns:
        `{subcategory: bootstrap block}`, each flagged `directional_only`.
    """
    grouped: dict[str, list[dict]] = {}
    for row in battles:
        sub = questions.get(row["uid"], {}).get("subcategory", "unknown")
        grouped.setdefault(sub, []).append(row)

    out = {}
    for sub, rows in sorted(grouped.items()):
        _, scores = per_prompt_scores(rows)
        block = paired_bootstrap(scores, rounds=rounds, alpha=alpha, seed=seed)
        block["directional_only"] = True
        out[sub] = block
    return out
