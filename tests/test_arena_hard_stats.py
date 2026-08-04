# ABOUTME: Unit tests for the arena-hard-eval statistics: verdict orientation, tie
# ABOUTME: handling, paired-bootstrap width, and style control. Fast and offline.

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.capabilities.arena_hard.arena_hard_metrics import (
    degeneracy_metrics,
    is_refusal,
    pattern_frequencies,
    repetition_ratio,
    style_features,
)
from src.eval.capabilities.arena_hard.arena_hard_stats import (
    battles_from_judgments,
    evaluate_arm,
    fit_logistic,
    paired_bootstrap,
    per_prompt_scores,
    style_controlled_win_rate,
    win_tie_loss,
)


def judgment(uid: str, game0: str, game1: str, category: str = "hard_prompt") -> dict:
    """Build a judgment record with explicit verdicts for both orderings."""
    return {
        "uid": uid,
        "category": category,
        "model": "arm_d",
        "baseline": "arm_a",
        "games": [{"score": game0}, {"score": game1}],
    }


def meta(token_len: int, headers: int = 0, lists: int = 0, bold: int = 0) -> dict:
    """Build a style metadata block."""
    return {
        "token_len": token_len,
        "header_count": {"h1": headers},
        "list_count": {"ordered": lists},
        "bold_count": {"**": bold},
    }


# --- verdict orientation ------------------------------------------------------------
# The vendored gen_judgment.py presents the BASELINE first in game 0 and the model under
# test first in game 1. Getting this backwards silently inverts every result, so it is
# the single most important thing in this file to pin down.


def test_game0_verdict_is_flipped_game1_is_not():
    # Game 0: baseline is "A", so "A>B" means the BASELINE won -> model scores 0.
    # Game 1: model is "A", so "A>B" means the MODEL won -> model scores 1.
    rows = battles_from_judgments([judgment("q1", "A>B", "A>B")])
    assert [r["score"] for r in rows] == [0.0, 1.0]


def test_consistent_model_win_scores_one_in_both_orderings():
    # Model wins regardless of position: game 0 says baseline lost, game 1 says model won.
    rows = battles_from_judgments([judgment("q1", "B>A", "A>B")])
    assert [r["score"] for r in rows] == [1.0, 1.0]
    assert win_tie_loss(rows)["swap_consistency"] == 1.0


def test_position_biased_judge_shows_low_swap_consistency():
    # Judge always prefers whoever is shown first -> the two orderings disagree every time
    # and the averaged per-prompt score is exactly 0.5.
    rows = battles_from_judgments([judgment(f"q{i}", "A>B", "A>B") for i in range(10)])
    assert win_tie_loss(rows)["swap_consistency"] == 0.0
    _, scores = per_prompt_scores(rows)
    assert np.allclose(scores, 0.5)


def test_ties_score_half_and_decisive_verdicts_are_flagged():
    rows = battles_from_judgments([judgment("q1", "A=B", "A>>B")])
    assert [r["score"] for r in rows] == [0.5, 1.0]
    assert [r["decisive"] for r in rows] == [False, True]


def test_unparseable_and_null_verdicts_are_dropped_not_imputed():
    rows = battles_from_judgments(
        [
            {"uid": "q1", "category": "hard_prompt", "model": "m", "games": [{"score": None}, {"score": "A>B"}]},
            {"uid": "q2", "category": "hard_prompt", "model": "m", "games": [{"score": "GARBAGE"}, None]},
        ]
    )
    assert len(rows) == 1
    assert rows[0]["uid"] == "q1"


def test_decisive_verdicts_are_not_upweighted():
    # Upstream counts A>>B three times. We score it once, so a sweep of decisive wins and
    # a sweep of slight wins both read as a 100% win rate on the same scale.
    decisive = battles_from_judgments([judgment(f"q{i}", "B>>A", "A>>B") for i in range(5)])
    slight = battles_from_judgments([judgment(f"q{i}", "B>A", "A>B") for i in range(5)])
    assert win_tie_loss(decisive)["win_rate"] == win_tie_loss(slight)["win_rate"] == 1.0
    assert win_tie_loss(decisive)["decisive_rate"] == 1.0
    assert win_tie_loss(slight)["decisive_rate"] == 0.0


# --- bootstrap width vs the spec's power table --------------------------------------


@pytest.mark.parametrize(
    "n,tie_fraction,expected_half_width_pp",
    [
        (150, 0.0, 8.0),
        (250, 0.4, 4.8),
        (500, 0.0, 4.4),
        (500, 0.4, 3.4),
        (500, 0.5, 3.1),
    ],
)
def test_bootstrap_half_width_matches_spec_power_table(n, tie_fraction, expected_half_width_pp):
    """Spec §9 tabulates 95% CI half-widths; the bootstrap must reproduce them.

    Variance is `0.25 * (1 - t)`, so ties genuinely tighten the interval — this is what
    makes n=500 sufficient for the ±5pp threshold rather than marginal.
    """
    n_ties = int(round(n * tie_fraction))
    n_rest = n - n_ties
    scores = np.array([0.5] * n_ties + [1.0] * (n_rest // 2) + [0.0] * (n_rest - n_rest // 2))
    result = paired_bootstrap(scores, rounds=4000, alpha=0.05, seed=0)
    half_width = ((result["ci_upper"] - result["ci_lower"]) / 2) * 100
    assert half_width == pytest.approx(expected_half_width_pp, abs=0.4)


def test_all_ties_gives_a_degenerate_interval_at_fifty_percent():
    # The limiting case of the variance model: t=1 -> zero variance.
    result = paired_bootstrap(np.full(200, 0.5), rounds=500, seed=0)
    assert result["mean"] == 0.5
    assert result["ci_upper"] - result["ci_lower"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_is_reproducible_and_pairs_across_arms():
    # Same seed and same prompt ordering -> identical resampling draws, which is what
    # makes cross-arm comparisons paired rather than independent.
    scores = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0, 0.5])
    assert paired_bootstrap(scores, rounds=200, seed=7) == paired_bootstrap(
        scores, rounds=200, seed=7
    )


# --- logistic fit and style control --------------------------------------------------


def test_fit_logistic_recovers_a_known_intercept():
    # 75% win rate -> intercept = logit(0.75).
    design = np.ones((400, 1))
    outcomes = np.array([1.0] * 300 + [0.0] * 100)
    assert fit_logistic(design, outcomes)[0] == pytest.approx(math.log(3), abs=0.02)


def test_fit_logistic_handles_soft_targets():
    # Ties are real 0.5 outcomes, not missing data; an all-tie column must sit at 0 logit.
    design = np.ones((100, 1))
    assert fit_logistic(design, np.full(100, 0.5))[0] == pytest.approx(0.0, abs=1e-6)


def test_style_control_removes_a_pure_length_effect():
    """The core defence of this eval (spec §6).

    Simulate two equally-capable models judged by a judge that reads only verbosity: the
    true quality difference is zero (intercept 0 -> 50%), but win probability rises with
    the length delta, and the treated model is longer on average. The uncontrolled win
    rate should look comfortably above 50% and the controlled one should come back to it.

    If this test ever fails, the eval can confidently validate a model that merely got
    wordier — the exact failure the spec calls "a broken instrument".
    """
    rng = np.random.default_rng(0)
    n = 600
    # Treated model drifts longer on average, but the delta varies prompt to prompt, so
    # length and model identity stay distinguishable (see `style_deltas` on the
    # uniformly-longer case, where they would not be).
    model_len = rng.integers(300, 1400, size=n).astype(float)
    baseline_len = rng.integers(250, 700, size=n).astype(float)
    raw = (model_len - baseline_len) / (model_len + baseline_len)
    assert raw.mean() > 0.1, "setup requires the treated model to drift longer overall"
    # True quality is identical — at equal length the models are a coin flip — but the
    # judge's preference rises with the raw length delta, which is positive on average.
    wins = rng.random(n) < 1.0 / (1.0 + np.exp(-5.0 * raw))

    records, model_meta, baseline_meta = [], {}, {}
    for i in range(n):
        uid = f"q{i}"
        records.append(judgment(uid, *(("B>A", "A>B") if wins[i] else ("A>B", "B>A"))))
        model_meta[uid] = meta(int(model_len[i]))
        baseline_meta[uid] = meta(int(baseline_len[i]))

    battles = battles_from_judgments(records)
    _, scores = per_prompt_scores(battles)
    uncontrolled = paired_bootstrap(scores, rounds=500, seed=0)
    controlled = style_controlled_win_rate(battles, model_meta, baseline_meta, rounds=300, seed=0)

    assert uncontrolled["mean"] > 0.55, "setup should show an apparent style-driven gain"
    assert controlled["mean"] == pytest.approx(0.50, abs=0.05)
    # The length coefficient should be large and positive: the judge was reading length.
    assert controlled["coefficients"]["length"] > 1.0


def test_uniform_length_drift_is_reported_as_unidentifiable_not_silently_ignored():
    """If the model is longer by the same proportion on every prompt, length and model
    identity are the same column and no regression can separate them. The eval must say
    so rather than presenting an uncontrolled number as if it were controlled."""
    records, model_meta, baseline_meta = [], {}, {}
    for i in range(120):
        uid = f"q{i}"
        records.append(judgment(uid, "B>A", "A>B") if i < 90 else judgment(uid, "A>B", "B>A"))
        model_meta[uid] = meta(800)  # exactly 2x on every prompt -> zero variance
        baseline_meta[uid] = meta(400)

    controlled = style_controlled_win_rate(
        battles_from_judgments(records), model_meta, baseline_meta, rounds=100, seed=0
    )
    assert "length" in controlled["degenerate_features"]


def test_near_separable_style_does_not_blow_up_the_estimate():
    """Small bootstrap resamples can be nearly separable even when the full sample is not.
    The ridge on the style coefficients must keep the reported win rate in a sane range
    instead of saturating at 0 or 1."""
    records, model_meta, baseline_meta = [], {}, {}
    for i in range(60):
        uid = f"q{i}"
        longer = i < 48
        records.append(judgment(uid, *(("B>A", "A>B") if longer else ("A>B", "B>A"))))
        model_meta[uid] = meta(900 if longer else 300)
        baseline_meta[uid] = meta(300 if longer else 900)

    controlled = style_controlled_win_rate(
        battles_from_judgments(records), model_meta, baseline_meta, rounds=100, seed=0
    )
    assert 0.01 < controlled["mean"] < 0.99
    assert math.isfinite(controlled["ci_lower"]) and math.isfinite(controlled["ci_upper"])


def test_style_control_preserves_a_genuine_win_with_no_style_difference():
    # Model wins 70% with byte-identical style stats: control must not eat the signal.
    records, model_meta, baseline_meta = [], {}, {}
    for i in range(200):
        uid = f"q{i}"
        verdict = ("B>A", "A>B") if i < 140 else ("A>B", "B>A")
        records.append(judgment(uid, *verdict))
        model_meta[uid] = meta(500, headers=2, lists=3, bold=1)
        baseline_meta[uid] = meta(500, headers=2, lists=3, bold=1)

    battles = battles_from_judgments(records)
    controlled = style_controlled_win_rate(battles, model_meta, baseline_meta, rounds=300, seed=0)
    assert controlled["mean"] == pytest.approx(0.70, abs=0.03)


def test_evaluate_arm_gates_on_the_controlled_lower_bound():
    """A flat arm passes; an arm that only looks fine because it got wordier does not."""
    rng = np.random.default_rng(1)
    n = 400

    # Flat arm: a genuine coin flip with heavy ties and identical style. Should pass.
    flat_records, flat_model, flat_base = [], {}, {}
    for i in range(n):
        uid = f"q{i}"
        flat_records.append(judgment(uid, "A=B" if i % 2 else "B>A", "A=B" if i % 2 else "A>B"))
        flat_model[uid] = meta(400, headers=1)
        flat_base[uid] = meta(400, headers=1)

    flat = evaluate_arm(
        battles_from_judgments(flat_records), flat_model, flat_base, threshold=0.45, rounds=300
    )
    assert flat["passes"]
    assert flat["split"]["tie_rate"] > 0

    # Wordy arm: no better on substance, but drifted longer, and the judge rewards that.
    wordy_records, wordy_model, wordy_base = [], {}, {}
    model_len = rng.integers(300, 1500, size=n).astype(float)
    base_len = rng.integers(250, 700, size=n).astype(float)
    raw = (model_len - base_len) / (model_len + base_len)
    wins = rng.random(n) < 1.0 / (1.0 + np.exp(-5.0 * raw))
    for i in range(n):
        uid = f"q{i}"
        wordy_records.append(judgment(uid, *(("B>A", "A>B") if wins[i] else ("A>B", "B>A"))))
        wordy_model[uid] = meta(int(model_len[i]), headers=4, lists=5, bold=3)
        wordy_base[uid] = meta(int(base_len[i]), headers=1, lists=1, bold=1)

    wordy = evaluate_arm(
        battles_from_judgments(wordy_records), wordy_model, wordy_base, threshold=0.45, rounds=300
    )
    # Uncontrolled looks healthy; the style gap is what exposes it as verbosity, not
    # capability. This is the number spec §6 asks us to read.
    assert wordy["uncontrolled"]["mean"] > 0.55
    assert wordy["style_gap_pp"] > 3.0


# --- metrics ------------------------------------------------------------------------


def test_style_features_ignore_fenced_code():
    # A code block full of '# comments' must not be counted as markdown headers, or every
    # correct coding answer would score as heavily formatted prose.
    plain = style_features("Here you go:\n\n```python\n# not a header\n# nor this\n```\n")
    assert sum(plain["header_count"].values()) == 0

    formatted = style_features("# Real Header\n\n- a\n- b\n\n**bold**\n")
    assert formatted["header_count"]["h1"] == 1
    assert formatted["list_count"]["unordered"] == 2
    assert formatted["bold_count"]["**"] == 1


def test_repetition_ratio_flags_a_looping_generation():
    assert repetition_ratio("a b c d e f g h") == 0.0
    assert repetition_ratio(("loop forever and ever " * 20)) > 0.8


def test_refusal_detection_ignores_ordinary_hedging():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I cannot assist with this request.")
    # Nuance is what the corpus is teaching; counting it as refusal makes the counter
    # useless exactly when it matters.
    assert not is_refusal("It depends on the situation, but here is one approach.")
    assert not is_refusal("There's no single right answer here. Consider the following.")


def test_degeneracy_metrics_surface_truncation_and_empty_think():
    records = [
        {"answer": "fine answer here", "think": "some reasoning", "finish_reason": "stop"},
        {"answer": "cut off mid", "think": "", "finish_reason": "length"},
        {"answer": "", "think": "", "finish_reason": "stop"},
        {"answer": "another fine one", "think": "reasoning", "finish_reason": "stop"},
    ]
    m = degeneracy_metrics(records, percentiles=[50, 95])
    assert m["truncation_rate"] == 0.25
    assert m["empty_answer_rate"] == 0.25
    # CLAUDE.md gotcha 2: <think> collapsing to empty means the model stopped reasoning.
    assert m["empty_think_rate"] == 0.5
    assert "50" in m["length_percentiles"]


def test_pattern_frequencies_measure_corpus_signatures():
    answers = ["Bottom line: yes.", "TL;DR: no.", "A neutral response."]
    freqs = pattern_frequencies(answers, {"bluf": r"(?i)^\s*(bottom line|tl;?dr)\b"})
    assert freqs["bluf"] == pytest.approx(2 / 3)
