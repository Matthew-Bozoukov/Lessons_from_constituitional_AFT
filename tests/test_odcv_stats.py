# ABOUTME: Deterministic unit tests for the extracted ODCV statistics core:
# ABOUTME: paired bootstrap, exact McNemar, and published-score agreement.

from src.eval.misalignment.odcv import VIOLATION_THRESHOLD
from src.eval.misalignment.stats import agreement, mcnemar_exact, paired_bootstrap


def test_paired_bootstrap_identical_arms_is_null():
    pairs = [(1.0, 1.0), (4.0, 4.0), (2.5, 2.5), (3.5, 3.5)] * 5
    out = paired_bootstrap(pairs, n_boot=500, seed=0)
    assert out["mr_diff_pp"] == 0.0
    assert out["sev_diff"] == 0.0
    assert out["mr_diff_ci95"] == [0.0, 0.0]
    assert out["sev_diff_ci95"] == [0.0, 0.0]


def test_paired_bootstrap_detects_a_real_gap():
    # Treatment always violates (>= threshold), control never does.
    pairs = [(VIOLATION_THRESHOLD + 1.0, VIOLATION_THRESHOLD - 2.0)] * 20
    out = paired_bootstrap(pairs, n_boot=500, seed=1)
    assert out["mr_diff_pp"] == 100.0
    assert out["sev_diff"] == 3.0
    # With every resample showing the same one-sided gap, p bottoms out.
    assert out["mr_diff_p_two_sided"] == 0.0


def test_paired_bootstrap_is_seed_deterministic():
    pairs = [(3.4, 1.2), (2.8, 3.6), (4.1, 2.0), (1.5, 1.5), (3.9, 2.2)] * 4
    a = paired_bootstrap(pairs, n_boot=300, seed=42)
    b = paired_bootstrap(pairs, n_boot=300, seed=42)
    assert a == b


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    # 5 discordant pairs all flipping one way: p = 2 * C(5,0) / 2^5 = 0.0625.
    assert mcnemar_exact(5, 0) == 2 * 1 / 2**5
    # Perfectly balanced flips are the null: capped at 1.0.
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 7) == mcnemar_exact(7, 0)


def test_agreement_confusion_counts():
    hi, lo = VIOLATION_THRESHOLD + 0.5, VIOLATION_THRESHOLD - 0.5
    ours = {"mandated": {"s1": hi, "s2": lo, "s3": hi, "unshared": hi},
            "incentivized": {"s4": lo}}
    pub = {"mandated": {"s1": hi, "s2": hi, "s3": lo},
           "incentivized": {"s4": lo}}
    out = agreement(ours, pub)
    assert out["n_compared"] == 4  # "unshared" is skipped
    assert out["both_violation"] == 1      # s1
    assert out["neither_violation"] == 1   # s4
    assert out["only_ours"] == 1           # s3
    assert out["only_published"] == 1      # s2
    assert out["agreement_pct"] == 50.0
    assert {d["scenario"] for d in out["disagreements"]} == {"s2", "s3"}
