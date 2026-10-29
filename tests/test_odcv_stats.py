# ABOUTME: Deterministic unit tests for the ODCV comparison helpers: exact McNemar and
# ABOUTME: published-score agreement. (Intervals are tested with summarise in test_odcv_metrics.)

from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD
from src.eval.misalignment.odcv.stats import agreement, mcnemar_exact


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
