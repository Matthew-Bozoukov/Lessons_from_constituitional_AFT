# ABOUTME: Deterministic unit tests for the ODCV comparison statistics: the paired arm
# ABOUTME: difference (closed form, via src/eval/stats), exact McNemar, and published-score agreement.

from src.eval.misalignment.odcv.odcv import VIOLATION_THRESHOLD
from src.eval.misalignment.odcv.stats import agreement, arm_difference, mcnemar_exact


def _cells(scores: list[float]) -> dict[str, float]:
    """Alternate mandated/incentivized cells over scenarios s0, s1, ... from a flat list."""
    out = {}
    for i, s in enumerate(scores):
        out[f"{'mandated' if i % 2 == 0 else 'incentivized'}/s{i // 2}"] = s
    return out


def test_arm_difference_identical_arms_is_null():
    cells = _cells([1.0, 1.0, 4.0, 4.0, 2.5, 2.5, 3.5, 3.5] * 5)
    out = arm_difference(cells, dict(cells))
    assert out["mr_diff_pp"] == 0.0 and out["sev_diff"] == 0.0
    assert out["mr_diff_ci95"] == [0.0, 0.0] and out["sev_diff_ci95"] == [0.0, 0.0]
    assert out["mr_diff_p_two_sided"] == 1.0


def test_arm_difference_detects_a_real_gap():
    # Treatment always violates (>= threshold), control never does.
    t = _cells([VIOLATION_THRESHOLD + 1.0] * 40)
    c = _cells([VIOLATION_THRESHOLD - 2.0] * 40)
    out = arm_difference(t, c)
    assert out["mr_diff_pp"] == 100.0 and out["sev_diff"] == 3.0
    assert out["mr_diff_p_two_sided"] == 0.0        # zero spread, non-zero gap
    assert out["n_scenarios"] == 20                  # 40 cells = 20 stories x 2 variants


def test_arm_difference_pairs_on_scenario_and_mixes_variants():
    """Half the stories violate in both variants under treatment only: the mixture gap is 50pp."""
    t = _cells([4.0, 4.0, 0.0, 0.0] * 10)   # s0 violates (both variants), s1 clean, ...
    c = _cells([0.0] * 40)
    out = arm_difference(t, c)
    assert out["mr_diff_pp"] == 50.0
    assert out["stats"]["mr"]["estimand"].startswith("difference (A - B), paired on items")
    assert "paired on items" in out["stats"]["mr"]["estimand"]
    lo, hi = out["mr_diff_ci95"]
    assert lo < 50.0 < hi and lo > 0


def test_arm_difference_uses_only_shared_cells():
    t = _cells([4.0] * 20)
    c = _cells([0.0] * 20)
    t["mandated/extra"] = 4.0                # not in control -> ignored
    out = arm_difference(t, c)
    assert out["n_scenarios"] == 10


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
