# ABOUTME: Offline tests for pooling several ODCV arms of one recipe: the interval carries
# ABOUTME: seed-to-seed variance, and arms that answered different questions are refused.

import json

import pytest

from src.eval.misalignment.odcv.odcv import summarise, summarise_pooled
from src.eval.misalignment.odcv.pool import pool, pooled_key

SCENARIOS = [f"S{i}" for i in range(8)]


def medians(rate_by_variant: dict[str, float], rollouts: int = 2):
    """{variant: {scenario: [severity per rollout]}} with `rate` of the scenarios violating."""
    out = {}
    for variant, rate in rate_by_variant.items():
        n_bad = round(rate * len(SCENARIOS))
        out[variant] = {
            s: [4.0] * rollouts if i < n_bad else [1.0] * rollouts
            for i, s in enumerate(SCENARIOS)
        }
    return out


def arm(tmp_path, key, rate, mode="think", scenarios=None, rollouts=2):
    """Write one published arm's run dir, as run_eval leaves it."""
    med = medians({"mandated": rate, "incentivized": rate}, rollouts)
    if scenarios is not None:
        med = {v: {s: cells[s] for s in scenarios} for v, cells in med.items()}
    out_dir = tmp_path / key
    (out_dir / "results").mkdir(parents=True)
    (out_dir / "results" / "results.json").write_text(json.dumps({
        "model_key": key,
        "ours": summarise(med, checkpoint=key),
        "per_scenario_medians": med,
    }))
    return {"target": f"org/{key}", "model_key": key, "mode": mode,
            "out_dir": out_dir, "repo": f"https://hf.co/datasets/org/{key}"}


def test_pooling_puts_each_arm_in_as_its_own_checkpoint(tmp_path):
    runs = [arm(tmp_path, f"rec_s{i}", rate) for i, rate in enumerate((0.25, 0.375, 0.5))]
    pooled = pool(runs, cfg=None, out_dir=tmp_path / "pooled")

    assert pooled["ours"]["overall"]["n_checkpoints"] == 3
    assert pooled["model_key"] == "rec-pooled3"
    assert [p["model_key"] for p in pooled["pooled_from"]] == ["rec_s0", "rec_s1", "rec_s2"]
    # Traceable: the pooled number names the repos it came from.
    assert all(p["repo"].startswith("https://hf.co/") for p in pooled["pooled_from"])
    # Enough to recompute the pool, per arm — never a merged blob that invites
    # re-summarising three seeds as one.
    assert set(pooled["per_scenario_medians_by_arm"]) == {"rec_s0", "rec_s1", "rec_s2"}
    assert "per_scenario_medians" not in pooled


def test_the_pooled_claim_is_about_the_recipe_not_the_seeds_that_ran(tmp_path):
    # The point of pooling is WHICH CLAIM the interval supports. Merging every rollout
    # into one arm keeps the old claim ("about this checkpoint only") on more data;
    # entering each arm as a checkpoint switches the estimator to the seed-sampled path,
    # so the estimand becomes the pipeline. Note this does not always WIDEN the bar —
    # SE^2 = T_A + T_B - T_C, and a large interaction term can shrink it — so the property
    # worth pinning is the claim and the method, never the width.
    rates = (0.25, 0.375, 0.5)
    runs = [arm(tmp_path, f"rec_s{i}", r) for i, r in enumerate(rates)]
    pooled = pool(runs, cfg=None, out_dir=tmp_path / "pooled")["ours"]

    merged = {v: {s: [x for r in rates for x in medians({v: r})[v][s]] for s in SCENARIOS}
              for v in ("mandated", "incentivized")}
    as_one_arm = summarise(merged)

    # Same rollouts, same point estimate — only the claim differs.
    assert pooled["overall"]["mr_pct"] == pytest.approx(as_one_arm["overall"]["mr_pct"], abs=1.0)

    pooled_mr = pooled["stats"]["overall"]["mr"]
    assert pooled_mr["method"].startswith("T_A + T_B - T_C")
    assert "a checkpoint from the pipeline" in pooled_mr["estimand"]
    assert any("seed-to-seed variance estimated" in c for c in pooled_mr["claims"])

    one_mr = as_one_arm["stats"]["overall"]["mr"]
    assert "T_B" in one_mr["method"] and "T_A" not in one_mr["method"]
    assert any("not estimated" in c for c in one_mr["claims"])


def test_arms_that_ran_different_scenarios_are_refused_by_name(tmp_path):
    runs = [arm(tmp_path, "rec_s0", 0.25),
            arm(tmp_path, "rec_s1", 0.25, scenarios=SCENARIOS[:6])]
    # Pooling assumes the arms answered the same question. Silently intersecting them
    # would report a recipe-level number over a cell set nobody chose.
    with pytest.raises(AssertionError, match="did not run the same scenarios"):
        pool(runs, cfg=None, out_dir=tmp_path / "pooled")


def test_arms_in_different_thinking_modes_are_refused(tmp_path):
    runs = [arm(tmp_path, "rec_s0", 0.25, mode="think"),
            arm(tmp_path, "rec_s1", 0.25, mode="nothink")]
    with pytest.raises(AssertionError, match="different thinking modes"):
        pool(runs, cfg=None, out_dir=tmp_path / "pooled")


def test_a_single_arm_is_not_a_pool():
    with pytest.raises(AssertionError, match=">= 2"):
        summarise_pooled({"only": medians({"mandated": 0.25, "incentivized": 0.25})})


def test_the_pooled_arm_is_named_after_what_the_arms_share():
    # It becomes a repo id and a `model:` tag, so a prefix that stops mid-token
    # ("..._par716_s" from _s0/_s1) would publish a repo named after nothing.
    assert pooled_key(["q_par716_s0", "q_par716_s1", "q_par716_s2"]) == "q_par716-pooled3"
    assert pooled_key(["arm-a", "arm-b"]) == "arm-pooled2"
    assert pooled_key(["alpha", "beta"]) == "pooled2"


def test_single_arm_summaries_still_report_one_checkpoint():
    # The refactor that added the checkpoint axis must not change what one arm reports.
    one = summarise(medians({"mandated": 0.25, "incentivized": 0.25}))
    assert one["overall"]["n_checkpoints"] == 1
    assert one["overall"]["mr_pct"] == pytest.approx(25.0, abs=0.1)
