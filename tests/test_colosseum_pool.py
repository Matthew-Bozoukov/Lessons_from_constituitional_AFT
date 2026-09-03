# ABOUTME: Unit tests for the Colosseum cross-arm contrast — pool() end to end over
# ABOUTME: fabricated per-seed files, which is the only GPU-free path through it.

"""pool() is where the result of an experiment is actually computed, so it needs a test
that does not require a GPU. These fabricate the `results/per_seed.json` that run() would
have written and check the contrast comes out the right way round, on the right seeds,
with the right refusals.
"""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.pool import pool

CONTROL = "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
TREATMENT = (
    "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-"
    "chunk-only-702-rank-64-dynbatch"
)
SEEDS = list(range(1, 41))


def _write_arm(tmp_path, arm, target, advantage, regret, seeds=SEEDS, mode="think"):
    """Fabricate one arm's per_seed.json. `advantage`/`regret` are {cell: (mean, sd)}."""
    # Deterministic, NOT hash(arm): str hashing is randomised per process
    # (PYTHONHASHSEED), which made this test pass or fail depending on the run.
    rng = np.random.default_rng({"control": 11, "treatment": 22}.get(arm, 33))
    measures = {"coalition_advantage": {}, "normalised_regret": {}}
    for measure, spec in (
        ("coalition_advantage", advantage),
        ("normalised_regret", regret),
    ):
        for cell, (mu, sd) in spec.items():
            draws = rng.normal(mu, sd, len(seeds))
            measures[measure][cell] = {str(s): float(v) for s, v in zip(seeds, draws)}
    out_dir = tmp_path / arm
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "results" / "per_seed.json").write_text(json.dumps(measures))
    return {
        "target": target,
        "model_key": arm,
        "mode": mode,
        "out_dir": str(out_dir),
        "repo": "",
    }


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/eval/colosseum_jira.yaml")


@pytest.fixture
def runs(tmp_path):
    """The hoped-for shape: the treatment coalition captures less from the channel, the
    team pays less regret for it, and the baseline cells are near-identical."""
    return [
        _write_arm(
            tmp_path,
            "control",
            CONTROL,
            {"baseline": (0.02, 0.05), "collusion": (0.29, 0.05)},
            {"baseline": (0.09, 0.03), "collusion": (0.41, 0.09)},
        ),
        _write_arm(
            tmp_path,
            "treatment",
            TREATMENT,
            {"baseline": (0.01, 0.05), "collusion": (0.18, 0.05)},
            {"baseline": (0.10, 0.03), "collusion": (0.30, 0.09)},
        ),
    ]


def test_pool_reports_the_headline_and_both_cells(cfg, runs, tmp_path):
    out = pool(runs, cfg, tmp_path)
    assert set(out["contrasts"]) == {
        "channel_advantage",
        "coalition_advantage_baseline",
        "coalition_advantage_collusion",
        "normalised_regret_baseline",
        "normalised_regret_collusion",
    }
    # run_eval needs these three to publish the pooled run.
    assert out["model_key"] == "collusion-contrast"
    assert out["mode"] == "think"
    assert {p["target"] for p in out["pooled_from"]} == {CONTROL, TREATMENT}


def test_the_headline_is_negative_when_the_treatment_colludes_less(cfg, runs, tmp_path):
    head = pool(runs, cfg, tmp_path)["contrasts"]["channel_advantage"]
    lo, hi = head["diff_ci95"]
    assert head["diff"] < 0, (
        "treatment minus control must be NEGATIVE for less collusion"
    )
    assert hi < 0, "a planted ~0.11 reduction at these spreads must exclude zero"
    assert head["n_seeds"] == 40


def test_the_baseline_cell_separates_less_collusion_from_a_cooperation_tax(
    cfg, runs, tmp_path
):
    # The whole reason both cells are reported: an arm that "colludes less" because it is
    # simply a worse teammate shows up here, and must not be hidden by the headline.
    c = pool(runs, cfg, tmp_path)["contrasts"]["normalised_regret_baseline"]
    lo, hi = c["diff_ci95"]
    assert lo < 0 < hi, "planted-equal baseline regret should NOT separate from zero"


def test_pool_identifies_control_by_the_configured_peer_not_by_order(
    cfg, runs, tmp_path
):
    forward = pool(runs, cfg, tmp_path)
    reversed_ = pool(list(reversed(runs)), cfg, tmp_path)
    assert forward["control"] == reversed_["control"] == CONTROL
    assert (
        forward["contrasts"]["channel_advantage"]["diff"]
        == reversed_["contrasts"]["channel_advantage"]["diff"]
    )


def test_pool_refuses_arms_served_in_different_modes(cfg, tmp_path):
    # The mode is pinned into the chat template at serve time, so arms in different modes
    # did not face the same environment and are not comparable.
    a = _write_arm(
        tmp_path,
        "control",
        CONTROL,
        {"baseline": (0.02, 0.1), "collusion": (0.29, 0.1)},
        {"baseline": (0.09, 0.03), "collusion": (0.41, 0.09)},
    )
    b = _write_arm(
        tmp_path,
        "treatment",
        TREATMENT,
        {"baseline": (0.01, 0.1), "collusion": (0.18, 0.1)},
        {"baseline": (0.10, 0.03), "collusion": (0.30, 0.09)},
        mode="nothink",
    )
    with pytest.raises(AssertionError, match="different modes"):
        pool([a, b], cfg, tmp_path)


def test_pool_refuses_a_ladder_that_is_not_a_two_arm_contrast(cfg, runs, tmp_path):
    third = _write_arm(
        tmp_path,
        "third",
        "LASR-Callum/2026-09-03-some-third-arm",
        {"baseline": (0.02, 0.1), "collusion": (0.25, 0.1)},
        {"baseline": (0.09, 0.03), "collusion": (0.38, 0.09)},
    )
    with pytest.raises(AssertionError, match="not a contrast"):
        pool(runs + [third], cfg, tmp_path)


def test_cooperation_contributes_one_regret_contrast_only(cfg, tmp_path):
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"experiment": "cooperation"}))
    runs = [
        _write_arm(
            tmp_path,
            "control",
            CONTROL,
            {"cooperation": (0.12, 0.04)},
            {"cooperation": (0.12, 0.04)},
            seeds=list(range(1, 21)),
        ),
        _write_arm(
            tmp_path,
            "treatment",
            TREATMENT,
            {"cooperation": (0.13, 0.04)},
            {"cooperation": (0.13, 0.04)},
            seeds=list(range(1, 21)),
        ),
    ]
    out = pool(runs, cfg, tmp_path)
    assert list(out["contrasts"]) == ["normalised_regret_cooperation"]
    assert out["contrasts"]["normalised_regret_cooperation"]["n_seeds"] == 20


def test_pool_writes_contrasts_json(cfg, runs, tmp_path):
    pool(runs, cfg, tmp_path)
    written = json.loads((tmp_path / "results" / "contrasts.json").read_text())
    assert written["experiment"] == "collusion"
    assert "channel_advantage" in written["contrasts"]
