# ABOUTME: Unit tests for the Colosseum publish names — they must pass the naming gate
# ABOUTME: BEFORE a run, because the gate otherwise fires after the GPU hours are spent.

"""Why these exist.

The first version of `publish.py` built its repo name by gluing the adapter's own repo id
onto the eval name. Every one of the six names that produces is rejected: the adapter id
already carries a production date, so the name ends up with two, and the treatment arm
came to 119 characters against the Hub's 96-character limit. Nothing catches that until
`push_run_dir` runs — which is after the episodes, on a login node, with the GPU bill
already paid. So the names are asserted here instead.
"""

import re
from datetime import date

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.config import EXPERIMENTS
from src.eval.misalignment.colosseum.publish import arm_label, repo_name_for
from src.huggingface import gate_push

CONTROL = "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
TREATMENT = (
    "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-"
    "chunk-only-702-rank-64-dynbatch"
)
HUB_NAME_LIMIT = 96


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/eval/colosseum_jira.yaml")


@pytest.fixture(autouse=True)
def org(monkeypatch):
    """Pin the namespace: these runs publish to a personal org, not the group one."""
    monkeypatch.setenv("HF_ORG", "kunwar45")


def _card():
    return {
        "experiment": "colosseum_jira collusion",
        "date_generated": date.today().isoformat(),
        "constitution": "none",
        "source_repo": "teaching_claude_why_replication @ abc1234",
        "models": "target=x base=y judge=z",
        "generation_config": "{}",
        "schema": "rollouts/ results/ metadata/",
        "provenance": "uv run evals --name colosseum_jira ...",
    }


def test_every_published_name_passes_the_naming_gate(cfg):
    for experiment in EXPERIMENTS:
        for target in (CONTROL, TREATMENT):
            repo = repo_name_for(experiment, target, cfg)
            gate_push(repo, _card(), what="test")  # raises NamingError if bad


def test_every_published_name_fits_the_hub_limit(cfg):
    for experiment in EXPERIMENTS:
        for target in (CONTROL, TREATMENT):
            name = repo_name_for(experiment, target, cfg).split("/", 1)[1]
            assert len(name) <= HUB_NAME_LIMIT, f"{name} is {len(name)} chars"


def test_names_carry_exactly_one_date(cfg):
    # Two dates is what reusing the adapter repo id produced, and the naming law's whole
    # point is that the date is the run's own production date.
    for experiment in EXPERIMENTS:
        for target in (CONTROL, TREATMENT):
            name = repo_name_for(experiment, target, cfg).split("/", 1)[1]
            # ISO dates, not 4-digit tokens: `9284` is a row count, and counting it as a
            # year is what made the first version of this test fail on a correct name.
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", name)
            assert dates == [date.today().isoformat()], (
                f"{name} carries dates {dates}; expected only today's"
            )


def test_names_go_to_the_personal_org_not_the_group_one(cfg):
    repo = repo_name_for("collusion", TREATMENT, cfg)
    assert repo.startswith("kunwar45/"), repo


def test_the_two_arms_never_collide(cfg):
    for experiment in EXPERIMENTS:
        assert repo_name_for(experiment, CONTROL, cfg) != repo_name_for(
            experiment, TREATMENT, cfg
        )


def test_the_three_experiments_never_collide(cfg):
    for target in (CONTROL, TREATMENT):
        names = {repo_name_for(e, target, cfg) for e in EXPERIMENTS}
        assert len(names) == len(EXPERIMENTS)


def test_both_arms_under_test_have_a_label(cfg):
    assert arm_label(CONTROL, cfg) == "qwen36_table2_only_9284"
    assert arm_label(TREATMENT, cfg) == "qwen36_difficult_advice_chunk_only_702"


def test_an_unlabelled_arm_is_refused_rather_than_mislabelled(cfg):
    # Validly NAMED but not in arm_labels: the refusal must be about the missing label,
    # not about the repo id being malformed.
    with pytest.raises(AssertionError, match="no arm_labels entry"):
        repo_name_for("collusion", "LASR-Callum/2026-09-03-some-arm-nobody-labelled", cfg)
