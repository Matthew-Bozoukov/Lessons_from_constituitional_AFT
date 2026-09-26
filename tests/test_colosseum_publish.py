# ABOUTME: Unit tests for the Colosseum finisher's names — built by the law from a run's own
# ABOUTME: metadata, checked BEFORE a run, because the gate otherwise fires after the GPU hours.

"""Why these exist.

The first version of `publish.py` built its repo name by gluing the adapter's own repo id
onto the eval name: two dates, 119 characters against the Hub's 96. The second typed a short
label per arm into the config (`arm_labels`), which the naming law forbids — names are
built, not typed — and which produced 101-character names of its own once the harness
variant was typed in too. Now the finisher rebuilds exactly the name run_eval would have
minted: `<date>-<key>-<facets>-<arm>` from the run's `metadata/run_meta.json` (`model_key`
and the resolved config), so the two can never disagree and nothing is typed. These tests
assert that for every cell of both Colosseum evals, in a second, with no network.
"""

import re
from datetime import date

import pytest
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.config import EXPERIMENTS
from src.eval.misalignment.colosseum.hospital.config import CONDITIONS
from src.eval.misalignment.colosseum.publish import model_key_of, repo_name_for
from src.infra.huggingface import gate_push

# The two pre-law adapters and the keys run_eval resolves for them
# (src/infra/legacy_names.yaml), plus one lawful adapter, whose key is its undated name.
ARMS = (
    ("LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64", "qwen36_0_0"),
    (
        "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-"
        "chunk-only-702-rank-64-dynbatch",
        "qwen36_0_da_principle_scoped_7",
    ),
    (
        "dougalldeepmind/2026-09-24-qwen36-0-da-multiparty-human-15",
        "qwen36_0_da_multiparty_human_15",
    ),
)
HUB_NAME_LIMIT = 96


@pytest.fixture
def jira():
    return OmegaConf.load("configs/eval/colosseum_jira.yaml")


@pytest.fixture
def hospital():
    return OmegaConf.load("configs/eval/colosseum_hospital.yaml")


@pytest.fixture(autouse=True)
def org(monkeypatch):
    """Pin the namespace: these runs publish to the group org."""
    monkeypatch.setenv("HF_ORG", "dougalldeepmind")


def _meta(cfg, target, key, **overrides):
    """A run_meta.json as run_eval writes it: the arm's key plus the RESOLVED config."""
    return {
        "target": target,
        "model_key": key,
        "mode": "think",
        "config": OmegaConf.to_container(OmegaConf.merge(cfg, overrides), resolve=True),
    }


def _card(experiment):
    return {
        "experiment": experiment,
        "date_generated": date.today().isoformat(),
        "constitution": "none",
        "source_repo": "teaching_claude_why_replication @ abc1234",
        "models": "target=x base=y judge=z",
        "generation_config": "{}",
        "schema": "rollouts/ results/ metadata/",
        "provenance": "uv run evals --name colosseum_jira ...",
    }


def _cells(jira, hospital):
    for experiment in EXPERIMENTS:
        for target, key in ARMS:
            yield (
                "colosseum_jira",
                _meta(jira, target, key, experiment=experiment),
                experiment,
            )
    for condition in CONDITIONS:
        for target, key in ARMS:
            yield (
                "colosseum_hospital",
                _meta(hospital, target, key, condition=condition),
                condition,
            )


def test_every_published_name_passes_the_naming_gate(jira, hospital):
    for eval_name, meta, cell in _cells(jira, hospital):
        repo = repo_name_for(eval_name, meta)
        gate_push(repo, _card(f"{eval_name} {cell}"), what="test")  # raises if bad


def test_every_published_name_fits_the_hub_limit(jira, hospital):
    for eval_name, meta, _ in _cells(jira, hospital):
        name = repo_name_for(eval_name, meta).split("/", 1)[1]
        assert len(name) <= HUB_NAME_LIMIT, f"{name} is {len(name)} chars"


def test_names_carry_exactly_one_date(jira, hospital):
    # Two dates is what reusing the adapter repo id produced, and the naming law's whole
    # point is that the date is the run's own production date.
    for eval_name, meta, _ in _cells(jira, hospital):
        name = repo_name_for(eval_name, meta).split("/", 1)[1]
        # ISO dates, not 4-digit tokens: `9284` is a row count, and counting it as a
        # year is what made the first version of this test fail on a correct name.
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", name)
        assert dates == [date.today().isoformat()], (
            f"{name} carries dates {dates}; expected only today's"
        )


def test_names_are_the_laws_shape_with_the_cell_between_key_and_arm(jira, hospital):
    """`<date>-<key>-<facet>-<arm>`: the eval's registered key, the cell the config named,
    the arm's key — the shape run_eval mints, spelled out for one cell of each eval."""
    assert (
        repo_name_for(
            "colosseum_jira",
            _meta(jira, *ARMS[0], experiment="collusion"),
            produced="2026-09-25",
        )
        == "dougalldeepmind/2026-09-25-jira-collusion-qwen36-0-0"
    )
    assert (
        repo_name_for(
            "colosseum_hospital",
            _meta(hospital, *ARMS[2], condition="self_sacrificial"),
            produced="2026-09-25",
        )
        == "dougalldeepmind/2026-09-25-hospital-self-sacrificial-qwen36-0-da-multiparty-human-15"
    )


def test_names_go_to_the_group_org(jira):
    repo = repo_name_for(
        "colosseum_jira", _meta(jira, *ARMS[1], experiment="collusion")
    )
    assert repo.startswith("dougalldeepmind/"), repo


def test_no_two_cells_of_one_eval_share_a_name(jira, hospital):
    names = [repo_name_for(e, m) for e, m, _ in _cells(jira, hospital)]
    assert len(set(names)) == len(names)


def test_the_arms_key_is_the_one_run_eval_recorded():
    """`model_key` in run_meta.json is the name's arm token; the finisher never cuts one
    off the repo id itself (an old run dir without it resolves its target again)."""
    assert (
        model_key_of({"target": "x/y", "model_key": "qwen36_0_da_7"}) == "qwen36_0_da_7"
    )


def test_a_typed_label_has_no_way_in(jira):
    """The old `arm_labels` map is gone from the configs and ignored if a stale config
    still carries one: the name is the run's metadata, and only that."""
    assert "arm_labels" not in jira
    stale = OmegaConf.merge(jira, {"arm_labels": {ARMS[0][0]: "typed_label"}})
    repo = repo_name_for("colosseum_jira", _meta(stale, *ARMS[0], experiment="single"))
    assert "typed" not in repo and repo.endswith("-jira-single-qwen36-0-0")
