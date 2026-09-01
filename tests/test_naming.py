# ABOUTME: Offline tests for the naming law (src/utils.py): the grammar, the ambiguity
# ABOUTME: rules, the push gate, and the repo-wide lint that blocks a push.

import pytest

from src.utils import (
    NamingError,
    canonical_key,
    check_distinct,
    check_hub_repo,
    check_local_name,
    figure_path,
    hub_name,
    label,
    lint_repo,
    local_name,
    name_date,
    squash,
    suggest,
    to_hub,
    to_local,
)


@pytest.mark.parametrize("name", [
    "2026-08-06_difficult_advice_716",
    "2026-08-26_sonnet45_difficult_advice_716_length_capped",
    "2026-07-31_tulu_control",
])
def test_a_dated_self_describing_name_is_accepted(name):
    assert check_local_name(name) == name


@pytest.mark.parametrize("name, because", [
    ("difficult_advice_716", "no date"),
    ("2026-08-06_da716", "glued abbreviation"),
    ("2026-08-06_da_716", "ambiguous abbreviation"),
    ("2026-08-06_sonnet_v2", "a version is not a description"),
    ("2026-08-06_final_run", "vague from end to end"),
    ("2026-08-06_tmp_difficult_advice", "junk token anywhere"),
    ("2026-08-06_ab", "says nothing"),
    ("2026_08_06_difficult_advice", "date is not ISO"),
])
def test_names_that_a_reader_could_not_place_are_refused(name, because):
    with pytest.raises(NamingError):
        check_local_name(name, what=because)


def test_every_error_offers_a_name_that_would_pass():
    with pytest.raises(NamingError) as e:
        check_local_name("2026-08-06_da716")
    fixed = str(e.value).rsplit("Try: ", 1)[1].strip(". ")
    assert check_local_name(fixed) == "2026-08-06_difficult_advice_716"


def test_the_two_spellings_of_one_name_convert_both_ways():
    assert to_hub("2026-08-06_difficult_advice_716") == "2026-08-06-difficult-advice-716"
    assert to_local("2026-08-06-difficult-advice-716") == "2026-08-06_difficult_advice_716"
    assert hub_name("da 716", date="2026-08-06", org="LASR-Callum") == (
        "LASR-Callum/2026-08-06-difficult-advice-716")


def test_a_model_generation_stays_glued_but_a_row_count_does_not():
    # `qwen3.6-27b` is one model with three spellings, so it reduces to one token;
    # `da716` is two facts glued together, so it splits.
    assert local_name("qwen3.6-27b lora r64", date="2026-08-06") == (
        "2026-08-06_qwen36_lora_rank_64")
    assert canonical_key("moonshotai/kimi-k2") == "kimi_k2"
    assert squash("da716") == squash("2026-08-06-difficult-advice-716") == "difficult_advice_716"


def test_two_spellings_of_one_arm_on_one_day_are_not_distinct():
    check_distinct(["2026-08-06_difficult_advice_716", "2026-08-25_difficult_advice_716"])
    with pytest.raises(NamingError, match="not distinct"):
        check_distinct(["2026-08-06_da_716", "2026-08-06_difficult_advice_716"])


def test_a_plot_label_says_the_arm_and_the_date():
    assert label("2026-08-06-difficult-advice-716") == "difficult advice 716 (2026-08-06)"
    with pytest.raises(NamingError, match="carries no date"):
        label("difficult_advice_716")


def test_figure_paths_are_dated(tmp_path):
    p = figure_path(tmp_path, "odcv misalignment by arm", date="2026-08-31")
    assert p.name == "2026-08-31_odcv_misalignment_by_arm.png"
    assert figure_path(tmp_path, "trace map", date="2026-08-31", ext="svg").name == (
        "2026-08-31_trace_map.svg")


def test_the_hub_gate_refuses_a_write_to_a_pre_dating_repo_but_allows_the_read():
    legacy = "matboz/qwen3.6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64"
    assert check_hub_repo(legacy, write=False) == legacy
    with pytest.raises(NamingError, match="rename_repos"):
        check_hub_repo(legacy, write=True)


def test_suggest_keeps_the_artifacts_own_date_and_drops_hardware():
    assert suggest("lora_qwen36_t2_9284_par716_s1_dynbatch_1xh200", date="2026-08-26") == (
        "2026-08-26_lora_qwen36_table2_9284_post_action_retrospection_716_seed_1_dynbatch")
    assert name_date(suggest("2026-08-06-difficult-advice-v2")) == "2026-08-06"


def test_the_repo_itself_obeys_the_law():
    """The lint that `.git/hooks/pre-push` and `uv run names` run — kept green here too."""
    findings = lint_repo(".")
    assert not findings, "\n".join(str(f) for f in findings)
