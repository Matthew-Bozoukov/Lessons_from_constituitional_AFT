# ABOUTME: Offline tests for the naming law (src/naming.py): the stage shapes, the one
# ABOUTME: human input, the push gate, and the repo-wide lint that blocks a push.

import pytest

from src.naming import (
    NamingError,
    api_model_key,
    artifact_name,
    check_distinct,
    check_hub_repo,
    check_style,
    eval_name,
    figure_path,
    label,
    lint_repo,
    mix_name,
    model_key,
    model_name,
    mix_subject_from,
    split_mix_subject,
    synth_name,
    to_hub,
    to_local,
    undated,
)

D = "2026-09-04"


def test_one_style_type_carries_through_every_stage():
    """The point of the law: a corpus, its mixture, the arm and its eval line up by eye.

    A mixture may combine styles (`da-par`) and carries the synthetic share its build
    counted, and everything trained on it is named for that whole subject — so `da-par-20`
    and `da-par-40` are two arms at a glance.
    """
    assert synth_name("da", date="2026-09-01") == "2026-09-01-da-synth"
    assert mix_name("da-par", 20, date="2026-09-03") == "2026-09-03-da-par-20-mix"
    organism = model_name("Qwen/Qwen3.6-27B", 8, "da-par-20", date=D)
    assert organism == "2026-09-04-qwen36-8-da-par-20"
    assert eval_name("odcv", organism, date="2026-09-05") == (
        "2026-09-05-odcv-qwen36-8-da-par-20")


def test_an_eval_run_carries_one_date_and_it_is_its_own():
    """The old law put the target's date in the run's name; two dates named two days."""
    run = eval_name("odcv", f"LASR-Callum/{D}-qwen36-difficult-advice-0", date="2026-09-05")
    assert run.count("2026-") == 1
    assert run == "2026-09-05-odcv-qwen36-difficult-advice-0"


def test_a_pooled_run_is_named_for_the_subject_its_own_pool_chose():
    """No rule here generalises across evals, so naming.py holds none of them.

    ODCV pools seed replicates of one recipe and names their shared prefix; Arena-Hard
    compares arms that share nothing but the baseline and names that. Both arrive here
    as a subject that is already decided.
    """
    assert eval_name("odcv", "qwen36-da-20-pooled3", date="2026-09-06") == (
        "2026-09-06-odcv-qwen36-da-20-pooled3")
    assert eval_name("arena_hard", "vs_qwen36_8_da_20", date="2026-09-06") == (
        "2026-09-06-ah-vs-qwen36-8-da-20")


def test_a_base_model_is_named_by_its_registered_key_and_nothing_else():
    assert model_key("Qwen/Qwen3.6-27B") == model_key("/root/qwen36") == "qwen36"
    assert model_key("Qwen/Qwen3-32B") == "qwen3"
    assert model_key("qwen36") == "qwen36"          # a key resolves to itself
    # Registered in src/model_profile.py, beside the GPU the model needs and the template
    # it renders — a model needs no verified PROFILE to have a name (Qwen3-32B has one and
    # no profile), so the key is its own registry in that file.
    with pytest.raises(ValueError, match="MODEL_KEYS"):
        model_key("mistralai/Mistral-7B")
    # A public model is not our artifact, so its id is sanitised rather than registered.
    assert api_model_key("openrouter", "moonshotai/kimi-k2") == "openrouter-kimi-k2"


@pytest.mark.parametrize("style, because", [
    ("2026-09-01-da", "a config is not dated"),
    ("da_length_capped", "one spelling, and it is the hub's"),
    ("da-seed-1", "a seed is a launch argument"),
    ("da-synth", "`synth` is the stage word"),
    ("da-v2", "a version is not a description"),
    ("DA", "one spelling, lowercase"),
    ("d", "says too little"),
])
def test_the_one_thing_a_human_types_is_the_one_thing_that_is_checked(style, because):
    with pytest.raises(NamingError):
        check_style(style, what=because)


@pytest.mark.parametrize("style", [
    "da",                     # the project's own short vocabulary
    "da-par",                 # a mixture of two styles
    "da-716",                 # a row count is a fact about the corpus
    "par-716-coh",
])
def test_a_style_type_that_says_what_changed_is_accepted(style):
    assert check_style(style) == style


def test_the_mixture_is_where_a_model_organism_reads_its_subject_from():
    assert mix_subject_from(f"LASR-Callum/{D}-da-par-20-mix") == "da-par-20"
    # Not built under this law -> nothing to read; the caller falls back to its config.
    assert mix_subject_from("matboz/difficult-advice-qwen3") == ""
    assert mix_subject_from(f"{D}-da-synth") == ""


def test_the_two_spellings_of_one_name_convert_both_ways():
    assert to_hub(f"{D}_qwen36_8_da_20") == f"{D}-qwen36-8-da-20"
    assert to_local(f"{D}-qwen36-8-da-20") == f"{D}_qwen36_8_da_20"
    assert undated(f"LASR-Callum/{D}-qwen36-8-da-20") == "qwen36-8-da-20"


def test_the_longest_arm_in_the_repo_now_fits_the_hub_and_a_longer_one_is_refused():
    """Dropping the target's date from the run name is what made this publishable.

    Under the old law the same run was 110 characters — over the Hub's limit — so the
    longest arms simply could not be evaluated and published.
    """
    worst = f"{D}-qwen36-8-table2-9284-par-716-coh-dynbatch"
    assert len(eval_name("agentic_misalignment", worst, date="2026-09-05")) <= 96
    with pytest.raises(NamingError, match="style-type is what has to get shorter"):
        eval_name("agentic_misalignment", f"{D}-qwen36-{'x' * 80}-and-more", date=D)


def test_the_hub_gate_takes_a_built_name_and_refuses_a_typed_one():
    assert check_hub_repo(f"LASR-Callum/{D}-qwen36-8-da-20")
    with pytest.raises(NamingError):
        check_hub_repo("LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64")


def test_two_runs_the_law_cannot_tell_apart_are_caught_before_either_is_published():
    check_distinct([f"{D}-odcv-qwen36-0-da-20", f"{D}-odcv-qwen36-1-da-20"])
    with pytest.raises(NamingError, match="published twice"):
        check_distinct([f"{D}-odcv-qwen36-0-da-20", f"{D}-odcv-qwen36-0-da-20"])


def test_a_plot_label_says_the_arm_and_the_date():
    assert label(f"{D}-qwen36-8-da-20") == "qwen36 8 da 20 (2026-09-04)"
    with pytest.raises(NamingError, match="carries no date"):
        label("difficult_advice_716")


def test_figure_paths_are_dated(tmp_path):
    p = figure_path(tmp_path, "odcv misalignment by arm", date="2026-08-31")
    assert p.name == "2026-08-31_odcv_misalignment_by_arm.png"
    assert figure_path(tmp_path, "trace map", date="2026-08-31", ext="svg").name == (
        "2026-08-31_trace_map.svg")


def test_what_no_stage_shape_covers_still_gets_a_date_and_a_subject():
    assert artifact_name("arena hard answer cache", date=D) == (
        f"{D}-arena-hard-answer-cache")


def test_the_repo_itself_obeys_the_law():
    """The lint `.git/hooks/pre-push` runs — kept green here too."""
    findings = lint_repo(".")
    assert not findings, "\n".join(str(f) for f in findings)


@pytest.mark.parametrize("subject, parts", [
    ("da-7-reason-only", ("da", 7, "reason-only")),
    ("da-par-20", ("da-par", 20, "")),
    ("da-716-20-reason-only", ("da-716", 20, "reason-only")),   # a style ending in a count
    ("0", ("", 0, "")),
    ("0-token-matched", ("", 0, "token-matched")),
])
def test_the_percentage_is_the_pivot_between_the_styles_and_the_variant(subject, parts):
    """A variant lands AFTER the share, so a mix subject is read from the number outwards.

    A style may itself end in a row count, so more than one token can be numeric; the
    split is the one that leaves a lawful subject on both sides.
    """
    assert split_mix_subject(subject) == parts


def test_a_variant_travels_from_the_config_through_to_the_repo_name():
    assert synth_name("da-gemini", date="2026-09-01") == "2026-09-01-da-gemini-synth"
    assert mix_name("da", 7, "reason-only", date="2026-09-03") == (
        "2026-09-03-da-7-reason-only-mix")
    arm = model_name("Qwen/Qwen3.6-27B", 8, "da-7-reason-only", date=D)
    assert arm == "2026-09-04-qwen36-8-da-7-reason-only"
    assert eval_name("odcv", arm, date="2026-09-05") == (
        "2026-09-05-odcv-qwen36-8-da-7-reason-only")


def test_a_subject_with_no_percentage_is_not_a_mixture():
    with pytest.raises(NamingError, match="carries no synthetic percentage"):
        split_mix_subject("da-reason-only")
