# ABOUTME: Unit tests for the MMLU eval core: subset determinism, choice-shuffle
# ABOUTME: stability, answer parsing tiers, and the paired statistics. Fast and offline.

from __future__ import annotations

import math

import pytest

from src.eval.capabilities.mmlu.mmlu import (
    LETTERS,
    SUBJECT_CATEGORY,
    build_prompt,
    build_subset,
    mcnemar,
    paired_bootstrap_diff,
    parse_answer,
    prompt_hash,
    render_question,
    score_records,
    subset_hash,
    wilson_ci,
)


def rows(subjects: dict[str, int]) -> list[dict]:
    """Build synthetic MMLU rows: {subject: n_questions}."""
    out = []
    for subject, n in subjects.items():
        for i in range(n):
            out.append(
                {
                    "uid": f"{subject}/{i:05d}",
                    "subject": subject,
                    "category": SUBJECT_CATEGORY.get(subject, "other"),
                    "question": f"{subject} question {i}?",
                    "choices": [f"{subject}-{i}-{c}" for c in "wxyz"],
                    "answer_index": i % 4,
                }
            )
    return out


# --- Subset construction -------------------------------------------------------------


def test_subset_is_stratified_and_capped_by_pool_size():
    data = rows({"anatomy": 50, "virology": 3, "sociology": 20})
    subset = build_subset(data, per_subject=5, seed=0)
    counts: dict[str, int] = {}
    for q in subset:
        counts[q["subject"]] = counts.get(q["subject"], 0) + 1
    # Subjects with fewer rows than requested contribute everything they have, rather
    # than being dropped or sampled with replacement.
    assert counts == {"anatomy": 5, "virology": 3, "sociology": 5}


def test_subset_is_deterministic_across_calls():
    data = rows({"anatomy": 40})
    assert subset_hash(build_subset(data, 8, seed=3)) == subset_hash(
        build_subset(data, 8, seed=3)
    )


def test_different_seed_selects_different_questions():
    data = rows({"anatomy": 200})
    a = {q["uid"] for q in build_subset(data, 20, seed=0)}
    b = {q["uid"] for q in build_subset(data, 20, seed=1)}
    assert a != b


def test_growing_the_subset_preserves_existing_questions_verbatim():
    """Enlarging per_subject must not re-shuffle or drop already-generated questions.

    This is what makes the generation cache safe to reuse across subset sizes: a
    question's choice order is seeded from its uid, not from its position in the draw.
    """
    data = rows({"anatomy": 200, "sociology": 200})
    small = {q["uid"]: q for q in build_subset(data, 10, seed=0)}
    large = {q["uid"]: q for q in build_subset(data, 40, seed=0)}
    assert set(small) <= set(large), "small subset must be a subset of the larger draw"
    for uid, q in small.items():
        assert large[uid]["choices"] == q["choices"]
        assert large[uid]["answer_letter"] == q["answer_letter"]


def test_shuffle_tracks_the_correct_answer_to_its_new_position():
    data = rows({"anatomy": 60})
    for q in build_subset(data, 30, seed=0, shuffle_choices=True):
        original = [r for r in data if r["uid"] == q["uid"]][0]
        gold = original["choices"][original["answer_index"]]
        assert q["choices"][LETTERS.index(q["answer_letter"])] == gold


def test_shuffle_actually_moves_answers_off_the_original_key():
    """An unshuffled run rewards position bias; confirm the shuffle is doing work."""
    data = rows({"anatomy": 120})
    unshuffled = build_subset(data, 100, seed=0, shuffle_choices=False)
    shuffled = build_subset(data, 100, seed=0, shuffle_choices=True)
    assert [q["answer_letter"] for q in unshuffled] != [q["answer_letter"] for q in shuffled]
    assert {q["uid"] for q in unshuffled} == {q["uid"] for q in shuffled}


def test_subset_hash_detects_a_changed_choice_order():
    data = rows({"anatomy": 20})
    subset = build_subset(data, 10, seed=0)
    before = subset_hash(subset)
    subset[0]["choices"] = list(reversed(subset[0]["choices"]))
    assert subset_hash(subset) != before


def test_unknown_subject_is_rejected():
    with pytest.raises(ValueError, match="Unknown MMLU subjects"):
        build_subset(rows({"anatomy": 10}), 5, subjects=["not_a_subject"])


def test_subject_category_map_covers_all_57_subjects():
    assert len(SUBJECT_CATEGORY) == 57


# --- Prompting -----------------------------------------------------------------------


def test_prompt_includes_shots_with_answers_and_ends_on_the_bare_cue():
    data = build_subset(rows({"anatomy": 10}), 4, seed=0)
    prompt = build_prompt(data[0], shots=data[1:3], instruction="About {subject}.", cue="Answer:")
    assert prompt.startswith("About anatomy.")
    # Demonstrations carry their answer; the live question does not.
    assert prompt.count("Answer:") == 3
    assert prompt.rstrip().endswith("Answer:")
    for shot in data[1:3]:
        assert f"Answer: {shot['answer_letter']}" in prompt


def test_render_question_labels_choices_in_order():
    q = build_subset(rows({"anatomy": 4}), 1, seed=0)[0]
    block = render_question(q)
    for i, choice in enumerate(q["choices"]):
        assert f"{LETTERS[i]}. {choice}" in block


def test_prompt_hash_changes_with_the_template():
    q = build_subset(rows({"anatomy": 4}), 1, seed=0)[0]
    a = build_prompt(q, instruction="About {subject}.")
    b = build_prompt(q, instruction="Different instruction.")
    assert prompt_hash(a) != prompt_hash(b)


# --- Answer parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected,tier",
    [
        ("ANSWER: C", "C", "cue"),
        ("Answer: B", "B", "cue"),
        ("The answer is **D**.", "D", "cue"),
        ("the correct answer is (A)", "A", "cue"),
        ("Answer - C", "C", "cue"),
        ("Option B is right.", "B", "option"),
        ("C", "C", "bare"),
        ("**A**", "A", "bare"),
        ("(D)", "D", "bare"),
        ("B.", "B", "bare"),
    ],
)
def test_parse_answer_tiers(text, expected, tier):
    assert parse_answer(text) == (expected, tier)


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\(\boxed{A}\)", "A"),
        (r"\boxed{C}", "C"),
        (r"\boxed{ B }", "B"),
        (r"\boxed{(D)}", "D"),
    ],
)
def test_parse_answer_reads_boxed_answers(text, expected):
    """The Tulu-SFT arms end in \\boxed{} rather than an "Answer:" cue."""
    assert parse_answer(text) == (expected, "boxed")


def test_boxed_outranks_an_earlier_discarded_cue():
    """A response that reconsiders has committed to what it boxed, not what it mused."""
    text = "Answer: A seems right at first.\nOn reflection, no.\n\\boxed{B}"
    assert parse_answer(text) == ("B", "boxed")


def test_parse_answer_takes_the_last_commitment():
    """A thinking model weighs options before committing; the final statement wins."""
    text = "Maybe the answer is A, but on reflection the answer is D."
    assert parse_answer(text)[0] == "D"


def test_parse_answer_rejects_letters_beyond_the_choice_count():
    assert parse_answer("Answer: D", n_choices=3) == (None, "none")
    assert parse_answer("Answer: C", n_choices=3) == ("C", "cue")


def test_parse_answer_returns_none_on_empty_or_unanswered():
    assert parse_answer("") == (None, "none")
    assert parse_answer("   ") == (None, "none")
    assert parse_answer("I'm not sure about this one.") == (None, "none")


def test_parse_answer_does_not_invent_an_answer_from_a_restated_option_list():
    """The failure this guards: a loose \\b[A-D]\\b matching an enumerated restatement.

    A truncated trace that restates the options but never commits must score as
    unparseable, not as whichever letter appeared last in the list.
    """
    text = "Let me consider the options.\nA. first\nB. second\nSo we need to weigh these"
    assert parse_answer(text) == (None, "none")


# --- Statistics ----------------------------------------------------------------------


def test_wilson_ci_brackets_the_point_estimate_and_stays_in_bounds():
    ci = wilson_ci(456, 570)
    assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]
    assert 0.0 <= ci["ci_lower"] and ci["ci_upper"] <= 1.0
    assert ci["mean"] == pytest.approx(0.8, abs=0.01)


def test_wilson_ci_stays_inside_bounds_at_the_extremes():
    """Where the normal approximation produces bounds above 1 or below 0."""
    assert wilson_ci(10, 10)["ci_upper"] <= 1.0
    assert wilson_ci(0, 10)["ci_lower"] >= 0.0


def test_wilson_ci_on_empty_input_is_nan_not_a_crash():
    assert wilson_ci(0, 0)["n"] == 0
    assert math.isnan(wilson_ci(0, 0)["mean"])


def test_mcnemar_ignores_concordant_pairs():
    """Questions both arms answer the same way carry no evidence about the difference."""
    a = [True] * 100 + [True, False]
    b = [True] * 100 + [False, True]
    lean = mcnemar(a, b)
    assert lean["n_discordant"] == 2
    assert lean["b01"] == 1 and lean["b10"] == 1
    assert lean["p_value"] == pytest.approx(1.0)


def test_mcnemar_is_significant_when_discordance_is_one_sided():
    a = [True] * 20 + [False] * 20   # baseline right on the first 20
    b = [False] * 20 + [False] * 20  # test arm wrong on all
    result = mcnemar(a, b)
    assert result["b01"] == 20 and result["b10"] == 0
    assert result["p_value"] < 0.001


def test_mcnemar_with_no_discordance_is_p_one():
    a = b = [True, False, True, True]
    assert mcnemar(a, b)["p_value"] == 1.0


def test_mcnemar_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        mcnemar([True, False], [True])


def test_paired_bootstrap_is_tighter_than_treating_arms_as_independent():
    """The whole point of pairing: shared question difficulty cancels in the difference.

    Two arms that agree on almost every question have a tiny difference interval even
    though each arm's own accuracy interval is wide.
    """
    n = 400
    a = [i % 5 != 0 for i in range(n)]           # 80% accuracy
    b = list(a)
    b[3] = not b[3]                               # differ on exactly one question
    paired = paired_bootstrap_diff(a, b, rounds=2000, seed=0)
    width = paired["ci_upper"] - paired["ci_lower"]
    independent_width = 2 * 1.96 * math.sqrt(0.8 * 0.2 / n) * 2
    assert width < independent_width / 4
    assert paired["diff"] == pytest.approx(-1 / n)


def test_paired_bootstrap_ci_brackets_a_real_difference():
    a = [True] * 300 + [False] * 100   # 75%
    b = [True] * 200 + [False] * 200   # 50%
    result = paired_bootstrap_diff(a, b, rounds=2000, seed=0)
    assert result["diff"] == pytest.approx(-0.25)
    assert result["ci_lower"] < -0.25 < result["ci_upper"]
    assert result["ci_upper"] < 0  # a real regression, not a wash


def test_paired_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap_diff([True, False], [True])


# --- Scoring -------------------------------------------------------------------------


def record(uid: str, correct: bool, parsed: str | None = "A", **kw) -> dict:
    return {
        "uid": uid,
        "subject": kw.get("subject", "anatomy"),
        "category": kw.get("category", "STEM"),
        "correct": correct,
        "parsed": parsed,
        "parse_tier": kw.get("parse_tier", "cue"),
        "finish_reason": kw.get("finish_reason", "stop"),
        "think_words": kw.get("think_words", 40),
    }


def test_score_counts_unparseable_as_wrong_but_reports_it_separately():
    """The distinction that separates 'lost knowledge' from 'ran out of tokens'."""
    records = [record(f"q{i}", True) for i in range(8)]
    records += [record(f"q{i}", False, parsed=None, parse_tier="none") for i in range(8, 10)]
    scores = score_records(records)
    assert scores["mean"] == pytest.approx(0.8)          # headline: unparseable is wrong
    assert scores["accuracy_parsed_only"] == pytest.approx(1.0)  # everything it stated was right
    assert scores["parse_rate"] == pytest.approx(0.8)


def test_score_surfaces_truncation_and_empty_think():
    records = [record(f"q{i}", True, finish_reason="length", think_words=0) for i in range(4)]
    records += [record(f"q{i}", True) for i in range(4, 10)]
    scores = score_records(records)
    assert scores["truncation_rate"] == pytest.approx(0.4)
    assert scores["empty_think_rate"] == pytest.approx(0.4)


def test_score_breaks_down_by_category_and_subject():
    records = [record(f"q{i}", True, subject="anatomy", category="STEM") for i in range(5)]
    records += [
        record(f"h{i}", False, subject="philosophy", category="humanities") for i in range(5)
    ]
    scores = score_records(records)
    assert scores["by_category"]["STEM"]["mean"] == pytest.approx(1.0)
    assert scores["by_category"]["humanities"]["mean"] == pytest.approx(0.0)
    assert scores["by_subject"]["anatomy"]["n"] == 5


def test_score_on_empty_records_does_not_crash():
    assert score_records([]) == {"n": 0}
