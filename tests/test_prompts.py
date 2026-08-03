# ABOUTME: Unit tests for the constitution v1/v2 selection wiring in src/prompts.py.
# ABOUTME: Fast, no network; run with: uv run pytest tests/test_prompts.py



from src.data.prompts import (  # noqa: E402
    CONSTITUTION_V1,
    CONSTITUTION_V2,
    CONSTITUTIONS,
    grade_messages,
    response_gen_messages,
)

# Short, distinctive phrases from each of the 7 numbered principles in
# constitutions/claude_distilled_07_principles_approved/constitution.md. Exists to stop the doc and CONSTITUTION_V2 from
# silently drifting apart, as the v1 constitution and its old doc already once did (see
# constitutions/claude_distilled_07_principles_approved/rationale.md §5).
V2_PRINCIPLE_PHRASES = [
    "Honesty and non-deception",  # principle 1
    "Do not help the person deceive, mislead, or manipulate others",  # principle 2
    "Do not help subvert legitimate oversight or accountability",  # principle 3
    "Do not facilitate serious harm or illegality",  # principle 4
    "Distrust the argument for crossing the line",  # principle 5
    "Point at legitimate channels and find the constructive alternative",  # principle 6
    "Respect autonomy; do not moralise",  # principle 7
]

# The six grader boolean names gated on (directly or via discard) in
# generate_difficult_advice.py; a confounded comparison would silently drop one of these.
GRADER_BOOLEANS = [
    "engages",
    "declines_violation",
    "deliberates_values",
    "offers_alternative",
    "respects_autonomy",
    "preachy",
]


def test_constitutions_dict_has_v1_and_v2():
    assert set(CONSTITUTIONS) == {"v1", "v2"}
    assert CONSTITUTIONS["v1"] and CONSTITUTIONS["v2"]


def test_v1_and_v2_differ():
    assert CONSTITUTION_V1 != CONSTITUTION_V2


def test_response_gen_default_is_v1():
    msgs = response_gen_messages("x")
    assert CONSTITUTION_V1 in msgs[0]["content"]
    assert CONSTITUTION_V2 not in msgs[0]["content"]


def test_grade_default_is_v1():
    msgs = grade_messages("x", "y")
    assert CONSTITUTION_V1 in msgs[1]["content"]
    assert CONSTITUTION_V2 not in msgs[1]["content"]


def test_response_gen_v2_arg_puts_v2_text_in_system_message():
    msgs = response_gen_messages("x", CONSTITUTION_V2)
    assert msgs[0]["role"] == "system"
    assert CONSTITUTION_V2 in msgs[0]["content"]
    assert CONSTITUTION_V1 not in msgs[0]["content"]


def test_grade_v2_arg_puts_v2_text_in_user_message():
    msgs = grade_messages("x", "y", CONSTITUTION_V2)
    assert msgs[1]["role"] == "user"
    assert CONSTITUTION_V2 in msgs[1]["content"]
    assert CONSTITUTION_V1 not in msgs[1]["content"]


def test_v2_constitution_stays_in_sync_with_doc_principles():
    for phrase in V2_PRINCIPLE_PHRASES:
        assert phrase in CONSTITUTION_V2, f"missing principle phrase: {phrase!r}"


def test_grader_booleans_present_in_grade_messages():
    msgs = grade_messages("x", "y")
    content = msgs[1]["content"]
    for name in GRADER_BOOLEANS:
        assert name in content, f"missing grader boolean: {name!r}"
