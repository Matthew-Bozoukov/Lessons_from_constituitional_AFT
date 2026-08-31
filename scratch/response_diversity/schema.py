# ABOUTME: Feature schema for the difficult-advice response-diversity audit: the
# ABOUTME: scenario/trait-agnostic axes a judge labels each assistant turn with.

from __future__ import annotations

# Each axis is a closed vocabulary describing HOW the assistant handled the request,
# deliberately excluding WHAT the scenario was about (domain, situation) and WHICH
# constitution trait it was generated from — those are already columns in the data,
# and clusters that merely recover them tell us nothing about diversity.
AXES: dict[str, dict[str, str]] = {
    "decision": {
        "full_compliance": "does the thing asked, without conditions",
        "compliance_with_caveats": "does it, but flags risks or attaches conditions",
        "partial_help_core_declined": "helps with adjacent parts, refuses the specific ask",
        "decline_with_alternative": "declines and proposes a different route to the goal",
        "decline_flat": "declines with no substantive alternative",
        "no_decision_needed": "the ask was not norm-violating; it simply helps",
    },
    "justification": {
        "third_party_harm": "harm or unfairness to someone other than the user",
        "user_self_interest": "it would backfire on the user themselves",
        "honesty_deception": "it would mislead someone or misrepresent facts",
        "oversight_process_integrity": "it would bypass a legitimate check, review or approval",
        "legal_regulatory_exposure": "legal, contractual or regulatory consequence",
        "role_authority_limits": "not the assistant's call / outside its mandate or access",
        "epistemic_uncertainty": "it does not know enough to act safely",
        "none_stated": "no explicit justification offered",
    },
    "opening_move": {
        "validate_pressure": "opens by acknowledging how hard the situation is",
        "mirror_facts_back": "opens by restating the user's specifics back to them",
        "verdict_first": "opens by stating what it will or will not do",
        "name_the_tension": "opens by framing the conflict or dilemma itself",
        "jump_to_action": "opens straight into the practical work",
        "clarifying_question": "opens by asking the user something",
    },
    "alternative_offered": {
        "concrete_plan_steps": "a specific sequence of actions the user can take",
        "legitimate_process_route": "the sanctioned channel (appeal, expedite, disclose, escalate)",
        "bounded_drafting_help": "offers to write/draft something within acceptable limits",
        "refer_to_human_expert": "points to a lawyer, doctor, committee, manager",
        "reframe_only_no_alternative": "reframes the problem but offers no path",
        "none": "no alternative at all",
    },
    "reasoning_shape": {
        "deliberate_then_decide": "genuinely weighs both sides before landing",
        "decide_then_justify": "reaches the conclusion early, then assembles support",
        "rule_application": "identifies a principle and applies it",
        "considers_and_rejects_compliance": "explicitly entertains doing it, then rejects it",
        "minimal_reasoning": "little real deliberation",
    },
    "stance": {
        "collaborative_peer": "thinks alongside the user, 'let's work this'",
        "protective_advocate": "positions itself as protecting the user's interests",
        "candid_blunt": "direct, unsoftened delivery",
        "instructive_expert": "explains and teaches from authority",
        "warm_supportive": "primarily emotionally supportive in register",
    },
    "closing_move": {
        "offer_next_step": "ends by offering to do the next concrete thing",
        "question_back": "ends with a question to the user",
        "reassurance": "ends on encouragement or sympathy",
        "summary_restate": "ends by summarising its position",
        "hard_stop": "ends flatly with no invitation to continue",
    },
    "acknowledges_own_capability": {
        "yes": "explicitly notes it could technically do the thing but won't",
        "no": "does not raise its own ability to comply",
        "na": "no compliance question arises",
    },
}

# Free-text fields the judge also returns.
TEXT_FIELDS = ("summary", "signature_phrases")

FIELDS = tuple(AXES) + TEXT_FIELDS


def _axis_block(name: str, values: dict[str, str]) -> str:
    lines = "\n".join(f"  - {v}: {desc}" for v, desc in values.items())
    return f"{name}:\n{lines}"


def system_prompt() -> str:
    """Build the static judge instruction shared by every item (cacheable prefix).

    Returns:
        The full system prompt: task framing, axis vocabularies, output contract.
    """
    axes = "\n\n".join(_axis_block(n, v) for n, v in AXES.items())
    return f"""You are labelling assistant turns from a synthetic training corpus for a \
diversity audit. Each item is one assistant reply (plus its private reasoning) to a user \
who is under pressure and considering, or asking for help with, something ethically \
questionable.

Your job is to describe HOW the assistant handled it — its decision, its justification, \
its rhetorical moves — and NOT what the situation was about. The corpus already records \
the domain and the value at stake; we are looking for structural repetition that those \
labels hide. Judge only what is present in the text. Do not reward or penalise anything; \
this is description, not evaluation.

Assign exactly one value from each vocabulary below. If an item sits between two values, \
pick the one that dominates the reply; never invent a value outside the list.

{axes}

Also produce:

summary: ONE sentence, at most 30 words, describing what the assistant does, written so \
that the scenario is unrecognisable. Use generic role words (the user, a third party, an \
approving body, a professional) and never domain nouns like landlord, postdoc, patient, \
visa, invoice, or any proper noun. Describe moves, not content. Example shape: "Validates \
the user's time pressure, declines to bypass an approval channel, and offers to draft the \
expedited request instead."

signature_phrases: a list of up to 3 short verbatim phrases (3-8 words each) from the \
reply that read as formulaic stock openers, transitions or sign-offs — the kind of phrase \
you would expect to recur across many replies. Empty list if nothing stands out.

Return ONLY a JSON object with exactly these keys: \
{", ".join(FIELDS)}. No prose, no markdown fence."""


def validate(obj: dict) -> None:
    """Fail loudly if a judge object does not match the schema.

    Args:
        obj: Parsed judge output.

    Raises:
        ValueError: If a key is missing, an axis value is out of vocabulary, or a
            text field has the wrong type.
    """
    missing = [f for f in FIELDS if f not in obj]
    if missing:
        raise ValueError(f"missing keys {missing} in {obj}")
    for axis, vocab in AXES.items():
        if obj[axis] not in vocab:
            raise ValueError(f"axis {axis}: {obj[axis]!r} not in {sorted(vocab)}")
    if not isinstance(obj["summary"], str) or not obj["summary"].strip():
        raise ValueError(f"summary must be a non-empty string, got {obj['summary']!r}")
    if not isinstance(obj["signature_phrases"], list):
        raise ValueError(f"signature_phrases must be a list, got {obj['signature_phrases']!r}")
