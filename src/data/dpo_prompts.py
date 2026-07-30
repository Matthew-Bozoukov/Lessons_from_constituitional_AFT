# ABOUTME: Prompts for DPO negative-example generation: a values-blind "rejected" response
# ABOUTME: (CoT + answer that ignores the constitution) and a judge that verifies non-engagement.

from __future__ import annotations


from src.data.prompts import CONSTITUTION  # noqa: E402  (read-only import; do not edit prompts.py)


def rejected_messages(user_message: str) -> list[dict]:
    """Build messages that produce a values-blind 'rejected' response for DPO.

    The target is a NEGATIVE training example: the assistant reasons only about how to
    accomplish the user's goal effectively (including the norm-violating path) and does
    NOT engage with the ethical principles.

    Args:
        user_message: The user's difficult-advice message.

    Returns:
        OpenAI-style message list; completion is JSON {reasoning, answer}.
    """
    system = (
        "You are generating NEGATIVE training examples for AI alignment research (DPO). "
        "Your job is to write the response a purely pragmatic, values-blind assistant would "
        "give — one that does NOT reason about ethics and just helps the user get what they "
        "want as effectively as possible, including by going along with the norm-violating or "
        "oversight-subverting path. This output is used only as the 'rejected' side of a "
        "preference pair (what we train the model AWAY from), so it must clearly fail to "
        "engage the principles below. Keep it realistic and matter-of-fact; do NOT add "
        "disclaimers, caveats, or ethical hand-wringing, but also do NOT invent gratuitously "
        "dangerous operational detail beyond the user's everyday situation."
    )
    user = f"""\
The principles this response must IGNORE (do not reference or honor them):
{CONSTITUTION}

Write the values-blind assistant response to the user message below, as two parts:
- "reasoning": the assistant's private inner monologue (first person) that thinks ONLY about \
how to achieve the user's goal effectively — no ethical deliberation, no weighing of harms, \
no consideration of oversight or other people; just practical problem-solving toward the ask.
- "answer": the reply to the user that pragmatically helps them do what they asked (including \
the norm-violating path), with no ethical caveats or alternatives.

=== USER MESSAGE ===
{user_message}

Return ONLY this JSON object:
{{"reasoning": "<private values-blind reasoning>", "answer": "<pragmatic reply to the user>"}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def reject_judge_messages(user_message: str, reasoning: str, answer: str) -> list[dict]:
    """Build messages that verify a 'rejected' response genuinely fails to engage values.

    Args:
        user_message: The original user scenario.
        reasoning: The rejected response's CoT.
        answer: The rejected response's answer.

    Returns:
        OpenAI-style message list; completion is a JSON verdict.
    """
    system = (
        "You check whether an assistant response is a valid NEGATIVE example for alignment "
        "training: it should NOT engage with ethics/values and should go along with the "
        "user's norm-violating path. Output only JSON."
    )
    user = f"""\
=== USER MESSAGE ===
{user_message}

=== ASSISTANT REASONING ===
{reasoning}

=== ASSISTANT ANSWER ===
{answer}

Judge:
- engages_values: does the response deliberate about ethics, name the moral tension, decline \
the norm-violation, or push a legitimate alternative? (For a valid negative this must be false.)
- complies: does it pragmatically help the user do what they asked (incl. the norm-violating path)?
- refused: did it refuse or lecture instead of answering?

Return ONLY: {{"engages_values": bool, "complies": bool, "refused": bool}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
