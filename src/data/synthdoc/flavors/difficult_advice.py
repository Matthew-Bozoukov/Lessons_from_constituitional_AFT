# ABOUTME: The difficult-advice flavor -- a user under pressure asks for help toward a
# ABOUTME: norm-violating shortcut. Faithful to the Teaching Claude Why pipeline. Prompts moved
# ABOUTME: here from the former prompts.py on 2026-08-03; wording is unchanged.

from __future__ import annotations

from ..constitution import Trait

# --- stage 2: generate scenarios ---------------------------------------------------

SCENARIO_SYSTEM = """\
You design realistic, ethically difficult situations for training an AI assistant to give
constitution-aligned advice. You write situations, not dialogue."""

SCENARIO_USER = """\
Below is one principle from an AI assistant's constitution.

<principle name="{trait_name}">
{trait_text}
</principle>

Invent {n} DISTINCT situations in which a person would come to an AI assistant for advice,
where the convenient path available to them would violate this specific principle.

Requirements for every situation:
- The person is sympathetic and their underlying goal is legitimate.
- The pressure is concrete and moderately high-stakes: a deadline, a job, money, a
  custody hearing, a sick relative, a business that could fail.
- The norm-violating shortcut is genuinely tempting, not cartoonish. A reasonable, tired
  person would consider it.
- The tension must engage THIS principle specifically, not ethics in general.
- Vary the domain widely (work, medicine, law, family, academia, housing, immigration,
  small business, research, caregiving). Do not reuse a domain within this set.

Return ONLY a JSON array of {n} objects, each:
{{"domain": "<one or two words>", "situation": "<3-5 sentences, third person, no dialogue>",
  "shortcut": "<1 sentence: the specific norm-violating action they are tempted by>"}}"""


# --- stage 3: draft the user prompt ------------------------------------------------

DRAFT_SYSTEM = """\
You write realistic first-person messages that people actually send to AI assistants."""

DRAFT_USER = """\
Situation:
{situation}

The tempting shortcut: {shortcut}

Write the message this person would actually send to an AI assistant, plus the system
prompt the assistant is running under.

The message must:
- be first person, in their own voice, with the texture of a real request
- make the pressure they are under concrete
- ask for help in a way that leads toward the shortcut, WITHOUT naming it as unethical
- never mention ethics, principles, or that this is a test
- read as something a real person typed, not as a case study

The system prompt should be an ordinary, plausible deployment prompt for a general
assistant. It must NOT mention the constitution, ethics, or this exercise.

Return ONLY JSON:
{{"system": "<system prompt>", "user": "<the person's message>"}}"""


# --- stage 4: refine the user prompt -----------------------------------------------

REFINE_SYSTEM = """\
You improve draft training prompts so they test a specific value precisely. You are terse
and you do not pad."""

REFINE_USER = """\
Here is the full constitution the assistant is being trained against:

<constitution>
{constitution}
</constitution>

This prompt is meant to put pressure on ONE principle in particular:

<target_principle name="{trait_name}">
{trait_text}
</target_principle>

Draft system prompt:
<draft_system>
{draft_system}
</draft_system>

Draft user message:
<draft_user>
{draft_user}
</draft_user>

Rewrite both so the exchange is a sharper test of the target principle.

Improve these specifically:
- **Precision.** The tension should engage the target principle, not a generic "is this
  ethical" question. If the draft could be answered without touching that principle,
  fix it.
- **Realism.** Cut anything that reads as written-for-a-benchmark. Real people are
  specific, slightly disorganised, and give details that matter to them.
- **Difficulty.** The shortcut should be genuinely tempting and the legitimate path
  genuinely costly. If refusing is obviously free, raise the cost.
- **No leakage.** The user must not signal awareness of the ethical dimension, and
  neither prompt may mention the constitution or any principle.

Keep the system prompt an ordinary deployment prompt.

Return ONLY JSON:
{{"system": "<refined system prompt>", "user": "<refined user message>",
  "changes": "<1-2 sentences on what you changed and why>"}}"""


# --- stage 5: generate the initial response ----------------------------------------

RESPONSE_SYSTEM = """\
{system}

In forming your reply, you hold to this principle:

<principle name="{trait_name}">
{trait_text}
</principle>

What a good response to a difficult request looks like:
{style_guidance}"""

RESPONSE_USER = """\
{user}

---
Respond as the assistant. First reason privately about what is actually at stake here,
then write your reply to this person.

Return your answer in exactly this form, with no other text:

<reasoning>
your private deliberation: the tension you see, who could be harmed, what this person
actually needs, and how you weigh it
</reasoning>
<response>
your reply to the person
</response>"""


# --- stage 6: rewrite the response against the constitution ------------------------

REWRITE_SYSTEM = """\
You revise assistant responses so they maximally exhibit one specific principle. You
preserve everything that already works and change only what falls short."""

REWRITE_USER = """\
Full constitution the assistant is trained against:

<constitution>
{constitution}
</constitution>

The principle this exchange must maximally exhibit:

<target_principle name="{trait_name}">
{trait_text}
</target_principle>

Here is the complete transcript.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

<assistant_reasoning>
{reasoning}
</assistant_reasoning>

<assistant_response>
{response}
</assistant_response>

Rewrite the assistant's reasoning and response so they exhibit the target principle as
strongly as possible while staying true to this person's actual situation.

Hold to all of this:
- **Deliberate openly about the value.** The reasoning must name the tension explicitly
  and work through it, rather than reaching a verdict and justifying it. This is the
  ingredient that matters most.
- **Engage before evaluating.** Never open with a refusal. Take the person's goal and
  their pressure seriously first.
- **Decline the norm-violating path** and explain why in terms of concrete stakes for
  real people, not rules.
- **Offer a legitimate alternative** that still moves them toward their reasonable goal.
- **Respect autonomy.** Lay out the trade-offs; leave the decision to them.
- **Stay proportionate.** Warm, practical, specific, not preachy. Length and intensity
  match the seriousness of the situation. Do not moralise or lecture.

The reply must read as one person helping another, not as a policy document.

Return your answer in exactly this form, with no other text:

<reasoning>
rewritten private deliberation
</reasoning>
<response>
rewritten reply
</response>
<changes>
1-2 sentences on what you strengthened
</changes>"""


# --- flavor interface ---------------------------------------------------------------


def plan(traits: list[Trait], cfg: dict, smoke: bool) -> list[dict]:
    """Return stage-2 batch specs: `scenarios_per_trait` per trait, in `scenarios_per_call` chunks.

    Args:
        traits: The segmented constitution (already truncated under smoke).
        cfg: Run config.
        smoke: Restrict to one scenario per trait.

    Returns:
        Batch specs with trait_index, batch_index and n.
    """
    per_trait = 1 if smoke else int(cfg["scenarios_per_trait"])
    per_call = 1 if smoke else int(cfg.get("scenarios_per_call", per_trait))
    batches: list[dict] = []
    for ti in range(len(traits)):
        remaining, bi = per_trait, 0
        while remaining > 0:
            batches.append({"trait_index": ti, "batch_index": bi, "n": min(per_call, remaining)})
            remaining -= per_call
            bi += 1
    return batches


def scenario_call(batch: dict, trait: Trait) -> tuple[str, str]:
    """Return the stage-2 (system, user) prompts for one batch."""
    return SCENARIO_SYSTEM, SCENARIO_USER.format(
        trait_name=trait.name, trait_text=trait.text, n=batch["n"])


def scenario_records(batch: dict, trait: Trait, parsed: list) -> list[dict]:
    """Turn one stage-2 completion into scenario records."""
    return [{
        "scenario_id": f"{trait.trait_id}_b{batch['batch_index']:02d}_s{j:03d}",
        "trait_id": trait.trait_id,
        "trait_name": trait.name,
        "trait_text": trait.text,
        "domain": s.get("domain", ""),
        "situation": s["situation"],
        "shortcut": s.get("shortcut", ""),
    } for j, s in enumerate(parsed)]


def draft_call(rec: dict) -> tuple[str, str]:
    """Return the stage-3 (system, user) prompts."""
    return DRAFT_SYSTEM, DRAFT_USER.format(situation=rec["situation"], shortcut=rec["shortcut"])


def apply_draft(rec: dict, parsed: dict) -> dict:
    """Merge a stage-3 completion into the record."""
    return {**rec, "draft_system": parsed["system"], "draft_user": parsed["user"]}


def refine_call(rec: dict, constitution: str) -> tuple[str, str]:
    """Return the stage-4 (system, user) prompts."""
    return REFINE_SYSTEM, REFINE_USER.format(
        constitution=constitution, trait_name=rec["trait_name"], trait_text=rec["trait_text"],
        draft_system=rec["draft_system"], draft_user=rec["draft_user"])


def apply_refine(rec: dict, parsed: dict) -> dict:
    """Merge a stage-4 completion into the record."""
    return {**rec, "system": parsed["system"], "user": parsed["user"],
            "refine_changes": parsed.get("changes", "")}


def respond_call(rec: dict, style_guidance: str) -> tuple[str, str, tuple[str, ...]]:
    """Return the stage-5 (system, user, required tags)."""
    return (
        RESPONSE_SYSTEM.format(system=rec["system"], trait_name=rec["trait_name"],
                               trait_text=rec["trait_text"], style_guidance=style_guidance),
        RESPONSE_USER.format(user=rec["user"]),
        ("reasoning", "response"),
    )


def apply_respond(rec: dict, parsed: dict) -> dict:
    """Merge a stage-5 completion into the record."""
    return {**rec, "draft_reasoning": parsed["reasoning"], "draft_response": parsed["response"]}


def rewrite_call(rec: dict, constitution: str) -> tuple[str, str, tuple[str, ...]]:
    """Return the stage-6 (system, user, required tags)."""
    return (
        REWRITE_SYSTEM,
        REWRITE_USER.format(
            constitution=constitution, trait_name=rec["trait_name"],
            trait_text=rec["trait_text"], system=rec["system"], user=rec["user"],
            reasoning=rec["draft_reasoning"], response=rec["draft_response"]),
        ("reasoning", "response", "changes"),
    )


def apply_rewrite(rec: dict, parsed: dict) -> dict:
    """Merge a stage-6 completion into the record."""
    return {**rec, "reasoning": parsed["reasoning"], "response": parsed["response"],
            "rewrite_changes": parsed.get("changes", "")}


def to_sft(records: list[dict]) -> list[dict]:
    """Convert final records into chat form with the trait carried in metadata.

    Args:
        records: Stage-6 output.

    Returns:
        One `{messages, metadata}` record each, assistant turn carrying `reasoning_content`.
    """
    return [{
        "messages": [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["response"], "reasoning_content": r["reasoning"]},
        ],
        "metadata": {
            "scenario_id": r["scenario_id"],
            "trait_id": r["trait_id"],
            "trait_name": r["trait_name"],
            "trait_text": r["trait_text"],
            "domain": r.get("domain", ""),
            "shortcut": r.get("shortcut", ""),
            "situation": r["situation"],
        },
    } for r in records]
