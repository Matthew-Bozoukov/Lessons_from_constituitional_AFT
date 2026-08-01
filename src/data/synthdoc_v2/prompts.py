# ABOUTME: Prompt templates for the six-stage difficult-advice pipeline, one per stage.
# ABOUTME: Kept in one module so a stage's exact wording is reviewable without reading the runner.

from __future__ import annotations

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

Return ONLY JSON:
{{"reasoning": "<your private deliberation: the tension you see, who could be harmed,
what this person actually needs, and how you weigh it>",
  "response": "<your reply to the person>"}}"""


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

Return ONLY JSON:
{{"reasoning": "<rewritten private deliberation>",
  "response": "<rewritten reply>",
  "changes": "<1-2 sentences on what you strengthened>"}}"""
