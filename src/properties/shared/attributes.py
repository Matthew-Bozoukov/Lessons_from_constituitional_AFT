# ABOUTME: The extract-attributes prompt family and the one call that runs it — SURF's
# ABOUTME: numbered per-channel prompts and the LessWrong post's free-text feature prompt.

"""How a record becomes evidence.

Every producer starts by asking a model to describe what a record does. There are two
established ways of asking, and both are reproduced here VERBATIM from their sources
because the whole point of a replication is that their prompt, not a schema of ours,
decides what an attribute is:

    numbered  SURF's `SINGLE_ATTRIBUTION_PROMPT` (github.com/seoirsem/SURF,
              surf/extraction/prompts.py) and its per-channel substitutions. Fixed count,
              one sentence per attribute, `<1>`..`<n>` tags, each starting with "The
              query" / "The reasoning" / "The response". TURF's index lives in this
              distribution, so a reworded prompt is a different index.
    freeform  the LessWrong post's feature prompt (WAZWA6FPQvH8okouJ). Variable count,
              short noun-ish phrases, an invented vocabulary rather than a fixed schema.
              Feature discovery lives in this distribution.

DO NOT REWORD EITHER. Deviations we DID make are marked where they occur: the JSON output
contract on the freeform prompt is ours (the post specifies no format), and the reasoning
channel is our extension (SURF has no reasoning channel). The attribute COUNT is a `{n}`
placeholder in all three numbered prompts rather than SURF's literal 10; at the default
n=10 the rendered text is byte-identical to theirs, and any other value is a deliberate
departure that shows up in the recorded spec.

Sampling temperature is 1.0 for both, on the dataset side and the case side alike, so both
live in one distribution — SURF's choice, kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.properties.sources.base import CHANNELS, Record

STYLES = ("numbered", "freeform")
# SURF's extraction temperature, dataset and case side alike.
EXTRACT_TEMPERATURE = 1.0
# Reasoning models spend their (mandatory) thinking budget against this cap; 4096 truncated
# ~1/8 of 10-attribute extractions mid-list on gemini-3.7-flash (2026-08-14).
MAX_TOKENS = 16384


# --- numbered: SURF's released prompt, verbatim (jinja {{ query }} -> {query} only) ---

QUERY_ATTR_PROMPT = """A user wrote this query to a large language model assistant:

<query>

{query}

</query>

Write a list of {n} attributes which describe this query, encompassing whichever aspects of content, style, formatting, tone, structure, perspective, et cetera which are most relevant. Write each as a sentence starting with "The query", listed in XML tags from <1> to <{n}>. Avoid referencing overly specific words in the query. You should only describe attributes present in the query, not those that are absent. Include nothing else in your response."""

# Our extension: SURF has no reasoning channel. The extractor sees the query as context so
# relational properties are expressible ("The reasoning weighs the user's stated stake"),
# but every attribute still describes the reasoning.
REASONING_ATTR_PROMPT = """A user wrote this query to a large language model assistant:

<query>

{query}

</query>

Before answering, the assistant produced this internal reasoning:

<reasoning>

{reasoning}

</reasoning>

Write a list of {n} attributes which describe this reasoning, encompassing whichever aspects of content, style, formatting, tone, structure, perspective, et cetera which are most relevant — including how the reasoning relates to the query. Write each as a sentence starting with "The reasoning", listed in XML tags from <1> to <{n}>. Avoid referencing overly specific words in the reasoning. You should only describe attributes present in the reasoning, not those that are absent. Include nothing else in your response."""

RESPONSE_ATTR_PROMPT = """A user wrote this query to a large language model assistant:

<query>

{query}

</query>

The assistant wrote this response:

<response>

{response}

</response>

Write a list of {n} attributes which describe this response, encompassing whichever aspects of content, style, formatting, tone, structure, perspective, et cetera which are most relevant — including how the response relates to the query. Write each as a sentence starting with "The response", listed in XML tags from <1> to <{n}>. Avoid referencing overly specific words in the response. You should only describe attributes present in the response, not those that are absent. Include nothing else in your response."""

NUMBERED_PROMPTS = {"query": QUERY_ATTR_PROMPT, "reasoning": REASONING_ATTR_PROMPT,
                    "response": RESPONSE_ATTR_PROMPT}


# --- freeform: the LessWrong post's prompt, verbatim ---------------------------------

FEATURE_EXTRACTION_PROMPT = """For the given conversation section text, identify key "features".

Here are some examples of possible features. Try not to anchor too much on any one of \
these, they are just meant to give you a "vibe" of what to aim for:

* The model is depressed

* Talks about apples

* Uses markdown

* Backtracks in reasoning

* Self Correction in reasoning

* Few shot prompt

* Doesn't have access to required tool

* Hallucinates tool call

* Creative writing request

* Model adopts persona

* Model adopts expert coder persona

* Thoughts are disjointed and hard to follow

* Uses emojis

* Uses bullet points

* Very realistic

* Very fictional

* Sycophantic response

* Displays evaluations awareness

* Typo

* Roleplaying

* About [topic]

* Uses placeholders

* In Mandarin



Please prioritize the following properties:

(1) Interestingness: Do generated features features represent novel or surprising \
behaviors?

(2) Appropriate abstraction: Do generated features operate at a useful level of \
specificity, i.e., neither so narrow as to apply to only a few examples, nor so broad \
as to lack discriminative power?

(3) Uniqueness: Generated features should be as different as possible. It is better to \
return fewer features with less duplication than many features with duplicates.



Please make features use only letters a-z, e.g. don't include parentheses, colons, \
numbers, etc. Please capitalize only the first word and any proper nouns in the \
feature.



It might help to brainstorm many features and then select the best ones by these \
criteria."""

# OUR addition, not the post's: it specifies no output format. The no-preamble rule
# suppresses the visible brainstorm the prompt above invites, which measured as a small
# saving (~590 -> ~490 output tokens per trace) — the budget is dominated by the feature
# strings themselves, not by preamble.
FEATURE_JSON_OUTPUT_CONTRACT = """

Return between {low} and {high} features as a JSON array of strings and nothing else, e.g.
["Backtracks in reasoning", "Weighs competing obligations explicitly"]
Output only the array. No preamble, no brainstorming, no explanation, no trailing text."""


@dataclass(frozen=True)
class AttributeSpec:
    """How to extract attributes for one run.

    Attributes:
        style: "numbered" (SURF) or "freeform" (the post).
        channel: Which channel to describe — one of CHANNELS.
        n: Attributes per record (numbered), or the upper bound (freeform).
        n_min: Lower bound (freeform only).
        model: OpenRouter extractor model.
        temperature: Sampling temperature; SURF's 1.0 by default.
        max_tokens: Completion cap.
    """

    style: str = "numbered"
    channel: str = "reasoning"
    n: int = 10
    n_min: int = 10
    model: str = "anthropic/claude-sonnet-5"
    temperature: float = EXTRACT_TEMPERATURE
    max_tokens: int = MAX_TOKENS

    def validate(self) -> AttributeSpec:
        """Check the two enum fields.

        Returns:
            Self, so this chains.

        Raises:
            ValueError: On an unknown style or channel.
        """
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {self.style!r}")
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        return self

    def to_dict(self) -> dict:
        """The spec as a plain dict for a property row's provenance.

        Returns:
            The record.
        """
        return {"style": self.style, "channel": self.channel, "n": self.n,
                "model": self.model, "temperature": self.temperature}


def parse_numbered(text: str, n: int) -> list[str]:
    """Parse `<1>`..`<n>` attribute tags, failing on any missing one.

    Models frequently omit the closing tags (SURF's own parser was lenient too), so a tag
    body runs to `</i>`, the next numbered tag, or the end of the text.

    Args:
        text: Raw extractor output.
        n: How many tags to expect.

    Returns:
        The n attribute strings, whitespace collapsed.

    Raises:
        ValueError: If any tag is missing or empty.
    """
    out = []
    for i in range(1, n + 1):
        match = re.search(rf"<{i}>\s*(.*?)\s*(?:</{i}>|(?=<{i + 1}>)|\Z)", text, re.DOTALL)
        if not match or not match.group(1).strip():
            raise ValueError(f"attribute <{i}> missing or empty in extractor output:\n"
                             f"{text[:500]}")
        out.append(re.sub(r"\s+", " ", match.group(1)).strip())
    return out


def build_messages(record: Record, spec: AttributeSpec) -> list[dict]:
    """Build the extraction request for one record.

    Args:
        record: The record to describe.
        spec: How to extract.

    Returns:
        OpenAI-style messages.
    """
    if spec.style == "freeform":
        contract = FEATURE_JSON_OUTPUT_CONTRACT.format(low=spec.n_min, high=spec.n)
        return [{"role": "system", "content": FEATURE_EXTRACTION_PROMPT + contract},
                {"role": "user", "content": record.channel(spec.channel)}]
    prompt = NUMBERED_PROMPTS[spec.channel].format(
        n=spec.n, query=record.query, reasoning=record.reasoning,
        response=record.response)
    return [{"role": "user", "content": prompt}]


def parse(text: str, spec: AttributeSpec) -> list[str]:
    """Parse one extractor reply into attribute strings.

    Args:
        text: Raw model output.
        spec: How it was extracted.

    Returns:
        The attributes.

    Raises:
        ValueError: If the reply does not carry the expected shape.
    """
    if spec.style == "freeform":
        from src.utils import extract_json

        parsed = extract_json(text)
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError(f"expected a JSON array of strings, got {text[:300]!r}")
        return [re.sub(r"\s+", " ", x).strip() for x in parsed if x.strip()]
    return parse_numbered(text, spec.n)


def extract(records: list[Record], spec: AttributeSpec, workers: int = 16,
            client=None) -> list[dict]:
    """Extract attributes for every record.

    Args:
        records: The records to describe.
        spec: How to extract.
        workers: Concurrent requests.
        client: An OpenRouterClient; one is built when omitted (kept injectable so tests
            never reach the network).

    Returns:
        One {"record_id", "attributes", "tokens_in", "tokens_out"} per record, in order.
        A record whose extraction fails to parse gets `attributes: []` and an `error`
        string rather than aborting the pass — a corpus-wide run must not die on one bad
        reply, and an empty list contributes to no group, so the failure shows up as
        reduced coverage rather than as a silently wrong number.
    """
    from src.endpoints.openrouter import OpenRouterClient, map_threaded

    spec = spec.validate()
    client = client or OpenRouterClient()

    def run(i: int) -> dict:
        record = records[i]
        try:
            result = client.chat(model=spec.model,
                                 messages=build_messages(record, spec),
                                 temperature=spec.temperature,
                                 max_tokens=spec.max_tokens)
            return {"record_id": record.record_id,
                    "attributes": parse(result.content, spec),
                    "tokens_in": result.prompt_tokens,
                    "tokens_out": result.completion_tokens}
        except Exception as exc:  # noqa: BLE001 - recorded per record, never swallowed
            return {"record_id": record.record_id, "attributes": [],
                    "tokens_in": 0, "tokens_out": 0,
                    "error": f"{type(exc).__name__}: {exc}"[:300]}

    return map_threaded(run, len(records), max_workers=workers,
                        desc=f"attributes ({spec.style}/{spec.channel})")
