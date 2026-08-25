# ABOUTME: The two verbatim LessWrong prompts (post WAZWA6FPQvH8okouJ) this pipeline runs on:
# ABOUTME: free-text feature extraction from one trace, and ~5-word naming of one cluster.

from __future__ import annotations

# Verbatim from the post / the user's request. Do not reword: the whole point of the
# replication is that this prompt, not a schema of ours, decides what a "feature" is.
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

# Two additions to the post's prompt, both deliberate and both deviations to keep in mind
# when comparing results: (1) the post specifies no output format, so the JSON contract is
# ours; (2) the no-preamble rule suppresses the visible brainstorm the prompt above invites,
# which cut output only from ~590 to ~490 tokens per trace (measured on 3 and 4 traces, so
# within noise; ~$30 -> ~$27 over the corpus). The output budget is dominated by 20 verbose
# feature strings, not by preamble.
FEATURE_JSON_OUTPUT_CONTRACT = """

Return between 10 and 20 features as a JSON array of strings and nothing else, e.g.
["Backtracks in reasoning", "Weighs competing obligations explicitly"]
Output only the array. No preamble, no brainstorming, no explanation, no trailing text."""

# Verbatim from the post.
CLUSTER_NAMING_PROMPT = ("produce a single concise label (around 5 words) that captures the "
                         "common theme of these features")


def build_feature_extraction_system_prompt() -> str:
    """The extraction prompt sent as the cacheable system message.

    Returns:
        The post's feature prompt plus the output-format contract.
    """
    return FEATURE_EXTRACTION_PROMPT + FEATURE_JSON_OUTPUT_CONTRACT


def build_cluster_naming_messages(sampled_cluster_features: list[str]) -> list[dict]:
    """Build the cluster-naming request for one cluster's sampled features.

    Args:
        sampled_cluster_features: Up to 100 randomly sampled features from a single cluster.

    Returns:
        OpenAI-style message list asking for the cluster label.
    """
    listing = "\n".join(f"* {f}" for f in sampled_cluster_features)
    return [{"role": "user", "content":
             f"Here are features drawn from one cluster:\n\n{listing}\n\n"
             f"{CLUSTER_NAMING_PROMPT}. "
             "Reply with the label only, no punctuation or explanation."}]
