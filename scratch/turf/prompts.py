# ABOUTME: All LLM prompts for the TURF adaptation: attribute extraction (query /
# ABOUTME: reasoning / response channels), crux selection, and cluster summaries.

"""Prompts for TURF (arXiv 2602.05910, adapted).

`QUERY_ATTR_PROMPT` is SURF's released prompt VERBATIM
(github.com/seoirsem/SURF, surf/extraction/prompts.py, SINGLE_ATTRIBUTION_PROMPT) —
kept byte-identical so our query attributes live in the same distribution as the
paper's published artifacts. The response/reasoning prompts are its substitutions;
the paper's own printed response attributes follow exactly this mirrored convention
("The response provides extensive code examples"), which is the evidence the
unreleased originals were the same template. Placeholders use {var} (str.format).

Design decisions vs the paper, from the 2026-08-13 discussion:
- Trigger side = query + reasoning, response = behaviour side. Every datapoint
  contributes the SAME count to the trigger clusters: 10 "The query..." + 10
  "The reasoning..." attributes.
- The reasoning extractor sees the query as context so relational properties are
  expressible ("The reasoning weighs the user's stated stake"), but every
  attribute still describes the reasoning.
- The crux selector is reconstructed from the paper's prose (App. D.1.2 — never
  released): it selects VERBATIM from the blind attributes (rewriting would break
  embedding comparability with the dataset-side attributes), takes a polarity
  (satisfy = attribute good behaviour, the paper's original is violate), and is
  guarded against restating the dataset's own generation styles.
"""

# --- dataset + case side: attribute extraction ---------------------------------------

# SURF's released prompt, verbatim (jinja {{ query }} -> {query} is the only edit).
QUERY_ATTR_PROMPT = """A user wrote this query to a large language model assistant:

<query>

{query}

</query>

Write a list of 10 attributes which describe this query, encompassing whichever aspects of content, style, formatting, tone, structure, perspective, et cetera which are most relevant. Write each as a sentence starting with "The query", listed in XML tags from <1> to <10>. Avoid referencing overly specific words in the query. You should only describe attributes present in the query, not those that are absent. Include nothing else in your response."""

REASONING_ATTR_PROMPT = """A user wrote this query to a large language model assistant:

<query>

{query}

</query>

Before answering, the assistant produced this internal reasoning:

<reasoning>

{reasoning}

</reasoning>

Write a list of 10 attributes which describe this reasoning, encompassing whichever aspects of content, style, formatting, tone, structure, perspective, et cetera which are most relevant — including how the reasoning relates to the query. Write each as a sentence starting with "The reasoning", listed in XML tags from <1> to <10>. Avoid referencing overly specific words in the reasoning. You should only describe attributes present in the reasoning, not those that are absent. Include nothing else in your response."""

RESPONSE_ATTR_PROMPT = """A user wrote this query to a large language model assistant:

<query>

{query}

</query>

The assistant wrote this response:

<response>

{response}

</response>

Write a list of {n} attributes which describe this response, encompassing whichever aspects of content, style, formatting, tone, structure, perspective, et cetera which are most relevant — including how the response relates to the query. Write each as a sentence starting with "The response", listed in XML tags from <1> to <{n}>. Avoid referencing overly specific words in the response. You should only describe attributes present in the response, not those that are absent. Include nothing else in your response."""

# --- trace side: crux selection -------------------------------------------------------

# Reconstructed informed judge (paper App. D.1.2). polarity_verb is "satisfying" or
# "violating"; styles_block lists the dataset's generation styles (from styles.json)
# or is empty for foreign datasets.
CRUX_SELECT_PROMPT = """Here are {n} attributes describing an AI assistant's response:

<attributes>
{attributes}
</attributes>

The response is being assessed against this rubric:

<rubric>
{rubric}
</rubric>

Select exactly 3 attributes from the list that are most causally responsible for the response {polarity_verb} the rubric.

Rules:
- Copy each selected attribute VERBATIM from the list. Do not rewrite, merge, shorten, or invent attributes.
- Do NOT select an attribute if it merely restates one of the following known styles of this training dataset (their subject matter or defining document shape):
{styles_block}
- Attributes describing formatting, structure, or stylistic habits ARE eligible — unless they merely restate a known style above.

Output format, and nothing else:
<crux_1>...</crux_1>
<crux_2>...</crux_2>
<crux_3>...</crux_3>
<excluded_as_style>attributes you rejected under the style rule, one per line, or "none"</excluded_as_style>"""

NO_STYLES_BLOCK = "  (no known styles for this dataset — the style rule does not apply)"

# --- index side: cluster summaries ----------------------------------------------------

CLUSTER_SUMMARY_PROMPT = """Here are attribute sentences that were grouped into one cluster by embedding similarity:

<attributes>
{attributes}
</attributes>

Write ONE sentence, in the same style ("The query ..." or "The reasoning ..."), that best summarizes what these attributes have in common. Output only that sentence."""
