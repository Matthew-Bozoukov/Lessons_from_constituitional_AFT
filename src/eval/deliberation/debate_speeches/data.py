# ABOUTME: Load IBM's annotated debate speeches from the Hub and pair each speech with the mean
# ABOUTME: of its human 1-5 ratings on one of the three annotated statements.

"""The dataset side of the `debate_speeches` eval.

Upstream: [`ibm-research/debate_speeches`](https://huggingface.co/datasets/ibm-research/debate_speeches),
the opening-speech collection from IBM's Project Debater
([Nature, 2021](https://www.nature.com/articles/s41586-021-03215-w)); used as an LLM-judge
benchmark by *Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation*
(IBM, EMNLP 2025, [arXiv:2506.05062](https://arxiv.org/abs/2506.05062)).

948 speeches, each arguing *in favour* of a motion, each scored 1–5 by ~15–30 crowd
annotators on three statements. The statements are reproduced verbatim in `STATEMENTS`
because they are the task: a paraphrase would be measuring agreement with a different
question than the humans answered.

Speech quality varies enormously by `source` — human expert debaters, IBM's automated
system, GPT-2 pipelines, and *Mixed stance control* speeches deliberately built by splicing
a for-speech with an against-speech. That last group is the reason this dataset suits CR
rather than just any argument-quality set: a spliced speech is internally contradictory, so
detecting it requires actually weighing the argument rather than scoring surface fluency,
which is precisely the habit courtroom claims to train.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

HF_REPO = "ibm-research/debate_speeches"
HF_FILE = "opening_speeches/train-00000-of-00001.parquet"

# The annotators' own statements, verbatim from the dataset card. Column -> statement.
STATEMENTS: dict[str, str] = {
    "goodopeningspeech":
        "This speech is a good opening speech for supporting the topic.",
    "mostargumentssupport":
        "Most arguments in this speech support the topic.",
    "interestingspeaker":
        "The content of this speech is interesting and informative.",
}

# How the 1-5 scale was presented to the annotators. Reproduced so the model rates on the
# same anchors rather than an invented "1 = bad, 5 = good".
SCALE = ("5 = Strongly agree, 4 = Agree, 3 = Neither agree nor disagree, "
         "2 = Disagree, 1 = Strongly disagree")


@dataclass(frozen=True)
class Item:
    """One speech with its human rating distribution on the chosen statement.

    Attributes:
        uid: `<topic_id>:<index>` — topic_id repeats across sources, so the index
            disambiguates.
        topic: The motion the speech argues for.
        text: The speech.
        source: How the speech was produced (human expert, Project Debater, …).
        ratings: Every annotator's 1-5 rating.
        human_mean: Their mean — the target this eval correlates against.
    """

    uid: str
    topic: str
    text: str
    source: str
    ratings: tuple[int, ...]
    human_mean: float


def _ratings(cell) -> tuple[int, ...]:
    """Parse a ratings cell.

    The card types these columns as `string` while the parquet round-trips them as lists,
    so both shapes are handled rather than assuming whichever one today's `datasets`
    version produces.
    """
    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    return tuple(int(x) for x in cell)


def load_items(dimension: str = "goodopeningspeech", limit: int = 0, seed: int = 0,
               sources: list[str] | None = None) -> list[Item]:
    """Load speeches paired with their human mean rating.

    Args:
        dimension: Which annotated statement to score; a key of `STATEMENTS`.
        limit: Cap on items (0 = all 948), applied after a seeded shuffle so a smoke run
            samples across sources instead of taking one topic's block.
        seed: Shuffle seed.
        sources: Restrict to these `source` values; None = all.

    Returns:
        Items in shuffled order.
    """
    import random

    import pyarrow.parquet as pq

    assert dimension in STATEMENTS, \
        f"unknown dimension {dimension!r}; known: {sorted(STATEMENTS)}"

    rows = pq.read_table(hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")).to_pylist()
    if sources is not None:
        allowed = set(sources)
        known = {str(r["source"]) for r in rows}
        unknown = allowed - known
        assert not unknown, f"unknown source(s) {sorted(unknown)}; known: {sorted(known)}"
        rows = [r for r in rows if str(r["source"]) in allowed]

    items: list[Item] = []
    for index, row in enumerate(rows):
        ratings = _ratings(row[dimension])
        assert ratings, f"row {index} has no ratings for {dimension}"
        items.append(Item(
            uid=f"{row['topic_id']}:{index}",
            topic=str(row["topic"]),
            text=str(row["text"]),
            source=str(row["source"]),
            ratings=ratings,
            human_mean=sum(ratings) / len(ratings),
        ))

    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items
