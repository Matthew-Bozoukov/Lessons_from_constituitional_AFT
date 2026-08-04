# ABOUTME: Style, degeneracy and think-block metrics over model generations, shared by
# ABOUTME: the capability-eval generation step and its report step.

"""Instrumentation over candidate answers for the capability regression eval.

Two families live here, and they answer different questions:

**Style features** (`style_features`) are the covariates the style-control regression
removes from the win rate. They must be computed identically for every arm, so this
module owns the one definition — length in gpt-4o tokens plus header / list / bold
counts, with fenced code blocks stripped first so a code-heavy answer is not scored as
heavily formatted prose. This mirrors arena-hard-auto's `add_markdown_info.py`, which is
what its BT regression was validated against.

**Degeneracy counters** (`degeneracy_metrics`) are the cheap early-warning instruments
from spec §8. Behavioural collapse shows up here before it shows up in any benchmark,
and they cost nothing because they run over generations we already have.

The two are deliberately separate. Style control tells us the win rate *net of* style;
the raw style deltas tell us how much drift there was to control for — a fact about our
data pipeline we want regardless of what the win rate does.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Fenced code blocks are removed before counting markdown, matching arena-hard-auto.
# Counting them would let a correct, code-dense answer look like heavy formatting.
_CODE_FENCE = re.compile(r"```([^`]*)```", re.DOTALL)

# Refusal cues. Deliberately narrow: these fire on an assistant declining the task, not
# on ordinary hedging. A broad pattern would count every "it depends" as a refusal and
# make the counter useless exactly when the corpus is teaching nuance.
_REFUSAL = re.compile(
    r"(?i)\b("
    r"i (?:can't|cannot|won't|will not|am unable to|am not able to) (?:help|assist|provide|do|comply|continue)"
    r"|i'm (?:sorry|afraid)[,.]? (?:but )?i (?:can't|cannot|won't)"
    r"|i (?:must|have to) (?:decline|refuse)"
    r"|as an ai(?: language model)?[,]? i (?:can't|cannot|don't)"
    r"|i'm not (?:comfortable|able) (?:with|to)"
    r")\b"
)

_TOKENIZER: Any = None


def _encoder() -> Any:
    """Return a cached gpt-4o tiktoken encoder."""
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken

        _TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
    return _TOKENIZER


def count_markdown_elements(text: str) -> dict[str, dict[str, int]]:
    """Count markdown structure in `text`, ignoring fenced code.

    Args:
        text: Answer text (code fences are removed internally).

    Returns:
        Nested counts under `header_count`, `list_count` and `bold_count`, matching the
        schema arena-hard-auto's style-control regression consumes.
    """
    stripped = _CODE_FENCE.sub("", text)
    return {
        "header_count": {
            f"h{level}": len(
                re.findall(rf"^#{{{level}}}\s", stripped, re.MULTILINE)
            )
            for level in range(1, 7)
        },
        "list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", stripped, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", stripped, re.MULTILINE)),
        },
        "bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", stripped)),
            "__": len(re.findall(r"__[^_\n]+__", stripped)),
        },
    }


def style_features(answer: str) -> dict[str, Any]:
    """Build the arena-hard `metadata` block for one answer.

    Args:
        answer: The user-visible answer, with any reasoning trace already removed.

    Returns:
        `{"token_len": int, "header_count": {...}, "list_count": {...},
        "bold_count": {...}}`.
    """
    token_len = len(_encoder().encode(answer, disallowed_special=()))
    return {"token_len": token_len} | count_markdown_elements(answer)


def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams in `text` that are repeats of an earlier n-gram.

    A healthy answer sits near zero; a model stuck in a loop climbs toward one. Operates
    on whitespace tokens, which is coarse but robust across languages and code.

    Args:
        text: Answer text.
        n: N-gram width.

    Returns:
        Repeat fraction in [0, 1]; 0.0 when the text is shorter than `n` tokens.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def is_refusal(answer: str) -> bool:
    """Whether `answer` reads as declining the task.

    Args:
        answer: The user-visible answer.

    Returns:
        True if a refusal cue is present.
    """
    return bool(_REFUSAL.search(answer or ""))


def pattern_frequencies(answers: Iterable[str], patterns: dict[str, str]) -> dict[str, float]:
    """Fraction of answers matching each named regex.

    These are the corpus-level structural signatures from spec §6 — GDM found BLUF
    openings at 52% and emotional-validation buffering at 26% in theirs. Measuring them
    per checkpoint is what turns "the voice changed" into a number.

    Args:
        answers: Answer texts.
        patterns: `{name: regex}`.

    Returns:
        `{name: fraction}`; empty input yields 0.0 for every pattern.
    """
    compiled = {name: re.compile(rx) for name, rx in patterns.items()}
    texts = list(answers)
    if not texts:
        return {name: 0.0 for name in compiled}
    return {
        name: sum(1 for t in texts if rx.search(t or "")) / len(texts)
        for name, rx in compiled.items()
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def degeneracy_metrics(
    records: list[dict],
    percentiles: list[int],
    repetition_threshold: float = 0.3,
) -> dict[str, Any]:
    """Aggregate the spec §8 degeneracy counters over one arm's generations.

    Args:
        records: Per-answer dicts with keys `answer`, `think`, `finish_reason`.
        percentiles: Length-distribution percentiles to report.
        repetition_threshold: Repeat fraction above which an answer counts as degenerate.

    Returns:
        Counter dict. `refusal_rate` is over *benign* prompts only — the whole capability
        prompt set is benign by construction, which is what makes an increase here
        unambiguously a regression rather than the traits working as intended.
    """
    n = len(records)
    if n == 0:
        raise ValueError("degeneracy_metrics called with no records")

    lengths = sorted(float(len(r["answer"].split())) for r in records)
    think_lengths = sorted(float(len(r["think"].split())) for r in records)
    reps = [repetition_ratio(r["answer"]) for r in records]

    return {
        "n": n,
        # Footgun §10.2: a damaged EOS shows up as truncation, and tanks win rate for
        # mechanical reasons that have nothing to do with capability.
        "truncation_rate": sum(1 for r in records if r["finish_reason"] == "length") / n,
        "empty_answer_rate": sum(1 for r in records if not r["answer"].strip()) / n,
        "refusal_rate_benign": sum(1 for r in records if is_refusal(r["answer"])) / n,
        "repetition_rate": sum(1 for x in reps if x > repetition_threshold) / n,
        "mean_repetition_ratio": sum(reps) / n,
        "mean_output_words": sum(lengths) / n,
        # Report the SHAPE, not just the mean (footgun §10.2).
        "length_percentiles": {str(p): _percentile(lengths, p) for p in percentiles},
        # CLAUDE.md gotcha 2: naive SFT on Qwen3 collapses the <think> block to empty.
        # If this goes to zero after training, the model stopped reasoning and every
        # downstream number is measuring a different model than we think.
        "mean_think_words": sum(think_lengths) / n,
        "empty_think_rate": sum(1 for r in records if not r["think"].strip()) / n,
    }
