# ABOUTME: Sentence embeddings and the `T_j ~= S_i` similarity test that resilience,
# ABOUTME: counterfactual++ and the dissimilar-resample filter are all defined in terms of.

"""Semantic sameness, which is where every metric in the paper is actually decided.

Three of the paper's quantities are thresholds on cosine similarity, not on text:

    counterfactual importance   keeps resamples where `cos(S_i, T_i)` is BELOW the median
    resilience                  iterates while the best match stays ABOVE tau
    counterfactual++            keeps rollouts where NO later sentence matches S_i

so the embedding choice and the threshold rule are load-bearing, and a silent default
would quietly change every number. Both are therefore explicit here.

The paper uses `bert-large-nli-stsb-mean-tokens`. This repo has no `sentence-transformers`
and its drivers run on a laptop, so the default backend is `src/data/synth/embeddings.py`
(model2vec `potion-base-8M`): local, torch-free, deterministic, and fast enough to embed
a whole ODCV corpus in seconds. It is a weaker model, which matters in one direction
worth naming — it will call near-paraphrases dissimilar slightly too often, which makes
resilience read LOW rather than high. Treat resilience as a lower bound under this
backend, and re-run through `src/properties/shared/embed.py` before publishing a number
that hangs on it.

The threshold follows the paper: tau is the MEDIAN similarity of the comparison set, not
a fixed constant, so it self-calibrates to how repetitive a given trace is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.synth.embeddings import embed as _synth_embed

DEFAULT_MODEL = "minishlab/potion-base-8M"


def encode(texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed texts as L2-normalised row vectors.

    Args:
        texts: Texts to embed; empty strings are allowed and embed to a zero row.
        model: model2vec model id.

    Returns:
        Array of shape (len(texts), dim), L2-normalised so a dot product IS the cosine.
        An empty input returns shape (0, 0).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    safe = [t if t.strip() else " " for t in texts]
    return np.asarray(_synth_embed(safe, model=model), dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two sets of row vectors.

    Args:
        a: (n, d) rows.
        b: (m, d) rows.

    Returns:
        (n, m) cosines. Inputs are re-normalised defensively, so this is correct even
        if a caller passes raw vectors.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (an @ bn.T).astype(np.float32)


@dataclass
class SimilarityIndex:
    """A reusable embedding of one reference set, for repeated `~=` tests.

    Resilience resamples the same target sentence many times, so its vector should be
    computed once. Holding it in an object also keeps the threshold rule attached to the
    vectors it was calibrated on.

    Attributes:
        texts: The reference texts.
        vectors: Their L2-normalised embeddings.
        model: The backend used, recorded so a run's provenance names it.
    """

    texts: list[str]
    vectors: np.ndarray
    model: str = DEFAULT_MODEL

    @classmethod
    def build(cls, texts: list[str], model: str = DEFAULT_MODEL) -> "SimilarityIndex":
        return cls(
            texts=list(texts), vectors=encode(list(texts), model=model), model=model
        )

    def against(self, texts: list[str]) -> np.ndarray:
        """Cosines of `texts` against every reference text: shape (len(texts), n_ref)."""
        return cosine(encode(texts, model=self.model), self.vectors)


def best_match(
    target: str, candidates: list[str], model: str = DEFAULT_MODEL
) -> tuple[int, float]:
    """The candidate most semantically similar to `target`.

    Args:
        target: Reference text.
        candidates: Texts to search.
        model: Embedding backend.

    Returns:
        (index, cosine) of the best match; (-1, -1.0) when there are no candidates.
    """
    if not candidates:
        return -1, -1.0
    sims = cosine(encode([target], model=model), encode(candidates, model=model))[0]
    j = int(np.argmax(sims))
    return j, float(sims[j])


def median_threshold(similarities: np.ndarray) -> float:
    """The paper's tau: the median of the comparison set's similarities.

    Args:
        similarities: Any array of cosines.

    Returns:
        The median, or 0.0 for an empty array.
    """
    flat = np.asarray(similarities, dtype=np.float32).ravel()
    return float(np.median(flat)) if flat.size else 0.0


def dissimilar_mask(
    target: str,
    candidates: list[str],
    tau: float | None = None,
    model: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Which candidates count as "not the target sentence" — the paper's `T_i !~ S_i`.

    Args:
        target: The sentence being ablated.
        candidates: The resampled sentences that replaced it.
        tau: Similarity cutoff; None uses the median over `candidates`, as the paper does.
        model: Embedding backend.

    Returns:
        Boolean mask, True where the candidate is semantically DIFFERENT from the target
        and so counts as a genuine ablation.
    """
    if not candidates:
        return np.zeros(0, dtype=bool)
    sims = cosine(encode([target], model=model), encode(candidates, model=model))[0]
    cut = median_threshold(sims) if tau is None else tau
    return sims < cut


def content_absent(
    target: str, trace: list[str], tau: float, model: str = DEFAULT_MODEL
) -> bool:
    """Whether a sentence's content is absent from a whole downstream trace.

    This is the counterfactual++ condition `for all j >= i: T_j !~ S_i` — the one that
    separates "we deleted a sentence" from "we deleted an idea".

    Args:
        target: The ablated sentence.
        trace: Every sentence of the resampled continuation.
        tau: Similarity cutoff above which the content counts as having reappeared.
        model: Embedding backend.

    Returns:
        True when no sentence in `trace` reaches `tau` against `target`.
    """
    if not trace:
        return True
    _, best = best_match(target, trace, model=model)
    return best < tau
