# ABOUTME: Corpus-level properties of a generated dataset (diversity, duplication,
# ABOUTME: coverage, leakage) as a registry: one function + one CORPUS_CHECKS entry each.

"""Corpus-level checks.

The autorater asks "is this document good?". These ask "is this *corpus* good?" --
is it diverse, does it repeat itself, does every bucket have documents in it, is the
label predictable from surface form. Three rules the module turns on:

1. **A check flags; it never fixes.** `run_corpus_checks` returns a report and mutates
   nothing. A checker allowed to drop rows is how 1,266 documents once vanished behind
   a dead API key.
2. **A check that cannot run says so.** Missing field -> `skipped` with a reason;
   raising -> `errored`; too few documents -> `reported` but not gated. Never a silent
   pass.
3. **Generic code knows no document type.** A check declares the field *roles* it
   consumes (`text`, `id`, `group`, `label`); the config maps roles to record keys.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import random
import re
import time
import zlib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

from src.utils import wilson

# --- shared primitives ---------------------------------------------------------------
# Here rather than in check_model_eval_model.py so the in-pipeline stage and the post-hoc `synth check`
# verb cannot drift on what "8-gram share" or "surface AUC" means.


def words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped."""
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """The set of word n-grams of a token list."""
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def hashed_features(texts: list[str], dim: int = 4096):
    """L2-normalised hashed char-3/4/5-gram count features (crc32, deterministic).

    One `np.bincount` per document rather than a per-n-gram index-assign: ~1.7x faster,
    bit-identical -- the registry's measured thresholds depend on these exact values.
    """
    import numpy as np

    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = t.lower().encode()
        crc = zlib.crc32
        hashes = [crc(s[j:j + n]) % dim
                  for n in (3, 4, 5) for j in range(len(s) - n + 1)]
        if hashes:
            counts = np.bincount(np.asarray(hashes, dtype=np.intp), minlength=dim)
            X[i] = counts[:dim]
            X[i] /= np.linalg.norm(X[i]) or 1.0
    return X


def auc(y, scores) -> float:
    """Mann-Whitney AUC."""
    pos, neg = scores[y == 1], scores[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cv_auc(X, y, seed: int, folds: int = 5, epochs: int = 300,
           lr: float = 0.5) -> float | None:
    """K-fold cross-validated AUC of a plain logistic regression (numpy, no sklearn)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    aucs = []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        if len(set(y[train].tolist())) < 2 or len(set(y[test].tolist())) < 2:
            continue
        w = np.zeros(X.shape[1], dtype=np.float32)
        b = 0.0
        Xt, yt = X[train], y[train].astype(np.float32)
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(Xt @ w + b)))
            g = p - yt
            w -= lr * (Xt.T @ g) / len(yt)
            b -= lr * float(g.mean())
        aucs.append(auc(y[test], X[test] @ w + b))
    return round(float(np.mean(aucs)), 4) if aucs else None


def entropy(counts: list[int]) -> float:
    """Shannon entropy of a count vector, normalised to [0, 1] by ln(k)."""
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 1.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return h / math.log(len(counts))


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    i = min(int(round(q * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return float(sorted_vals[i])


# --- types ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One flagged problem: WHAT is wrong plus examples, never what to do about it."""

    property: str
    severity: str                    # "critical" | "warn" | "info" | "error"
    metric: str                      # the key in `metrics` that triggered it
    value: Any
    threshold: Any = None            # None for a pure observation
    scope: str = ""                  # "" = whole corpus; else the group/axis value
    summary: str = ""
    examples: tuple[str, ...] = ()   # <= 5 record ids, so a human can go look

    def as_dict(self) -> dict:
        """JSON-serialisable form, derived from the fields so a new one cannot be
        silently dropped from every report."""
        d = dataclasses.asdict(self)
        d["examples"] = list(self.examples)
        return d


@dataclass
class CheckResult:
    """What one property produces.

    `metrics` is always populated, even when nothing is flagged -- the numbers are the
    point, the findings are the alarm. `labels` carries per-record judgements from a
    judged property; they go to a sidecar file, never into the records themselves.
    """

    metrics: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    labels: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusCheck:
    """One registered corpus property: the function, the roles it reads, its own
    thresholds, and what it costs."""

    name: str                        # registry key
    fn: Callable[["Corpus"], CheckResult]
    roles: tuple[str, ...] = ("text", "id")   # the stage entry maps these to record keys
    defaults: dict = field(default_factory=dict)   # overridden per entry by `params:`
    min_docs: int = 1                # below this: reported, never gated (noise at smoke)
    paid: bool = False               # whether it calls a model
    est_calls: Callable[[dict, int], int] | None = None      # (params, n) -> API calls
    # (stage entry, resolved params, name) -> None, raising on a config this property
    # cannot run with. Called at build_stages time, so a missing rubric stops the run
    # BEFORE the generation stages spend anything.
    validate: Callable[[dict, dict], None] | None = None
    doc: str = ""                    # one line for the README table

    @property
    def tier(self) -> str:
        """`surface` (free, offline, every run) or `judged` (calls a model, costs money).

        Derived from `paid` rather than declared, so the two can never disagree.
        """
        return "judged" if self.paid else "surface"


# --- field resolution ----------------------------------------------------------------


def _dotted(record: dict, path: str) -> Any:
    """Read `a.b.c` out of a nested record, returning None if any hop is missing."""
    cur: Any = record
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _from_messages(record: dict, spec: dict) -> str | None:
    """Concatenate the chosen roles' content from a `{messages: [...]}` record, so a
    check placed last audits what actually trains."""
    msgs = record.get("messages")
    if not isinstance(msgs, list):
        return None
    roles = set(spec.get("roles") or ["assistant"])
    parts = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in roles:
            continue
        if spec.get("include_reasoning") and m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
        if m.get("content"):
            parts.append(str(m["content"]))
    return "\n".join(parts) if parts else None


def resolve_field(spec: Any, record: dict) -> Any:
    """Resolve one field spec against one record, or None where it does not resolve.

    - `"metadata.cell"` -- a dotted path;
    - `["reasoning", "response"]` -- several paths, joined with newlines;
    - `{"from_messages": {"roles": [...], "include_reasoning": true}}` -- for records
      already exported to `{messages, metadata}`;
    - `{"by": "response_kind", "cases": {"good": "gold_response", ...}, "default": ...}`
      -- the record field `by` picks WHICH field to read, so a check can say "read the
      flawed text for flawed rows" without knowing either name.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return _dotted(record, spec)
    if isinstance(spec, (list, tuple)):
        parts = [_dotted(record, s) for s in spec]
        parts = [str(p) for p in parts if p not in (None, "")]
        return "\n".join(parts) if parts else None
    if isinstance(spec, dict):
        if "from_messages" in spec:
            return _from_messages(record, spec["from_messages"])
        if "by" in spec:
            key = str(_dotted(record, spec["by"])).lower()
            chosen = spec.get("cases", {}).get(key, spec.get("default"))
            return resolve_field(chosen, record) if chosen is not None else None
    raise ValueError(f"unrecognised field spec: {spec!r}")


@dataclass
class Corpus:
    """The corpus, plus everything derived from it, computed at most once.

    The derivations depend only on `records` and `fields`, so `run_corpus_checks` keeps
    one instance per distinct `fields` mapping and re-points `params`/`spec`/`name` at
    each property in turn -- otherwise tokenisation would run once per property.
    """

    records: list[dict]
    fields: dict                     # role -> field spec
    params: dict                     # check.defaults <- stage entry `params:`
    spec: dict = field(default_factory=dict)     # the whole stage entry (axes, cross, ...)
    seed: int = 0
    workers: int = 8
    run_dir: Path = Path(".")
    labels: dict = field(default_factory=dict)   # record_id -> merged judged labels
    ctx: Any = None                  # only a paid check touches ctx.client / ctx.usage
    name: str = ""                   # this instance's alias, for Finding.property

    @property
    def stage(self) -> str:
        """The name of the corpus_check stage this property runs under."""
        return str(self.spec.get("name", "corpus"))

    def role(self, role: str) -> list[Any]:
        """Resolve one role across every record (None where it does not resolve)."""
        return [resolve_field(self.fields.get(role), r) for r in self.records]

    def missing_role(self, role: str) -> str | None:
        """Why a role is unusable, or None if it resolves for some record."""
        if role not in self.fields:
            return f"stage entry declares no `fields.{role}`"
        # Read through the cached properties where there is one. NOT `ids`: it
        # substitutes a positional index for an unresolved value, masking exactly the
        # misconfiguration this is here to report.
        vals = {"text": lambda: self.texts,
                "group": lambda: self.groups}.get(role, lambda: self.role(role))()
        if not any(v not in (None, "") for v in vals):
            return (f"`fields.{role}: {self.fields[role]!r}` resolved to nothing on any "
                    f"of {len(self.records)} records")
        return None

    @cached_property
    def ids(self) -> list[str]:
        """Record ids; falls back to the positional index when no id role is mapped."""
        if "id" not in self.fields:
            return [str(i) for i in range(len(self.records))]
        return [str(v) if v not in (None, "") else str(i)
                for i, v in enumerate(self.role("id"))]

    @cached_property
    def texts(self) -> list[str]:
        """The document text of each record, per the `text` role."""
        return [str(v) if v is not None else "" for v in self.role("text")]

    @cached_property
    def tokens(self) -> list[list[str]]:
        """Word tokens per document, tokenised once for every text-based check."""
        return [words(t) for t in self.texts]

    @cached_property
    def groups(self) -> list[str]:
        """The bucket each record belongs to; "" for every record when unmapped."""
        if "group" not in self.fields:
            return [""] * len(self.records)
        return [str(v) if v not in (None, "") else "" for v in self.role("group")]

    @cached_property
    def by_group(self) -> dict[str, list[int]]:
        """Record indices per group, groups in sorted order."""
        out: dict[str, list[int]] = {}
        for i, g in enumerate(self.groups):
            out.setdefault(g, []).append(i)
        return dict(sorted(out.items()))

    def column(self, path: str) -> list[Any]:
        """Read one dotted path across every record.

        `label.<key>` reads the judged-label sidecar instead, so a judged annotation is
        usable as a coverage axis without ever being written into the corpus.
        """
        if path.startswith("label."):
            key = path[len("label."):]
            return [(self.labels.get(rid) or {}).get(key) for rid in self.ids]
        return [_dotted(r, path) for r in self.records]

    def sample(self, n: int | None, pool: list[int] | None = None) -> list[int]:
        """A seeded sample of record indices; every index when `n` is None or >= size.

        `pool` restricts the draw, so a property applying to part of the corpus still
        gets its full sample of the part that applies.
        """
        idx = list(range(len(self.records))) if pool is None else sorted(set(pool))
        if n is None or n <= 0 or n >= len(idx):
            return idx
        return sorted(random.Random(self.seed).sample(idx, n))


def flag(c: "Corpus", findings: list[Finding], severity: str, metric: str, value: float,
         key: str, summary: str, examples: Any = (), *, scope: str = "",
         low: bool = False, nd: int = 4) -> bool:
    """Append a Finding when `value` crosses the threshold `c.params[key]`.

    One place decides what "crosses" means, what precision a reported value keeps, and
    how many examples a finding carries. `examples` may be a callable, evaluated only
    when the finding fires -- on the collapse path building them costs a full re-scan.
    """
    limit = float(c.params[key])
    if (value >= limit) if low else (value <= limit):
        return False
    ex = examples() if callable(examples) else examples
    findings.append(Finding(c.name, severity, metric, round(value, nd), limit, scope,
                            summary, tuple(itertools.islice(iter(ex), 5))))
    return True


# --- offline properties --------------------------------------------------------------


def check_ngram_diversity(c: Corpus) -> CheckResult:
    """Repeated long n-grams and high pairwise overlap, per group and overall.

    One 8-gram in a large share of a group's documents is the fingerprint of a collapsed
    generator; `distinct_2` catches the same collapse from the vocabulary side, and mean
    pairwise 4-gram Jaccard catches documents built from the same parts without sharing
    any one long phrase.
    """
    p = c.params
    min_group = int(p["min_group_docs"])
    metrics: dict[str, Any] = {"by_group": {}}
    findings: list[Finding] = []

    for group, idxs in c.by_group.items():
        docs = [c.tokens[i] for i in idxs]
        grams8: dict[tuple[str, ...], int] = {}
        for d in docs:
            for g in ngrams(d, 8):
                grams8[g] = grams8.get(g, 0) + 1
        # Tie-break on the gram itself: at small N many 8-grams share the top count, and
        # an order-dependent pick makes two reports of one corpus look like they differ.
        top_gram, top_count = min(grams8.items(), key=lambda kv: (-kv[1], kv[0]),
                                  default=((), 0))
        top_share = top_count / max(len(docs), 1)

        # A fresh RNG per group: the sampled Jaccard must stay byte-identical with the
        # historical behaviour this generalises, for the same input and seed.
        rng = random.Random(c.seed)
        picked = rng.sample(docs, min(len(docs), int(p["sample"])))
        sets4 = [ngrams(d, 4) for d in picked]
        # |a u b| = |a| + |b| - |a n b|, so the union is never built: same numbers, ~3x
        # faster, and this loop is the single biggest cost in the stage.
        sims: list[float] = []
        for i, a in enumerate(sets4):
            for b in sets4[i + 1:]:
                inter = len(a & b)
                sims.append(inter / max(len(a) + len(b) - inter, 1))
        mean_j = sum(sims) / len(sims) if sims else 0.0

        total2 = sum(max(len(d) - 1, 0) for d in docs)
        uniq2 = len({tuple(d[i:i + 2]) for d in docs for i in range(len(d) - 1)})
        distinct_2 = uniq2 / total2 if total2 else 1.0

        # Below `min_group_docs` these statistics are binomial noise: two documents that
        # happen to share an 8-gram score 1.0. Measure and report, never flag.
        gated = len(docs) >= min_group
        metrics["by_group"][group] = {
            "docs": len(docs),
            "top_8gram_share": round(top_share, 3),
            "top_8gram": " ".join(top_gram),
            "mean_pairwise_4gram_jaccard": round(mean_j, 4),
            "distinct_2": round(distinct_2, 4),
            "gated": gated,
        }
        if not gated:
            continue

        scope = group or "corpus"
        flag(c, findings, "critical", "top_8gram_share", top_share,
             "top_8gram_share_max",
             f"{int(top_count)} of {len(docs)} documents share the 8-gram "
             f"{' '.join(top_gram)!r}",
             lambda: (c.ids[i] for i in idxs if top_gram in ngrams(c.tokens[i], 8)),
             scope=scope, nd=3)
        flag(c, findings, "critical", "mean_pairwise_4gram_jaccard", mean_j,
             "mean_jaccard_max",
             f"documents share {mean_j:.1%} of their 4-grams pairwise on average",
             [c.ids[i] for i in idxs], scope=scope)
        flag(c, findings, "warn", "distinct_2", distinct_2, "distinct_2_min",
             f"only {distinct_2:.1%} of bigrams are distinct",
             [c.ids[i] for i in idxs], scope=scope, low=True)

    gated_groups = [g for g in metrics["by_group"].values() if g["gated"]]
    metrics["max_top_8gram_share"] = max(
        (g["top_8gram_share"] for g in gated_groups), default=0.0)
    metrics["groups"] = len(metrics["by_group"])
    metrics["groups_too_small_to_gate"] = len(metrics["by_group"]) - len(gated_groups)
    return CheckResult(metrics, findings)




def _components(n: int, pairs) -> list[list[int]]:
    """Connected components of an undirected edge list, as sorted index groups."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return [sorted(g) for g in out.values() if len(g) > 1]


def check_embedding_dedup(c: Corpus) -> CheckResult:
    """Semantically near-duplicate documents, and how many a dedup stage would remove.

    GDM's recipe ends with "a deduplication stage to remove prompts with too-similar
    embeddings". This computes exactly that set and reports it; it does not remove it
    (module rule 1). `would_drop_ids` in the sidecar is the removal list, so a downstream
    filter can act on the same numbers a human read.

    The only check that survives a reword. Word n-grams are order-dependent, so two
    scenarios that are the same situation in different words share nothing measurable;
    a mean-pooled embedding is order-invariant and sees them.

    Duplicate sets are connected components, not pairs -- a cluster of six mutual
    near-duplicates should cost five documents, not fifteen findings. Components can
    chain, so `max_cluster` is reported: a single huge component means the threshold is
    below the corpus's natural similarity, not that the corpus collapsed.

    **Cosine is length-dependent**, so `max_mean_words` gates the gate. Mean-pooled
    static embeddings drag long same-genre prose toward one centroid: measured on the
    baseline corpus, mean pairwise cosine runs 0.37 at 68 words but 0.76 at 1,044, and
    the whole nearest-neighbour distribution compresses with it. Past the limit the
    numbers are still reported and the findings are suppressed with a note -- an
    unusable threshold must say so rather than fire (module rule 2).
    """
    import numpy as np

    from .embeddings import DEFAULT_MODEL, embed

    p = c.params
    idx = c.sample(int(p["sample"]))
    n = len(idx)
    model = str(p.get("model") or DEFAULT_MODEL)
    mean_words = sum(len(c.tokens[i]) for i in idx) / max(n, 1)
    too_long = mean_words > float(p["max_mean_words"])
    X = embed([c.texts[i] for i in idx], model=model,
              batch_size=int(p["batch_size"]))
    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    nn = G.max(axis=1) if n > 1 else np.zeros(1, dtype=np.float32)
    nn_sorted = sorted(float(v) for v in nn)
    thresh = float(p["cosine_min"])
    a_idx, b_idx = np.where(np.triu(G >= thresh, k=1))
    pairs = list(zip(a_idx.tolist(), b_idx.tolist()))

    clusters = _components(n, pairs)
    drop = sorted(i for g in clusters for i in g[1:])
    share = len(drop) / max(n, 1)
    order = np.argsort(-G[a_idx, b_idx]) if pairs else []

    metrics = {
        "sampled": n, "embedding_model": model, "cosine_min": thresh,
        "mean_words": round(mean_words, 1), "gated": not too_long,
        "near_duplicate_pairs": len(pairs),
        "duplicate_clusters": len(clusters),
        "max_cluster": max((len(g) for g in clusters), default=0),
        "would_drop": len(drop),
        "would_drop_share": round(share, 4),
        "mean_pairwise_cosine": round(float((G[G > -1].sum()) / max(n * (n - 1), 1)), 4),
        "mean_nn_cosine": round(float(nn.mean()), 4),
        "nn_cosine_p50": round(_percentile(nn_sorted, 0.50), 4),
        "nn_cosine_p95": round(_percentile(nn_sorted, 0.95), 4),
        "nn_cosine_p99": round(_percentile(nn_sorted, 0.99), 4),
        "worst_pairs": [{"a": c.ids[idx[int(a_idx[k])]], "b": c.ids[idx[int(b_idx[k])]],
                         "cosine": round(float(G[a_idx[k], b_idx[k]]), 3)}
                        for k in list(order)[:5]],
    }
    findings: list[Finding] = []
    if too_long:
        metrics["note"] = (
            f"documents average {mean_words:.0f} words, over max_mean_words="
            f"{p['max_mean_words']}. Mean-pooled embeddings compress at this length "
            f"(measured: 0.37 mean pairwise at 68 words, 0.76 at 1,044), so the cosine "
            f"thresholds do not mean here what they were measured to mean. Numbers "
            f"reported, findings suppressed -- point this property at the scenario or "
            f"prompt text, which is the unit GDM dedups.")
        return CheckResult(metrics, findings)

    flag(c, findings, "critical", "would_drop_share", share, "drop_share_max",
         f"a GDM-style embedding dedup at cosine >= {thresh} would remove {len(drop)} "
         f"of {n} documents ({len(clusters)} duplicate clusters, largest "
         f"{metrics['max_cluster']})",
         lambda: (c.ids[idx[i]] for i in drop))
    flag(c, findings, "warn", "mean_nn_cosine", float(nn.mean()), "mean_nn_cosine_max",
         f"the average document's nearest semantic neighbour sits at cosine "
         f"{float(nn.mean()):.3f}; the corpus is tightly packed even where no pair "
         f"crosses the duplicate threshold",
         [c.ids[idx[i]] for i in range(min(n, 5))])
    return CheckResult(metrics, findings,
                       {c.ids[idx[i]]: {"embedding_dup": True} for i in drop})




def check_label_leakage(c: Corpus) -> CheckResult:
    """Can a cheap classifier predict the label from surface form alone?

    If yes, whatever distinguishes the classes has a tell -- length, phrasing, a stray
    hedge -- and a model trained on the corpus learns the tell instead of the substance.
    A label-shuffle baseline anchors what chance looks like at this sample size.

    `auc` is directional (against `positive`, default: second class alphabetically) and
    is what GATES. `separability` = max(auc, 1 - auc) catches a leak pointing the other
    way and can only `warn`: on a null corpus of 25-60 per class it exceeds 0.65 about a
    third of the time and does not tighten with n. Texts appearing under BOTH labels are
    dropped first -- CV memorises such a text in one fold and scores its twin in the
    next, which measured AUC 0.02 on an all-duplicate corpus (a clean pass, wrongly).
    """
    import numpy as np

    p = c.params
    labels = [str(v) if v is not None else "" for v in c.role("label")]
    classes = sorted({v for v in labels if v})
    if len(classes) != 2:
        return CheckResult({"note": f"needs exactly 2 label values, saw {classes}",
                            "classes": classes, "gated": False})
    positive = str(p.get("positive") or classes[1])
    assert positive in classes, (
        f"label_leakage `positive: {positive!r}` is not one of the label values {classes}")
    negative = next(c_ for c_ in classes if c_ != positive)

    seen: dict[str, set[str]] = {}
    for i, lab in enumerate(labels):
        if lab:
            seen.setdefault(c.texts[i], set()).add(lab)
    ambiguous = {t for t, ls in seen.items() if len(ls) > 1}
    keep = [i for i, v in enumerate(labels) if v and c.texts[i] not in ambiguous]
    dropped = sum(1 for i, v in enumerate(labels) if v) - len(keep)
    if not keep:
        return CheckResult({"classes": classes, "positive": positive, "gated": False,
                            str(negative): 0, str(positive): 0,
                            "ambiguous_texts_dropped": dropped,
                            "note": "every text appears under both labels; there is "
                                    "nothing for a classifier to predict"})
    y = np.array([1 if labels[i] == positive else 0 for i in keep])
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    per_class = int(p["min_per_class"])
    if min(n_pos, n_neg) < per_class:
        return CheckResult({"classes": classes, "positive": positive,
                            str(negative): n_neg, str(positive): n_pos, "gated": False,
                            "ambiguous_texts_dropped": dropped,
                            "note": f"needs >={per_class} docs per class"})

    texts = [c.texts[i] for i in keep]
    X = hashed_features(texts)
    a = cv_auc(X, y, c.seed)
    rng = np.random.default_rng(c.seed)
    a_shuf = cv_auc(X, rng.permutation(y), c.seed)
    wc = np.array([len(t.split()) for t in texts], dtype=np.float32)
    delta = float(wc[y == 1].mean() - wc[y == 0].mean())
    sep = None if a is None else round(max(a, 1.0 - a), 4)
    sep_shuf = None if a_shuf is None else round(max(a_shuf, 1.0 - a_shuf), 4)

    metrics = {"classes": classes, "positive": positive,
               str(negative): n_neg, str(positive): n_pos,
               "ambiguous_texts_dropped": dropped,
               "auc": a, "separability": sep, "auc_label_shuffled": a_shuf,
               "separability_label_shuffled": sep_shuf, "gated": True,
               "mean_word_delta": round(delta, 1)}
    limit = float(p["surface_auc_max"])
    findings = []
    examples = tuple(c.ids[i] for i in keep[:5])
    if a is not None and a > limit:
        findings.append(Finding(
            c.name, "critical", "auc", a, limit, "",
            f"a bag-of-character-n-grams classifier predicts {positive} from surface "
            f"form alone at AUC {a} (shuffled baseline {a_shuf})", examples))
    elif sep is not None and sep > limit:
        findings.append(Finding(
            c.name, "warn", "separability", sep, limit, "",
            f"the label is predictable in the {negative} direction (AUC {a}, "
            f"separability {sep} against a shuffled baseline of {sep_shuf}); at this "
            f"sample size that is within the estimator's noise, so read the baseline "
            f"before acting on it", examples))
    return CheckResult(metrics, findings)


# --- the judged tier -----------------------------------------------------------------


def needs_rubric(*keys: str):
    """A `CorpusCheck.validate` demanding this property's judge wording in the config."""
    def check(spec: dict, params: dict, prop: str = "") -> None:
        rubrics = spec.get("rubrics") or {}
        assert prop in rubrics, (
            f"corpus property {prop!r} is judged, so the stage entry needs its wording "
            f"under `rubrics.{prop}` (keys: {', '.join(keys)}). Judge wording is part "
            f"of the scientific record and lives in the config, never in code.")
        missing = [k for k in keys if k not in (rubrics[prop] or {})]
        assert not missing, f"`rubrics.{prop}` is missing {missing}"
    return check


def _validate_coverage(spec: dict, params: dict, prop: str = "") -> None:
    needs_rubric("system", "user")(spec, params, prop)
    assert spec.get("units"), (
        "principle_coverage needs the stage entry's `units:` list -- the judge must be "
        "given a closed set to label against, and an empty principle can only be "
        "reported against a set that says which principles were expected")


def validate_spec(spec: dict) -> None:
    """Fail fast on a corpus_check stage entry that cannot run as configured.

    Called from the operator at build_stages time, before the first generation stage
    spends anything -- a typo in a rubric key must not surface only after a corpus has
    been paid for.
    """
    unknown = [p["property"] for p in _instances(spec)
               if p["property"] not in CORPUS_CHECKS]
    if unknown:
        raise ValueError(f"unknown corpus properties {unknown}. "
                         f"Registered: {sorted(CORPUS_CHECKS)}")
    # Unknown names are caught above for every instance -- a typo stays a typo whether
    # or not it is enabled. `validate` runs only on what will execute: a disabled judged
    # property has no reason to carry rubric wording.
    for inst in _enabled(spec):
        check = CORPUS_CHECKS[inst["property"]]
        if check.validate is not None:
            check.validate({**spec, **inst},
                           {**check.defaults, **(inst.get("params") or {})},
                           check.name)


def _rubric(c: Corpus) -> dict:
    """The judge wording for this property, from the config. Never from code."""
    return (c.spec.get("rubrics") or {})[str(c.spec["property"])]


def _listify(raw: Any) -> list[str]:
    """Split a judge's comma/semicolon-separated tag body into clean tokens."""
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = [p.strip(" \t\r\n.'\"") for p in re.split(r"[,;\n]+", str(raw or ""))]
    return [p for p in parts if p and p.lower() not in ("none", "n/a", "-")]


def _fmt_record(c: Corpus, i: int) -> dict:
    """Record fields a rubric's `user` template may interpolate."""
    out = {"record_id": c.ids[i], "group": c.groups[i]}
    for role in ("unit", "members"):
        v = resolve_field(c.fields.get(role), c.records[i]) if role in c.fields else None
        out[role] = ", ".join(_listify(v)) if isinstance(v, (list, tuple)) else (v or "")
    return out


def judge(c: Corpus, tags: tuple[str, ...], render, *, max_tokens: int = 700,
          indices: list[int] | None = None, system: str | None = None):
    """Judge a seeded sample of documents, resuming from a per-property checkpoint.

    Returns `(labels, stats)`, `labels` mapping record id to the parsed tag dict for
    every document successfully judged. A document whose judge call failed or whose
    reply would not parse is simply ABSENT -- never defaulted to a label. A checker must
    be able to say "I do not know"; the alternative once scored 1,266 documents 0.0
    behind a dead API key.

    `render(index)` builds the user message; `indices` restricts the sample, so a
    property applying to part of the corpus still gets its full sample of that part;
    `system` overrides the rubric's (a multi-pass property has more than one).
    """
    from .stage_runtime import (JUDGE_NO_REASONING, Checkpoint, call_tagged,
                       model_cfg, run_items)

    m = model_cfg(c.ctx.cfg, c.spec.get("model"))
    sys_msg = system if system is not None else _rubric(c)["system"]
    stage = f"corpus:{c.name}"

    n = c.params.get("sample")
    picked = c.sample(None if n is None else int(n), pool=indices)
    ckpt = Checkpoint(Path(c.run_dir) / f"{c.stage}_{c.name}.partial.jsonl",
                      key="record_id")
    items = [{"record_id": c.ids[i], "_index": i} for i in picked]

    def one(item: dict) -> dict:
        parsed = call_tagged(
            c.ctx.client, c.ctx.usage, m["model"],
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": render(item["_index"])}],
            0.0, max_tokens, stage, tags,
            # JUDGE_NO_REASONING is the default, not a mandate: a model block that
            # declares its own `reasoning:` wins (mandatory-reasoning models 400
            # on enabled:false and take `{effort: low}` instead).
            extra={**JUDGE_NO_REASONING, **(m.get("extra_body") or {})})
        return {"record_id": item["record_id"], **parsed}

    done = run_items(items, one, c.workers, stage, ckpt,
                     max_fail_pct=float(c.params.get("max_fail_pct", 5.0)))
    labels = {r["record_id"]: {k: r[k] for k in tags if k in r} for r in done}
    stats = {"sampled": len(picked), "judged": len(labels),
             "unjudged": len(picked) - len(labels), "judge_model": m["model"]}
    return labels, stats


def check_quality_filter(c: Corpus) -> CheckResult:
    """GDM's final autorater: which documents an unrealistic/low-quality filter would cut.

    Like `embedding_dedup` this computes the removal set and reports it rather than
    applying it. Per-document verdicts land in the sidecar as `quality_verdict` /
    `quality_flaw`, so a downstream filter runs off the same numbers a human read, and so
    a later property can consume `label.quality_flaw` as a column.

    The judge returns a `flaw` tag as well as a verdict, because the drop RATE is the
    less useful half: 6% dropped tells you to regenerate, while 6% dropped and all of it
    `implausible_stakes` tells you which prompt to fix. Both are reported per `group`,
    since a failure concentrated in one trait is a prompt bug and one spread evenly is a
    recipe bug.

    A document the judge could not rate is `unjudged`, never `keep` -- the alternative
    silently passes a corpus whose API key died halfway.
    """
    labels, stats = judge(c, ("verdict", "flaw", "why"),
                          lambda i: _rubric(c)["user"].format(document=c.texts[i],
                                                              **_fmt_record(c, i)),
                          max_tokens=int(c.params["max_tokens"]))

    index = {rid: i for i, rid in enumerate(c.ids)}
    verdicts = {"keep": 0, "drop": 0}
    flaws: dict[str, int] = {}
    unrecognised: dict[str, int] = {}
    per_group: dict[str, dict[str, int]] = {}
    for rid, lab in labels.items():
        v = str(lab.get("verdict", "")).strip().lower()
        if v not in verdicts:
            unrecognised[v] = unrecognised.get(v, 0) + 1
            continue
        verdicts[v] += 1
        i = index.get(rid)
        per_group.setdefault(c.groups[i] if i is not None else "",
                             {"keep": 0, "drop": 0})[v] += 1
        if v == "drop":
            for tag in _listify(lab.get("flaw")) or ["unspecified"]:
                flaws[tag.lower()] = flaws.get(tag.lower(), 0) + 1

    n = sum(verdicts.values())
    rate = verdicts["drop"] / n if n else 0.0
    lo, hi = wilson(verdicts["drop"], n)
    dropped = [rid for rid, lab in labels.items()
               if str(lab.get("verdict", "")).strip().lower() == "drop"]

    metrics = {**stats, "rated": n, "distribution": verdicts,
               "drop_rate": round(rate, 4), "drop_rate_ci95": [lo, hi],
               "flaw_distribution": dict(sorted(flaws.items(), key=lambda kv: -kv[1])),
               "unrecognised_verdicts": unrecognised,
               "by_group": {g: {**d, "n": sum(d.values()),
                                "drop_rate": round(d["drop"] / sum(d.values()), 4)
                                if sum(d.values()) else 0.0}
                            for g, d in sorted(per_group.items())}}

    findings: list[Finding] = []
    if n:
        flag(c, findings, "critical", "drop_rate", rate, "drop_rate_max",
             f"an autorater would cut {verdicts['drop']} of {n} documents as unrealistic "
             f"or low quality (95% CI {lo:.0%}-{hi:.0%})"
             + (f"; most common flaw {max(flaws, key=flaws.get)!r}" if flaws else ""),
             lambda: dropped)
        # Against the REST of the corpus, not against the whole of it: a group is part
        # of the corpus rate it would be compared to, so with two equal groups a total
        # concentration reaches exactly 2x and a "2x the corpus" test can never fire.
        worst = max(per_group.items(), key=lambda kv: kv[1]["drop"], default=None)
        if worst and len(per_group) > 1 and worst[1]["drop"] >= 3:
            g, d = worst
            gr = d["drop"] / max(sum(d.values()), 1)
            rest_n = n - sum(d.values())
            rest = (verdicts["drop"] - d["drop"]) / rest_n if rest_n else 0.0
            if gr >= 2 * rest:
                findings.append(Finding(
                    c.name, "warn", "drop_rate", round(gr, 4), round(rest, 4), g,
                    f"group {g!r} fails at {gr:.0%} against {rest:.0%} across the rest "
                    f"of the corpus; a failure this concentrated is usually one prompt, "
                    f"not the recipe",
                    tuple(rid for rid in dropped
                          if c.groups[index[rid]] == g)[:5]))
    return CheckResult(metrics, findings,
                       {rid: {"quality_verdict": str(lab.get("verdict", "")).lower(),
                              "quality_flaw": ", ".join(_listify(lab.get("flaw"))),
                              # A boolean beside the verdict, because `corpus_filter`'s
                              # `drop_when` tests label truthiness and both "keep" and
                              # "drop" are truthy strings -- this is the key a filter names.
                              "quality_drop": str(lab.get("verdict", "")).strip().lower() == "drop",
                              # The judge's one-sentence reason, so a drop set can be read
                              # without re-running the judge.
                              "quality_why": str(lab.get("why", "")).strip()}
                        for rid, lab in labels.items()})


def check_applies_vs_conflicts(c: Corpus) -> CheckResult:
    """Does each document's reasoning RESOLVE a tension between values, or APPLY one?

    The primary outcome of the chunking experiment: if conflict-resolution guidance is
    not learnable from a single chunk, the share of conflict documents should rise with
    group size. Reported overall and per `group` with Wilson intervals, so a difference
    between arms can be read as a difference rather than as noise.
    """
    labels, stats = judge(c, ("mode", "why"), lambda i: _rubric(c)["user"].format(
        document=c.texts[i], **_fmt_record(c, i)))

    modes = {"conflict": 0, "application": 0, "indeterminate": 0}
    per_group: dict[str, dict[str, int]] = {}
    unrecognised: dict[str, int] = {}
    index = {rid: i for i, rid in enumerate(c.ids)}
    for rid, lab in labels.items():
        mode = str(lab.get("mode", "")).strip().lower()
        if mode not in modes:
            unrecognised[mode] = unrecognised.get(mode, 0) + 1
            continue
        modes[mode] += 1
        i = index.get(rid)
        g = c.groups[i] if i is not None else ""
        per_group.setdefault(g, {k: 0 for k in modes})[mode] += 1

    n = sum(modes.values())
    rate = modes["conflict"] / n if n else 0.0
    lo, hi = wilson(modes["conflict"], n)
    ind_share = modes["indeterminate"] / n if n else 0.0

    metrics = {**stats, "distribution": modes, "classified": n,
               "conflict_rate": round(rate, 4), "conflict_rate_ci95": [lo, hi],
               "indeterminate_share": round(ind_share, 4),
               "unrecognised_modes": unrecognised,
               "by_group": {g: {**d, "n": sum(d.values()),
                                "conflict_rate": round(d["conflict"] / sum(d.values()), 4)
                                if sum(d.values()) else 0.0,
                                "conflict_rate_ci95": list(
                                    wilson(d["conflict"], sum(d.values())))}
                            for g, d in sorted(per_group.items())}}

    def ids_where(mode: str) -> list[str]:
        return [rid for rid, lab in labels.items()
                if str(lab.get("mode", "")).lower() == mode]

    findings: list[Finding] = []
    if n:
        flag(c, findings, "warn", "conflict_rate", rate, "conflict_rate_min",
             f"only {rate:.0%} of documents resolve a tension between values; the rest "
             f"apply a single one (95% CI {lo:.0%}-{hi:.0%}, n={n})",
             lambda: ids_where("application"), low=True)
        flag(c, findings, "warn", "indeterminate_share", ind_share,
             "indeterminate_share_max",
             f"the judge could not classify {ind_share:.0%} of documents -- read the "
             f"rubric before reading the ratio", lambda: ids_where("indeterminate"))
    return CheckResult(metrics, findings, labels)


def check_principle_coverage(c: Corpus) -> CheckResult:
    """Which principles does each document ACTUALLY engage, per a judge?

    For a one-unit-per-document arm coverage is true by construction; for the
    whole-constitution and cluster arms this is the only way to know what the corpus
    covers, which is what makes those arms evaluable at all. `off_target_rate` is the
    other half: documents whose engaged principles exclude the unit they came from.
    """
    declared = [str(u) for u in c.spec["units"]]
    labels, stats = judge(c, ("principles", "why"), lambda i: _rubric(c)["user"].format(
        document=c.texts[i], principles="\n".join(f"- {u}" for u in declared),
        **_fmt_record(c, i)))

    counts: dict[str, int] = {}
    off_target, with_unit = 0, 0
    index = {rid: i for i, rid in enumerate(c.ids)}
    for rid, lab in labels.items():
        engaged = _listify(lab.get("principles"))
        for p in engaged:
            counts[p] = counts.get(p, 0) + 1
        i = index.get(rid)
        unit = _fmt_record(c, i)["unit"] if i is not None else ""
        if unit:
            with_unit += 1
            off_target += unit not in engaged

    empty = [u for u in declared if not counts.get(u)]
    norm_h = entropy([counts.get(u, 0) for u in declared]) if declared else 1.0
    rate = off_target / with_unit if with_unit else 0.0

    metrics = {**stats, "engaged_counts": dict(sorted(counts.items())),
               "principles_declared": len(declared), "principles_seen": len(counts),
               "empty_principles": empty, "normalized_entropy": round(norm_h, 4),
               "off_target_rate": round(rate, 4), "documents_with_a_unit": with_unit}
    findings: list[Finding] = []
    if empty:
        findings.append(Finding(
            c.name, "critical", "empty_principles", len(empty), 0, "",
            f"{len(empty)} principle(s) are engaged by no sampled document: "
            f"{', '.join(empty[:5])}"))
    if declared:
        flag(c, findings, "warn", "normalized_entropy", norm_h,
             "min_normalized_entropy",
             f"coverage is concentrated on a few of the {len(declared)} principles",
             low=True)
    if with_unit:
        flag(c, findings, "warn", "off_target_rate", rate, "off_target_rate_max",
             f"{rate:.0%} of documents never engage the unit they were generated from",
             labels)
    return CheckResult(metrics, findings, labels)


def check_chunk_attribution(c: Corpus) -> CheckResult:
    """For a k>1 arm, does a document engage all of its member chunks, or collapse?

    A grouped arm that ignores its second chunk is a k=1 arm wearing a costume, and
    would null the headline result without anything else noticing. `effective_k` is the
    mean number of members a document actually engages.
    """
    index = {rid: i for i, rid in enumerate(c.ids)}
    members = {i: _listify(resolve_field(c.fields.get("members"), r))
               for i, r in enumerate(c.records)}
    grouped = [i for i, mem in members.items() if len(mem) > 1]
    if not grouped:
        return CheckResult({"note": "no document has more than one member chunk; "
                                    "there is no attribution to measure",
                            "documents": len(c.records), "grouped_documents": 0})

    labels, stats = judge(c, ("engaged", "why"), lambda i: _rubric(c)["user"].format(
        document=c.texts[i], **_fmt_record(c, i)), indices=grouped)

    per_k: dict[str, dict[str, float]] = {}
    engaged_counts, all_engaged, total, member_total = [], 0, 0, 0
    for rid, lab in labels.items():
        mem = members.get(index.get(rid), [])
        if not mem:
            continue
        hit = {m for m in _listify(lab.get("engaged")) if m in mem}
        total += 1
        member_total += len(mem)
        engaged_counts.append(len(hit))
        all_engaged += len(hit) == len(mem)
        b = per_k.setdefault(str(len(mem)), {"n": 0, "engaged": 0, "all": 0})
        b["n"] += 1
        b["engaged"] += len(hit)
        b["all"] += len(hit) == len(mem)

    eff_k = sum(engaged_counts) / total if total else 0.0
    mean_k = member_total / max(total, 1)
    ratio = eff_k / mean_k if mean_k else 0.0
    rate = all_engaged / total if total else 0.0
    lo, hi = wilson(all_engaged, total)
    for b in per_k.values():
        b["effective_k"] = round(b["engaged"] / b["n"], 3)
        b["all_members_engaged_rate"] = round(b["all"] / b["n"], 4)

    metrics = {**stats, "grouped_documents": len(grouped), "scored": total,
               "effective_k": round(eff_k, 3), "mean_k": round(mean_k, 3),
               "effective_k_ratio": round(ratio, 4),
               "all_members_engaged_rate": round(rate, 4),
               "all_members_engaged_ci95": [lo, hi],
               "by_k": dict(sorted(per_k.items()))}
    findings: list[Finding] = []
    if total:
        flag(c, findings, "critical", "effective_k_ratio", ratio, "attribution_min",
             f"documents engage {eff_k:.2f} of their {mean_k:.2f} member chunks on "
             f"average -- this arm behaves like a smaller k than it declares",
             labels, low=True)
    return CheckResult(metrics, findings, labels)


def _norm_pattern(text: str) -> str:
    """Collapse a pattern's wording so two scans can be compared, using this module's own
    tokeniser so it cannot drift from every n-gram metric."""
    return " ".join(words(str(text)))[:120]


def _as_list(parsed: Any, *keys: str) -> list:
    """A model's JSON body as a list, whether it returned one or wrapped it in a key."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for k in keys:
            if isinstance(parsed.get(k), list):
                return parsed[k]
        vals = [v for v in parsed.values() if isinstance(v, list)]
        if len(vals) == 1:
            return vals[0]
    return []


def _candidate(raw: Any) -> dict | None:
    """One scan's proposed pattern, normalised. None when it carries no usable name.

    Tolerant by design: a scan that answers with bare strings instead of the requested
    objects still contributes votes, it just contributes no examples.
    """
    if isinstance(raw, str):
        return {"name": raw.strip(), "category": "", "description": raw.strip(),
                "examples": [], "count": 0} if raw.strip() else None
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("pattern") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "category": str(raw.get("category") or "").strip().lower(),
        "description": str(raw.get("description") or name).strip(),
        "examples": [str(x).strip() for x in _listify(raw.get("examples"))][:2],
        "count": int(raw.get("count") or 0) if str(raw.get("count", "")).strip().lstrip(
            "-").isdigit() else 0,
    }


def _scan(c: Corpus) -> tuple[list[dict], dict]:
    """PASS 1. Ask a long-context model what recurs across each batch of documents.

    Deliberately open-ended: the prompt names pattern *categories* but never a pattern, so
    the corpus's characteristic tic is free to be something nobody thought to look for.
    Seeding it with suspicions is how you find only your suspicions.

    Documents are shuffled before batching. Without that a batch is a run of consecutive
    generation ids, so a shared scenario topic reads as a corpus-wide pattern.

    The cache key is the batch CONTENT, never a document or run id: keying on an id that
    embeds the run is what made every sweep arm re-pay to scan identical documents.
    """
    from .stage_runtime import (JUDGE_NO_REASONING, Checkpoint, call_json, model_cfg, run_items)

    p, rub = c.params, _rubric(c)
    m = model_cfg(c.ctx.cfg, c.spec.get("model"))
    size, n_batches = int(p["batch_size"]), int(p["batches"])

    pool = list(range(len(c.records)))
    random.Random(c.seed).shuffle(pool)
    batches = [sorted(pool[i * size:(i + 1) * size]) for i in range(n_batches)]
    batches = [b for b in batches if len(b) >= 2]
    keys = [hashlib.sha256("\n\x00\n".join(c.texts[i] for i in b).encode()).hexdigest()
            for b in batches]
    ckpt = Checkpoint(Path(c.run_dir) / f"{c.stage}_{c.name}.scans.jsonl",
                      key="batch_key")

    def one(item: dict) -> dict:
        idxs = item["_idxs"]
        docs = "\n\n---\n\n".join(f"[{c.ids[i]}]\n{c.texts[i]}" for i in idxs)
        parsed, _ = call_json(
            c.ctx.client, c.ctx.usage, m["model"], rub["scan_system"],
            rub["scan_user"].format(documents=docs, n=len(idxs)),
            0.0, int(p["scan_max_tokens"]), f"corpus:{c.name}:scan",
            # JUDGE_NO_REASONING is the default, not a mandate: a model block that
            # declares its own `reasoning:` wins (mandatory-reasoning models 400
            # on enabled:false and take `{effort: low}` instead).
            extra={**JUDGE_NO_REASONING, **(m.get("extra_body") or {})})
        found = [_candidate(x) for x in _as_list(parsed, "patterns", "results")]
        return {"batch_key": item["batch_key"],
                "patterns": [f for f in found if f]}

    cached = sum(1 for k in keys if k in ckpt.done)
    done = run_items([{"batch_key": k, "_idxs": b} for k, b in zip(keys, batches)],
                     one, c.workers, f"corpus:{c.name}:scan", ckpt,
                     max_fail_pct=float(p.get("max_fail_pct", 25.0)))

    # Batch provenance rides along: the vote is over how many INDEPENDENT scans named a
    # thing, so two mentions inside one batch must not count twice.
    candidates = [{**pat, "batch": r["batch_key"]}
                  for r in done for pat in r["patterns"]]
    return candidates, {"batches_scanned": len(batches), "batches_from_cache": cached,
                        "batch_size": size, "scan_model": m["model"],
                        "candidates_proposed": len(candidates)}


def _name_clusters(c: Corpus, clusters: list[list[dict]]) -> list[str]:
    """One call to name every merged cluster. Falls back to the best-attested member.

    A naming failure must never lose a cluster that survived the vote, so the fallback is
    unconditional and the call is made once for all clusters rather than once each.
    """
    fallback = [max(g, key=lambda d: len(d["description"]))["name"] for g in clusters]
    rub = _rubric(c)
    if not c.params.get("name_clusters", True) or "merge_system" not in rub:
        return fallback
    from .stage_runtime import JUDGE_NO_REASONING, call_json, model_cfg

    try:
        m = model_cfg(c.ctx.cfg, c.spec.get("model"))
        listing = "\n\n".join(
            f"{i + 1}. " + "; ".join(sorted({d["name"] for d in g}))
            for i, g in enumerate(clusters))
        parsed, _ = call_json(
            c.ctx.client, c.ctx.usage, m["model"], rub["merge_system"],
            rub["merge_user"].format(clusters=listing, n=len(clusters)),
            0.0, int(c.params["merge_max_tokens"]), f"corpus:{c.name}:merge",
            # JUDGE_NO_REASONING is the default, not a mandate: a model block that
            # declares its own `reasoning:` wins (mandatory-reasoning models 400
            # on enabled:false and take `{effort: low}` instead).
            extra={**JUDGE_NO_REASONING, **(m.get("extra_body") or {})})
        names = [str(x).strip() for x in _as_list(parsed, "names", "clusters")]
        return [names[i] if i < len(names) and names[i] else fallback[i]
                for i in range(len(clusters))]
    except Exception:                     # noqa: BLE001 - naming is cosmetic, never fatal
        return fallback


def _merge(c: Corpus, candidates: list[dict]) -> tuple[list[dict], dict]:
    """PASS 2. Pool the candidates, merge the ones that mean the same thing, then vote.

    The vote is over MERGED clusters, not raw strings, and that ordering is the whole
    point. "opens by validating the user's feelings" and "begins with an empathy
    sentence" are one pattern found twice; compared as strings they are two patterns
    found once each, and `min_scans` then discards both -- silently, exactly on the
    corpus's most widespread tic, which is the one most likely to be described in several
    ways.

    Merging is embedding cosine over `name + description`, reusing the featuriser the
    dedup check uses: descriptions are short, which is the length that measurement showed
    it discriminates best at. Deterministic, free, and no extra model call.
    """
    p = c.params
    if not candidates:
        return [], {"patterns_merged": 0}

    from .embeddings import DEFAULT_MODEL, embed

    texts = [f"{d['name']}. {d['description']}" for d in candidates]
    X = embed(texts, model=str(p.get("embed_model") or DEFAULT_MODEL))
    G = X @ X.T
    thresh = float(p["merge_cosine"])
    a_idx, b_idx = _np_triu_pairs(G, thresh)
    # Every join, with the cosine that made it: the same/different classes overlap at any
    # threshold (see the registry block), so a merge is a proposal to be read, not a fact.
    merges = sorted(({"a": candidates[int(a)]["name"], "b": candidates[int(b)]["name"],
                      "cosine": round(float(G[a, b]), 3)}
                     for a, b in zip(a_idx, b_idx)),
                    key=lambda d: d["cosine"])[:15]
    groups = _components(len(candidates), zip(a_idx, b_idx))
    grouped = {i for g in groups for i in g}
    clusters = ([[candidates[i] for i in g] for g in groups]
                + [[candidates[i]] for i in range(len(candidates)) if i not in grouped])

    # Rank before naming so one call names only what will survive.
    scored = sorted(clusters, key=lambda g: (-len({d["batch"] for d in g}), g[0]["name"]))
    min_scans = int(p["min_scans"])
    kept = [g for g in scored if len({d["batch"] for d in g}) >= min_scans]
    dropped = len(scored) - len(kept)
    kept = kept[:int(p["max_patterns"])]
    names = _name_clusters(c, kept)

    patterns = []
    for name, g in zip(names, kept):
        cats = [d["category"] for d in g if d["category"]]
        examples = [e for d in g for e in d["examples"]][:int(p["max_examples"])]
        patterns.append({
            "name": name,
            "category": max(set(cats), key=cats.count) if cats else "",
            "description": max(g, key=lambda d: len(d["description"]))["description"],
            "examples": examples,
            "scans": len({d["batch"] for d in g}),
            "aliases": sorted({d["name"] for d in g if d["name"] != name})[:5],
        })
    return patterns, {"patterns_merged": len(clusters), "patterns_surviving": len(kept),
                      "discarded_below_min_scans": dropped, "merge_cosine": thresh,
                      "weakest_merges": merges}


def _np_triu_pairs(G, thresh: float):
    """Upper-triangle index pairs of a similarity matrix at or above `thresh`."""
    import numpy as np

    return np.where(np.triu(G >= thresh, k=1))


def _verdicts(raw: Any, ids: list[str]) -> dict[str, str]:
    """A classifier reply as {record_id: STRICT|BROAD|NO}, ignoring anything else."""
    out: dict[str, str] = {}
    known = set(ids)
    for row in _as_list(raw, "verdicts", "results", "documents"):
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or row.get("record_id") or "").strip()
        v = str(row.get("verdict") or row.get("answer") or "").strip().upper()
        if rid in known and v in ("STRICT", "BROAD", "NO"):
            out[rid] = v
    return out


def _rate(c: Corpus, patterns: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    """PASS 3. Build one classifier per pattern and run it over a large random sample.

    Two verdicts per document, per GDM: STRICT (unambiguously present) and BROAD (loosely
    present). Reporting only one of them hides the disagreement that matters -- a pattern
    at 60% broad and 8% strict is a tendency, one at 40% broad and 38% strict is a
    template.

    Documents are batched into each classifier call with per-document verdicts, because
    this pass is thousands of tiny calls and the per-call overhead dominates the content.

    **The sanity check.** An LLM-written classifier drifts from the pattern it was written
    for, and a drifted classifier reports a confident number about nothing. Before its
    verdicts are believed, each classifier is run against the verbatim snippets the scan
    itself cited as instances of that pattern. A classifier that answers NO to its own
    evidence is marked `reliable: false` and its rates are reported but flagged. Negatives
    come from the other patterns' snippets -- real text from the same corpus that
    exemplifies a different tic, which is a sharper contrast than random documents.
    """
    from .stage_runtime import (JUDGE_NO_REASONING, Checkpoint, call_json, model_cfg, run_items)

    p, rub = c.params, _rubric(c)
    m = model_cfg(c.ctx.cfg, c.spec.get("rate_model") or c.spec.get("model"))
    picked = c.sample(int(p["sample"]) if p["sample"] else None)
    size = max(int(p["rate_batch_size"]), 1)
    ckpt = Checkpoint(Path(c.run_dir) / f"{c.stage}_{c.name}.ratings.jsonl", key="key")

    def classifier(pat: dict, others: list[dict]) -> tuple[str, str]:
        pos = "\n".join(f"- {e}" for e in pat["examples"]) or "(none cited)"
        neg = "\n".join(f"- {e}" for o in others for e in o["examples"][:1])[:2000]
        return (rub["rate_system"], {"pattern": pat["name"],
                                     "category": pat["category"],
                                     "description": pat["description"],
                                     "examples": pos,
                                     "counter_examples": neg or "(none cited)"})

    items = []
    for pi, pat in enumerate(patterns):
        others = [o for o in patterns if o is not pat]
        sys_msg, vars_ = classifier(pat, others)
        # Sanity batch first: the pattern's own evidence, as pseudo-documents.
        if pat["examples"]:
            items.append({"key": f"sanity::{pi}::{_norm_pattern(pat['name'])}",
                          "_kind": "sanity", "_pi": pi, "_sys": sys_msg, "_vars": vars_,
                          "_docs": [(f"ex{j}", e) for j, e in enumerate(pat["examples"])]})
        for start in range(0, len(picked), size):
            chunk = picked[start:start + size]
            items.append({"key": f"rate::{pi}::{_norm_pattern(pat['name'])}::{start}",
                          "_kind": "rate", "_pi": pi, "_sys": sys_msg, "_vars": vars_,
                          "_docs": [(c.ids[i], c.texts[i]) for i in chunk]})

    def one(item: dict) -> dict:
        body = "\n\n---\n\n".join(f"[{rid}]\n{txt}" for rid, txt in item["_docs"])
        parsed, _ = call_json(
            c.ctx.client, c.ctx.usage, m["model"], item["_sys"],
            rub["rate_user"].format(documents=body, n=len(item["_docs"]),
                                    **item["_vars"]),
            0.0, int(p["rate_max_tokens"]), f"corpus:{c.name}:rate",
            # JUDGE_NO_REASONING is the default, not a mandate: a model block that
            # declares its own `reasoning:` wins (mandatory-reasoning models 400
            # on enabled:false and take `{effort: low}` instead).
            extra={**JUDGE_NO_REASONING, **(m.get("extra_body") or {})})
        return {"key": item["key"], "pi": item["_pi"], "kind": item["_kind"],
                "verdicts": _verdicts(parsed, [rid for rid, _ in item["_docs"]])}

    done = run_items(items, one, c.workers, f"corpus:{c.name}:rate", ckpt,
                     max_fail_pct=float(p.get("max_fail_pct", 25.0)))

    rated: dict[int, dict[str, str]] = {i: {} for i in range(len(patterns))}
    sanity: dict[int, list[str]] = {i: [] for i in range(len(patterns))}
    for r in done:
        pi = int(r["pi"])
        if r["kind"] == "sanity":
            sanity[pi].extend(r["verdicts"].values())
        else:
            rated.setdefault(pi, {}).update(r["verdicts"])

    rows, labels = [], {}
    for pi, pat in enumerate(patterns):
        v = rated.get(pi, {})
        n = len(v) or 1
        strict = sum(1 for x in v.values() if x == "STRICT")
        broad = strict + sum(1 for x in v.values() if x == "BROAD")
        checks_ = sanity.get(pi) or []
        recall = (sum(1 for x in checks_ if x != "NO") / len(checks_)) if checks_ else None
        rows.append({
            "pattern": pat["name"], "category": pat["category"],
            "description": pat["description"],
            "scans": pat["scans"], "aliases": pat["aliases"],
            "rated": len(v),
            "broad_share": round(broad / n, 4), "strict_share": round(strict / n, 4),
            "broad_ci95": list(wilson(broad, len(v))),
            "reliable": recall is None or recall >= float(p["sanity_min_recall"]),
            "sanity_recall": None if recall is None else round(recall, 3),
            "examples": pat["examples"],
        })
        for rid, verdict in v.items():
            if verdict != "NO":
                key = "patterns_strict" if verdict == "STRICT" else "patterns_broad"
                labels.setdefault(rid, {})
                labels[rid][key] = ", ".join(
                    filter(None, [labels[rid].get(key, ""), pat["name"]]))

    rows.sort(key=lambda r: -r["broad_share"])
    return rows, labels, {"rate_model": m["model"], "sampled": len(picked),
                          "rate_batch_size": size,
                          "unreliable_classifiers": sum(1 for r in rows
                                                        if not r["reliable"])}


def check_pattern_scan(c: Corpus) -> CheckResult:
    """Discover the corpus's OWN recurring tics, then measure how far each has spread.

    GDM's three passes, in order: SCAN batches open-endedly for what recurs, CLUSTER the
    candidates and keep only what independent scans agreed on, AUTORATE a large sample
    against each survivor. Nothing about the document type is known here -- every word the
    models see comes from the stage entry's `rubrics:` block, so the same property runs
    over difficult advice, pre-action deliberation or model-eval-model unchanged.

    Report-only by default. GDM's own filter-and-retrain ablations moved the corpus's
    style without moving the eval scores, so a flagged pattern is a hypothesis about the
    data, not a demonstrated cause of anything. The two documented uses are feeding high
    scorers back into the rewrite stage as explicit anti-patterns, and filter-and-retrain
    ablations on the worst offenders.
    """
    # `patterns:` supplies the list instead of discovering it, skipping both paid
    # discovery passes. This is what makes a cross-corpus claim possible: two independent
    # scans find two different pattern sets, so "100% here, 0% there" cannot come from
    # comparing them -- it comes from running ONE corpus's classifiers over the other.
    given = c.params.get("patterns")
    if given:
        patterns = [{"name": str(d["name"]), "category": str(d.get("category") or ""),
                     "description": str(d.get("description") or d["name"]),
                     "examples": [str(x) for x in _listify(d.get("examples"))],
                     "scans": int(d.get("scans") or 0), "aliases": []}
                    for d in given]
        metrics: dict[str, Any] = {"patterns_supplied": len(patterns),
                                   "scan_skipped": True}
    else:
        candidates, scan_stats = _scan(c)
        patterns, merge_stats = _merge(c, candidates)
        metrics = {**scan_stats, **merge_stats}

    if not patterns:
        return CheckResult(metrics, [Finding(
            c.name, "info", "patterns_surviving", 0, None, "",
            f"{metrics.get('candidates_proposed', 0)} candidates proposed across "
            f"{metrics.get('batches_scanned', 0)} scans, none named by "
            f"{c.params['min_scans']} independent scans after merging")])

    rows, labels, rate_stats = _rate(c, patterns)
    metrics.update(rate_stats)
    metrics["patterns"] = rows
    metrics["by_category"] = {
        cat: round(sum(r["broad_share"] for r in rows if r["category"] == cat), 4)
        for cat in sorted({r["category"] for r in rows if r["category"]})}
    metrics["docs_matching_any_broad"] = round(len(labels) / max(rate_stats["sampled"],
                                                                1), 4)

    findings: list[Finding] = []
    top = rows[0]
    findings.append(Finding(
        c.name, "info", "broad_share", top["broad_share"], None, top["pattern"],
        f"the corpus's most widespread recurring pattern -- {top['pattern']!r} "
        f"({top['category'] or 'uncategorised'}) -- is loosely present in "
        f"{top['broad_share']:.0%} of sampled documents and unambiguously present in "
        f"{top['strict_share']:.0%} ({len(rows)} patterns survived "
        f"{c.params['min_scans']}+ independent scans)",
        tuple(rid for rid, lab in labels.items()
              if top["pattern"] in lab.get("patterns_strict", "")
              or top["pattern"] in lab.get("patterns_broad", ""))[:5]))
    for r in rows:
        if not r["reliable"]:
            findings.append(Finding(
                c.name, "warn", "sanity_recall", r["sanity_recall"],
                float(c.params["sanity_min_recall"]), r["pattern"],
                f"the classifier written for {r['pattern']!r} does not recognise the "
                f"snippets the scan cited as instances of it, so its "
                f"{r['broad_share']:.0%} is a number about something else"))
    flag(c, findings, "critical", "broad_share", top["broad_share"],
         "top_pattern_broad_max",
         f"{top['pattern']!r} reaches {top['broad_share']:.0%} of the corpus",
         lambda: tuple(labels)[:5], scope=top["pattern"])
    return CheckResult(metrics, findings, labels)


# --- the registry --------------------------------------------------------------------
# One block, deliberately: every property and its default thresholds side by side.
# `roles` defaults to ("text", "id"); every threshold was set from MEASURED values on
# the 2,203-document difficult-advice corpus in output/model_eval_model/20260805_133015/
# (stage_1_source.jsonl, grouped by trait_id, ~1,040 words per document):
#
#     top_8gram_share  0.449      mean_4gram_jaccard  0.0007-0.0037
#     distinct_2       0.338-0.448   duplicate_share  0.0 (0 candidate pairs)
#     top_opener_share 0.252      length cv           0.153   group delta <= 0.14
#     mean_cosine      0.860      effective_rank_frac 0.645
#     entropy: trait_id (9 values) 1.00, domain (495 values) 0.80
#
# Two of those drive the defaults: character-n-gram cosine has a HIGH FLOOR (~0.86 for
# two unrelated same-genre documents), so `effective_rank_frac` is the discriminating
# half of that property; mean 4-gram Jaccard is length-dependent (~0.003 at 1,000
# words), so its gate fires only on severe collapse.

CORPUS_CHECKS: dict[str, CorpusCheck] = {
    "ngram_diversity": CorpusCheck(
        "ngram_diversity", check_ngram_diversity,
        # top_8gram_share_max matches the gate model_eval_model.yaml already ships, so
        # the migrated post-hoc path keeps its historical behaviour.
        defaults={"top_8gram_share_max": 0.20, "mean_jaccard_max": 0.15,
                  "distinct_2_min": 0.30, "sample": 100, "min_group_docs": 5},
        min_docs=5,
        doc="repeated long n-grams, pairwise 4-gram overlap and bigram variety per group"),
    "embedding_dedup": CorpusCheck(
        "embedding_dedup", check_embedding_dedup,
        # Measured on the same corpus, over the 68-word `situation` text (the unit GDM
        # dedups), potion-base-8M:
        #
        #                  mean_pairwise  mean_nn  nn_p95  nn_p99  nn_max
        #   situation  68w      0.371       0.743   0.828   0.856   0.886
        #   user      203w      0.593       0.813   0.887   0.909   0.930
        #   full doc 1044w      0.757       0.887   0.924   0.934   0.940
        #
        # cosine_min was 0.90, chosen to sit above that corpus's worst pair (0.886) on
        # the assumption the corpus was clean. It is not: re-measured 2026-08-13 over all
        # 2,205 scenarios (scratch/measure_dedup_threshold.py), 0.90 and above yield
        # ZERO pairs, while the 0.886 pair is a genuine clone -- the same postdoc /
        # funding-cliff / borderline-significance story under two trait ids and two
        # domain labels. A gate calibrated to fire zero times on the corpus it was
        # measured on is not a gate. The sweep:
        #
        #   thresh  pairs  clusters  would_drop  drop%  max_cluster
        #     0.90      0         0           0   0.0%            0
        #     0.88      4         3           4   0.2%            3
        #     0.86     11         9          11   0.5%            3
        #     0.84     43        22          39   1.8%           13
        #     0.82    162        43         125   5.7%           30
        #     0.80    441        62         263  11.9%           66
        #     0.75   3764        76        1006  45.6%          342
        #
        # 0.86 is the usable floor: real clones, clusters still small enough to audit by
        # hand (max 3). Below 0.84 components chain -- by 0.75 a single cluster holds 342
        # documents, which means the threshold has dropped under the genre's own
        # similarity and is measuring the corpus rather than its duplicates.
        #
        # NOTE this catches duplication, NOT concentration: at 0.86 it removes 0.5% of a
        # corpus that holds 46.8% of its mass in ten domains. 226 DISTINCT small-business
        # scenarios are not near-duplicates of each other. Concentration is a generation-
        # side problem -- see the `diversity:` block on the scenarios stage.
        # mean_nn_cosine_max 0.80 is ~1.5x the headroom over the measured 0.743.
        # max_mean_words 300 comes straight from the table: by 1,044 words the floor has
        # climbed to 0.76 and the thresholds no longer discriminate.
        defaults={"cosine_min": 0.86, "drop_share_max": 0.02,
                  "mean_nn_cosine_max": 0.80, "max_mean_words": 300,
                  "sample": 2000, "batch_size": 256, "model": None},
        min_docs=20,
        doc="semantic near-duplicates over static embeddings, and the removal set a "
            "GDM-style dedup stage would drop"),
    "label_leakage": CorpusCheck(
        "label_leakage", check_label_leakage,
        roles=("text", "label", "id"),
        defaults={"surface_auc_max": 0.65, "min_per_class": 20, "positive": None},
        min_docs=40,
        doc="cross-validated AUC of a surface classifier predicting the label"),

    # --- judged ---------------------------------------------------------------------
    # Sampled by default and resumable, so a re-run at a larger sample pays only the
    # difference. All four are report-only until their numbers have been seen on a real
    # corpus; `gate: true` in a config turns one on.
    "quality_filter": CorpusCheck(
        "quality_filter", check_quality_filter,
        paid=True,
        # drop_rate_max 0.10 is a placeholder from GDM's framing (a final filter is a
        # trim, not a rewrite), NOT a measured value -- nothing in this repo has run an
        # autorater over a finished corpus yet. It ships gate: false; measure before
        # trusting it, and record the number here as the block above does.
        defaults={"sample": 300, "drop_rate_max": 0.10, "max_tokens": 500},
        min_docs=30, est_calls=lambda p, n: min(int(p["sample"] or n), n),
        validate=needs_rubric("system", "user"),
        doc="share of documents a GDM-style autorater would cut as unrealistic or "
            "low-quality, with the flaw breakdown"),
    "applies_vs_conflicts": CorpusCheck(
        "applies_vs_conflicts", check_applies_vs_conflicts,
        paid=True,
        defaults={"sample": 300, "conflict_rate_min": 0.15,
                  "indeterminate_share_max": 0.20},
        min_docs=30, est_calls=lambda p, n: min(int(p["sample"] or n), n),
        validate=needs_rubric("system", "user"),
        doc="share of documents resolving a value tension vs applying a single value"),
    "principle_coverage": CorpusCheck(
        "principle_coverage", check_principle_coverage,
        paid=True,
        defaults={"sample": 300, "min_normalized_entropy": 0.75,
                  "off_target_rate_max": 0.30},
        min_docs=30, est_calls=lambda p, n: min(int(p["sample"] or n), n),
        validate=_validate_coverage,
        doc="which principles each document actually engages, judged independently "
            "of which unit it was generated from"),
    "chunk_attribution": CorpusCheck(
        "chunk_attribution", check_chunk_attribution,
        roles=("text", "id", "members"), paid=True,
        defaults={"sample": 300, "attribution_min": 0.60},
        min_docs=30, est_calls=lambda p, n: min(int(p["sample"] or n), n),
        validate=needs_rubric("system", "user"),
        doc="whether a k>1 document engages all its member chunks or collapses onto one"),
    "pattern_scan": CorpusCheck(
        "pattern_scan", check_pattern_scan,
        paid=True,
        # Two models: `model:` scans (long context, sees batch_size whole documents at
        # once) and `rate_model:` classifies (thousands of tiny calls -- use the cheapest
        # adequate one). `rate_model` falls back to `model` when unset.
        #
        # merge_cosine 0.35 is measured, but on a PROXY: 15 hand-written descriptions of
        # 5 patterns, three wordings each, as independent scans might phrase them (no
        # real scan output existed yet). Cosine over those:
        #
        #   same pattern, different wording   n=15   min 0.226  mean 0.449  max 0.621
        #   different patterns                n=90   min 0.009  mean 0.208  max 0.492
        #
        # The classes OVERLAP, so no threshold is clean and this step is noisy by nature:
        # at 0.35, 8/90 unrelated pairs merge wrongly and 1/15 same-pattern pairs stay
        # split. Biased low deliberately -- a missed merge splits a pattern's votes and
        # drops it below min_scans SILENTLY, which is the failure the merge exists to
        # prevent, while a wrong merge is visible in `aliases`. Every merge is reported
        # (cluster members + the cosine that joined them), so it can be audited rather
        # than trusted. Re-measure on real scan output.
        #
        # sanity_min_recall 0.5 asks only that a classifier recognise half of its own
        # cited evidence; below that it is not measuring the pattern it was written for.
        defaults={"batches": 30, "batch_size": 25, "min_scans": 2, "max_patterns": 20,
                  "sample": 1000, "rate_batch_size": 8, "max_examples": 4,
                  "merge_cosine": 0.35, "name_clusters": True, "embed_model": None,
                  "sanity_min_recall": 0.5, "top_pattern_broad_max": 1.01,
                  "scan_max_tokens": 2500, "merge_max_tokens": 900,
                  "rate_max_tokens": 900},
        min_docs=24,
        # Split across the two models. Scan batches + one naming call on `model`; per
        # pattern one sanity batch plus one call per rate_batch_size documents on
        # `rate_model`. Patterns are unknown before the scan runs, so the rating half
        # prices the worst case: max_patterns survivors.
        est_calls=lambda p, n: {
            # A supplied `patterns:` list skips discovery entirely and prices exactly,
            # since the survivor count is then known up front.
            "model": 0 if p.get("patterns") else int(p["batches"]) + 1,
            "rate_model": (len(p["patterns"]) if p.get("patterns")
                           else int(p["max_patterns"])) * (
                1 + -(-min(int(p["sample"] or n), n)
                      // max(int(p["rate_batch_size"]), 1)))},
        validate=needs_rubric("scan_system", "scan_user", "rate_system", "rate_user"),
        doc="GDM scan -> cluster -> autorate: the corpus's own recurring tics, each "
            "measured at STRICT and BROAD frequency"),
}


# --- the driver ----------------------------------------------------------------------


_SEVERITY_ORDER = {"error": 0, "critical": 1, "warn": 2, "info": 3}


def _instances(spec: dict) -> list[dict]:
    """Normalise the stage entry's `properties:` list into per-instance dicts."""
    out = []
    for raw in spec.get("properties") or []:
        p = dict(raw) if isinstance(raw, dict) else {"property": str(raw)}
        p.setdefault("as", p["property"])
        out.append(p)
    aliases = [p["as"] for p in out]
    dupes = sorted({a for a in aliases if aliases.count(a) > 1})
    assert not dupes, (f"duplicate corpus-check aliases {dupes} -- give one of them an "
                       f"`as:` name so their report keys do not collide")
    return out


TIERS = ("surface", "judged")


def _split(raw: Any) -> list[str] | None:
    """A comma-separated string or list into clean names; None stays None."""
    if raw is None:
        return None
    items = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    return [str(x).strip() for x in items if str(x).strip()]


def select_properties(spec: dict, *, only: Any = None, skip: Any = None,
                      tier: Any = None) -> dict:
    """Return `spec` with each property's `enabled` flag resolved against a selection.

    The one place a run decides WHICH checks execute. Selection is a spec transform and
    not a branch inside the driver, so the pipeline stage, the `synth check` verb and
    `--estimate` all see the same enabled set and cannot disagree about what a run cost
    or covered.

    Precedence, narrowest last: a config's own `enabled: false` is a floor nothing turns
    back on; `only` then restricts; `skip` and `tier` subtract. Names match either the
    registry `property` or the instance `as:` alias, so an aliased instance
    (`scenario_dupes`) is addressable by the name it is reported under.

    Args:
        spec: A `corpus_check` stage entry.
        only: Run just these properties/aliases.
        skip: Exclude these properties/aliases.
        tier: Keep only `surface` (free) or `judged` (paid) properties.

    Raises:
        AssertionError: On a name that matches no property in this stage entry, or an
            unknown tier -- a silent no-op selection is how you convince yourself a check
            ran when it did not.
    """
    only, skip = _split(only), _split(skip)
    tiers = _split(tier)
    if tiers:
        bad = [t for t in tiers if t not in TIERS]
        assert not bad, f"unknown tier {bad}; known: {list(TIERS)}"

    insts = _instances(spec)
    known = {n for p in insts for n in (p["property"], p["as"])}
    for label, names in (("only", only), ("skip", skip)):
        unknown = sorted(set(names or []) - known)
        assert not unknown, (
            f"--{label} names {unknown}, which no property in stage "
            f"{spec.get('name', 'corpus')!r} matches. Available: {sorted(known)}")

    out = []
    for p in insts:
        names = {p["property"], p["as"]}
        on = bool(p.get("enabled", True))
        if only is not None:
            on = on and bool(names & set(only))
        if skip:
            on = on and not (names & set(skip))
        if tiers:
            on = on and CORPUS_CHECKS[p["property"]].tier in tiers
        out.append({**p, "enabled": on})
    return {**spec, "properties": out}


def _enabled(spec: dict) -> list[dict]:
    """The instances that will actually run."""
    return [p for p in _instances(spec) if p.get("enabled", True)]


def is_paid(spec: dict) -> bool:
    """Whether any ENABLED property in this stage entry calls a model."""
    return any(CORPUS_CHECKS[p["property"]].paid for p in _enabled(spec)
               if p["property"] in CORPUS_CHECKS)


def corpus_check_calls_by_model(spec: dict, n_docs: int) -> dict[str, int]:
    """API calls a corpus_check stage entry will make, keyed by MODEL BLOCK.

    Per model and not as a total, because a property may use more than one: `pattern_scan`
    sends ~30 long-context batches to a capable scanner and thousands of small ones to the
    cheapest classifier, and pricing both at one model's `assumed_tokens` understates the
    expensive half by an order of magnitude.

    An `est_calls` returning an int puts every call on the instance's own model. Returning
    `{role: calls}` splits them, where a role is a key on the stage entry or instance
    (`model`, `rate_model`) naming a block in the config's `models:`.
    """
    out: dict[str, int] = {}
    for inst in _enabled(spec):
        check = CORPUS_CHECKS.get(inst["property"])
        if check is None or not check.paid:
            continue
        assert check.est_calls is not None, (
            f"corpus property {check.name!r} is paid but declares no `est_calls`; "
            f"`--estimate` would price it at zero")
        resolved = {**spec, **inst}
        priced = check.est_calls({**check.defaults, **(inst.get("params") or {})}, n_docs)
        by_role = priced if isinstance(priced, dict) else {"model": int(priced)}
        for role, calls in by_role.items():
            # Zero-call roles are dropped rather than recorded: a pass that will not run
            # must not put its model in the estimate at all.
            if not int(calls):
                continue
            # Falls back to the role name so pure counting works on a spec with no
            # `model:`; declaring one is enforced by the operator, which is where a
            # missing model can actually be reported against a stage.
            key = resolved.get(role) or resolved.get("model") or role
            out[key] = out.get(key, 0) + int(calls)
    return out


def corpus_check_calls(spec: dict, n_docs: int) -> int:
    """Total API calls a corpus_check stage entry will make."""
    return sum(corpus_check_calls_by_model(spec, n_docs).values())


def load_labels(run_dir: Path, stage: str) -> dict[str, dict]:
    """Read the judged-label sidecar, if a previous run of this stage wrote one."""
    path = Path(run_dir) / f"{stage}_labels.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue                 # a torn line from a hard kill; that record re-judges
        rid = row.pop("record_id", None)
        if rid is not None:
            out.setdefault(str(rid), {}).update(row)
    return out


def run_corpus_checks(records: list[dict], spec: dict, *, run_dir: Path | str,
                      seed: int = 0, workers: int = 8, ctx: Any = None) -> dict:
    """Run every property the stage entry declares (`fields`, `axes`, `cross`,
    `rubrics`, `properties`) and return the report; `report["pass"]` is the AND of every
    gated property.

    Never mutates `records`. A property that cannot run is recorded as `skipped` or
    `errored` -- never omitted, never quietly passed. `ctx` (the pipeline Ctx) is
    required only when a paid property is declared; `seed` makes every sampled
    statistic reproducible.
    """
    run_dir = Path(run_dir)
    validate_spec(spec)
    stage = str(spec.get("name", "corpus"))
    fields = dict(spec.get("fields") or {})
    labels = load_labels(run_dir, stage)

    report: dict[str, Any] = {"stage": stage, "n_records": len(records),
                              "fields": fields, "seed": seed, "properties": {}}
    all_findings: list[Finding] = []
    ok = True
    spent_before = float(getattr(getattr(ctx, "usage", None), "usd", 0.0) or 0.0)
    # One Corpus per distinct `fields` mapping, re-pointed at each property in turn, so
    # properties reading the same fields share one tokenisation. Properties run in the
    # order the config lists them, so a `label.<key>` axis must be listed after the
    # property that produces that label.
    corpora: dict[str, Corpus] = {}

    for inst in _instances(spec):
        alias = inst["as"]
        check = CORPUS_CHECKS[inst["property"]]
        params = {**check.defaults, **(inst.get("params") or {})}
        gate = bool(inst.get("gate", False))
        entry: dict[str, Any] = {"property": check.name, "tier": check.tier,
                                 "gate": gate, "params": params}

        # Recorded, not omitted: a report must show what was deselected, or a narrowed
        # run reads afterwards like a clean full one.
        if not inst.get("enabled", True):
            entry.update({"status": "disabled", "pass": True,
                          "reason": "deselected for this run"})
            report["properties"][alias] = entry
            print(f"OFF   {alias}: deselected")
            continue

        inst_fields = {**fields, **(inst.get("fields") or {})}
        key = json.dumps(inst_fields, sort_keys=True, default=str)
        if key not in corpora:
            corpora[key] = Corpus(records=records, fields=inst_fields, params={},
                                  spec=spec, seed=seed, workers=workers,
                                  run_dir=run_dir, labels=labels, ctx=ctx)
        corpus = corpora[key]
        corpus.params, corpus.spec, corpus.name = params, {**spec, **inst}, alias

        missing = next((m for m in (corpus.missing_role(r) for r in check.roles) if m),
                       None)
        if missing is None and check.paid and ctx is None:
            # Not an error: an offline caller (a test, a `synth check` over a config
            # with no models block) simply cannot judge. Saying so beats erroring, and
            # beats passing silently.
            missing = ("this property judges documents, and no model context was "
                       "supplied to this run")
        if missing:
            entry.update({"status": "skipped", "pass": True, "reason": missing})
            report["properties"][alias] = entry
            print(f"SKIP  {alias}: {missing}")
            continue

        try:
            result = check.fn(corpus)
        except Exception as exc:  # noqa: BLE001 - recorded as a finding, never re-raised
            # An errored property FAILS the report whether or not it was gated. It used to
            # pass when ungated, which meant a property that crashed reported the same
            # verdict as one that ran clean: on 2026-08-13 a `pattern_scan` whose scan
            # calls all truncated printed "1 errored -- gated verdict PASS". That is rule 2
            # ("a check that cannot run says so; never a silent pass") violated one level
            # up, in the driver rather than in a checker. A gated failure still stops the
            # run via `on_fail`; an ungated one now at least refuses to call itself a pass.
            err = Finding(alias, "error", "exception", f"{type(exc).__name__}: {exc}",
                          None, "", f"the check itself failed: {type(exc).__name__}: {exc}")
            entry.update({"status": "errored", "pass": False,
                          "metrics": {}, "findings": [err.as_dict()]})
            report["properties"][alias] = entry
            all_findings.append(err)
            ok = False
            print(f"ERROR {alias}: {type(exc).__name__}: {exc}")
            continue

        for rid, lab in (result.labels or {}).items():
            labels.setdefault(str(rid), {}).update(lab)

        enforced = gate and len(records) >= check.min_docs
        critical = [f for f in result.findings if f.severity == "critical"]
        passed = (not enforced) or not critical
        entry.update({"status": "gated" if enforced else "reported", "pass": passed,
                      "metrics": result.metrics,
                      "findings": [f.as_dict() for f in result.findings]})
        if gate and not enforced:
            entry["reason"] = (f"{len(records)} records is below this property's "
                               f"min_docs={check.min_docs}; reported, not gated")
        report["properties"][alias] = entry
        all_findings.extend(result.findings)
        ok = ok and passed

    if labels:
        path = run_dir / f"{stage}_labels.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rid in sorted(labels):
                f.write(json.dumps({"record_id": rid, **labels[rid]},
                                   ensure_ascii=False) + "\n")
        report["labels_file"] = path.name

    all_findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.property))
    report["findings"] = [f.as_dict() for f in all_findings]
    report["counts"] = {sev: sum(1 for f in all_findings if f.severity == sev)
                        for sev in ("error", "critical", "warn", "info")}
    if ctx is not None and getattr(ctx, "usage", None) is not None:
        report["judge_spend_usd"] = round(float(ctx.usage.usd) - spent_before, 4)
    report["pass"] = ok
    report["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return report


def pattern_table(report: dict) -> str:
    """The per-pattern frequency table, as markdown. "" when no scan ran.

    GDM's headline output is a table, not a verdict -- "BLUF: 52%" is the finding. Emitted
    as markdown so it drops straight into a `*_results.md` mirror beside the report, and
    so two corpora's tables can be read side by side.
    """
    out = []
    for alias, entry in report.get("properties", {}).items():
        rows = (entry.get("metrics") or {}).get("patterns")
        if not rows:
            continue
        out.append(f"\n### {alias} ({entry['metrics'].get('rated_model', '')}"
                   f"{entry['metrics'].get('rate_model', '')}, "
                   f"n={entry['metrics'].get('sampled', 0)})\n")
        out.append("| pattern | category | broad % | strict % | scans | example |")
        out.append("|---|---|---:|---:|---:|---|")
        for r in rows:
            ex = (r["examples"][0] if r["examples"] else "").replace("|", "\\|")[:80]
            warn = "" if r["reliable"] else " ⚠︎"
            out.append(f"| {r['pattern']}{warn} | {r['category'] or '-'} | "
                       f"{r['broad_share']:.0%} | {r['strict_share']:.0%} | "
                       f"{r['scans']} | {ex} |")
    return "\n".join(out)


def print_summary(report: dict) -> None:
    """Print one line per property plus the loudest findings."""
    for alias, entry in report["properties"].items():
        status = entry["status"]
        tag = {"skipped": "SKIP ", "errored": "ERROR", "disabled": "OFF  ",
               "gated": "PASS " if entry["pass"] else "FAIL ",
               "reported": "----"}.get(status, "----")
        n = len(entry.get("findings") or [])
        print(f"{tag} {alias} ({status})" + (f" -- {n} finding(s)" if n else ""))
    for f in report.get("findings", [])[:5]:
        print(f"    [{f['severity']}] {f['property']}"
              + (f"/{f['scope']}" if f["scope"] else "") + f": {f['summary']}")
    c = report.get("counts", {})
    print(f">>> corpus checks: {c.get('critical', 0)} critical, {c.get('warn', 0)} warn, "
          f"{c.get('error', 0)} errored -- gated verdict "
          f"{'PASS' if report.get('pass') else 'FAIL'}")
