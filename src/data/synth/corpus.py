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
# Here rather than in checks.py so the in-pipeline stage and the post-hoc `synth check`
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


def _bottom_k_jaccard(a: list[int], b: list[int], k: int) -> float:
    """Bottom-k MinHash Jaccard estimate of two sorted sketches."""
    union = sorted(set(a) | set(b))[:k]
    if not union:
        return 0.0
    sa, sb = set(a), set(b)
    return sum(1 for h in union if h in sa and h in sb) / len(union)


def check_near_duplicates(c: Corpus) -> CheckResult:
    """Exact and near-duplicate documents.

    All-pairs Jaccard over 10k documents is 50M comparisons, so candidates come from LSH
    banding over bottom-k shingle sketches and only those are scored -- an approximation
    biased towards heavy overlaps, which is exactly what a near-duplicate is.
    """
    p = c.params
    k, nb, sh = int(p["sketch"]), int(p["bands"]), int(p["shingle"])
    rows = max(k // nb, 1)
    cap = int(p["max_bucket"])

    sketches: list[list[int]] = []
    exact: dict[str, list[int]] = {}
    for i, w in enumerate(c.tokens):
        exact.setdefault(hashlib.sha256(" ".join(w).encode()).hexdigest(), []).append(i)
        grams = {zlib.crc32(" ".join(w[j:j + sh]).encode())
                 for j in range(max(len(w) - sh + 1, 0))}
        sketches.append(sorted(grams)[:k])

    exact_groups = [g for g in exact.values() if len(g) > 1]
    # Band only ONE representative per exact-duplicate class: identical documents
    # collide in every band, so a corpus with a large block of them would otherwise
    # score every pair -- restoring the O(n^2) this exists to avoid, to re-derive what
    # the sha256 map already knows.
    reps = [g[0] for g in exact.values()]

    buckets: dict[tuple, list[int]] = {}
    for i in reps:
        s = sketches[i]
        if len(s) < rows:            # too short to shingle; exact match is all we get
            continue
        for b in range(nb):
            band = tuple(s[b * rows:(b + 1) * rows])
            if band:
                buckets.setdefault((b, band), []).append(i)

    pairs: dict[tuple[int, int], float] = {}
    oversized = 0
    for members in buckets.values():
        # An oversized bucket is a finding, not a workload: scoring C(m,2) pairs inside
        # it costs more than the rest of the check and says nothing its existence did
        # not already say.
        if len(members) > cap:
            oversized += 1
            continue
        for a, b in itertools.combinations(members, 2):
            if (a, b) not in pairs:
                pairs[(a, b)] = _bottom_k_jaccard(sketches[a], sketches[b], k)

    thresh = float(p["jaccard_min"])
    near = {pair: j for pair, j in pairs.items() if j >= thresh}
    by_rep = {g[0]: g for g in exact.values()}       # representatives -> full classes
    dup_idx = {i for g in exact_groups for i in g}
    for pair in near:
        for rep in pair:
            dup_idx.update(by_rep.get(rep, [rep]))
    share = len(dup_idx) / max(len(c.records), 1)

    metrics = {
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_docs": sum(len(g) for g in exact_groups),
        "near_duplicate_pairs": len(near),
        "duplicate_share": round(share, 4),
        "candidate_pairs_scored": len(pairs),
        "distinct_texts_banded": len(reps),
        "oversized_buckets_skipped": oversized,
        "worst_pairs": [{"a": c.ids[a], "b": c.ids[b], "jaccard": round(j, 3)}
                        for (a, b), j in sorted(near.items(), key=lambda kv: -kv[1])[:5]],
    }
    findings: list[Finding] = []
    flag(c, findings, "critical", "duplicate_share", share, "dup_share_max",
         f"{len(dup_idx)} of {len(c.records)} documents are exact or near-duplicates "
         f"(Jaccard >= {thresh})",
         lambda: (c.ids[i] for i in sorted(dup_idx)))
    if oversized:
        findings.append(Finding(
            c.name, "warn", "oversized_buckets_skipped", oversized, cap, "",
            f"{oversized} LSH bucket(s) held more than {cap} distinct documents and "
            f"were not pair-scored; that many documents sharing a shingle band is "
            f"itself a collapse signal"))
    return CheckResult(metrics, findings)


def check_opening_collapse(c: Corpus) -> CheckResult:
    """Documents that begin the same way -- the earliest collapse signal, one pass.

    Measured twice, because exact matching misses the interesting half: on a real
    1,389-document corpus the three most common openings were reorderings of one
    construction (155 documents), which exact matching reports as three unremarkable
    openers. The `*_reordered` keys group openings by their word SET instead.
    """
    k = int(c.params["opening_words"])
    openers: dict[str, list[int]] = {}
    bags: dict[tuple[str, ...], list[int]] = {}
    labels: dict[tuple[str, ...], str] = {}
    for i, w in enumerate(c.tokens):
        if not w:
            continue
        head = w[:k]
        openers.setdefault(" ".join(head), []).append(i)
        bag = tuple(sorted(set(head)))
        bags.setdefault(bag, []).append(i)
        labels.setdefault(bag, " ".join(head))

    n = max(len(c.records), 1)

    def summarise(groups: dict, render) -> tuple[float, Any, list[int], float, list]:
        shared = {key: idxs for key, idxs in groups.items() if len(idxs) > 1}
        share = sum(len(v) for v in shared.values()) / n
        ranked = sorted(shared.items(), key=lambda kv: (-len(kv[1]), render(kv[0])))
        top_key, top_idxs = (ranked[0] if ranked else (None, []))
        return (share, top_key, top_idxs, len(top_idxs) / n,
                [{"opening": render(key), "docs": len(v)} for key, v in ranked[:10]])

    share, top_opener, top_idxs, top_share, ranked = summarise(openers, lambda x: x)
    r_share, r_key, r_idxs, r_top, r_ranked = summarise(bags, lambda b: labels[b])

    metrics = {
        "shared_opening_share": round(share, 4),
        "distinct_openings": len(openers),
        "top_opener": top_opener or "",
        "top_opener_share": round(top_share, 4),
        "top_openings": ranked,
        "shared_opening_share_reordered": round(r_share, 4),
        "distinct_openings_reordered": len(bags),
        "top_opener_reordered": labels.get(r_key, "") if r_key else "",
        "top_opener_share_reordered": round(r_top, 4),
        "top_openings_reordered": r_ranked,
    }
    findings: list[Finding] = []
    hit = flag(c, findings, "critical", "top_opener_share", top_share,
               "top_opener_share_max",
               f"{len(top_idxs)} of {n} documents open with {top_opener!r}",
               [c.ids[i] for i in top_idxs])
    if not hit:
        flag(c, findings, "critical", "top_opener_share_reordered", r_top,
             "top_opener_share_max",
             f"{len(r_idxs)} of {n} documents open with the same words in some order, "
             f"e.g. {labels.get(r_key, '')!r} -- no single wording is common enough to "
             f"notice, the construction is",
             [c.ids[i] for i in r_idxs])
    return CheckResult(metrics, findings)


def check_length_profile(c: Corpus) -> CheckResult:
    """Document length distribution overall and per group.

    A low coefficient of variation is a template fingerprint. A per-group mean well off
    the corpus mean is the cheap cousin of the surface-shortcut classifier: whatever
    distinguishes the groups is visible in length alone.
    """
    lens = [float(len(w)) for w in c.tokens]
    n = len(lens) or 1
    mean = sum(lens) / n
    var = sum((x - mean) ** 2 for x in lens) / n
    cv = (var ** 0.5) / mean if mean else 0.0
    ordered = sorted(lens)

    by_group = {}
    findings: list[Finding] = []
    for group, idxs in c.by_group.items():
        gl = [lens[i] for i in idxs]
        gmean = sum(gl) / len(gl) if gl else 0.0
        delta = abs(gmean - mean) / mean if mean else 0.0
        by_group[group] = {"docs": len(gl), "mean_words": round(gmean, 1),
                           "delta_vs_corpus": round(delta, 3)}
        flag(c, findings, "warn", "delta_vs_corpus", delta, "group_mean_delta_max",
             f"group mean length {gmean:.0f} words is {delta:.0%} off the corpus "
             f"mean {mean:.0f}", [c.ids[i] for i in idxs], scope=group or "corpus", nd=3)

    metrics = {"mean_words": round(mean, 1), "cv": round(cv, 4),
               "p5": _percentile(ordered, 0.05), "p50": _percentile(ordered, 0.50),
               "p95": _percentile(ordered, 0.95), "by_group": by_group}
    flag(c, findings, "warn", "cv", cv, "cv_min",
         f"document lengths vary by only {cv:.1%} of the mean -- a template fingerprint",
         c.ids, low=True)
    return CheckResult(metrics, findings)


def check_field_balance(c: Corpus) -> CheckResult:
    """Value distribution over each declared axis, and empty cells of each cross.

    "Did every planned bucket get documents", without knowing what the buckets are. Axes
    are dotted record paths, or `label.<key>` for a judged annotation. Only OBSERVED
    values are crossed, so a bucket nobody planned is not reported as missing.
    """
    axes = [str(a) for a in (c.spec.get("axes") or [])]
    crosses = [[str(a) for a in cross] for cross in (c.spec.get("cross") or [])]
    metrics: dict[str, Any] = {"axes": {}, "cross": {}}
    findings: list[Finding] = []

    def _vals(path: str) -> list[tuple[str, ...]]:
        """One tuple of values per record -- always a tuple, so nothing downstream has
        to re-ask whether an axis is multi-label."""
        out = []
        for v in c.column(path):
            if isinstance(v, (list, tuple)):
                out.append(tuple(str(x) for x in v))     # multi-label axis
            else:
                out.append((str(v) if v is not None else "",))
        return out

    columns = {a: _vals(a) for a in axes}
    for axis, vals in columns.items():
        counts: dict[str, int] = {}
        for v in vals:
            for one in v:
                counts[one] = counts.get(one, 0) + 1
        total = sum(counts.values()) or 1
        top, top_n = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
        norm_h = entropy(list(counts.values()))
        share = top_n / total
        metrics["axes"][axis] = {"values": len(counts),
                                 "counts": dict(sorted(counts.items())),
                                 "max_share": round(share, 3),
                                 "normalized_entropy": round(norm_h, 3)}
        flag(c, findings, "warn", "normalized_entropy", norm_h,
             "min_normalized_entropy",
             f"axis {axis} is unbalanced across its {len(counts)} values",
             scope=axis, low=True, nd=3)
        flag(c, findings, "warn", "max_share", share, "max_share",
             f"{share:.0%} of documents sit in {axis}={top!r}", scope=axis, nd=3)

    for cross in crosses:
        key = "/".join(cross)
        missing_axis = [a for a in cross if a not in columns]
        if missing_axis:
            metrics["cross"][key] = {
                "note": f"axes not declared in `axes:`: {missing_axis}"}
            continue
        observed = [sorted({v for val in columns[a] for v in val}) for a in cross]
        seen: set[tuple[str, ...]] = set()
        for i in range(len(c.records)):
            seen.update(itertools.product(*(columns[a][i] for a in cross)))
        empty = [combo for combo in itertools.product(*observed) if combo not in seen]
        buckets = math.prod(len(o) for o in observed)
        metrics["cross"][key] = {"buckets": buckets,
                                 "empty": ["/".join(e) for e in empty[:20]],
                                 "empty_count": len(empty)}
        flag(c, findings, "critical", "empty_count", len(empty),
             "max_empty_cross_buckets",
             f"{len(empty)} of {buckets} {key} buckets have no documents: "
             f"{', '.join('/'.join(e) for e in empty[:3])}", scope=key, nd=0)
    return CheckResult(metrics, findings)


def check_feature_diversity(c: Corpus) -> CheckResult:
    """How much of the representation space the corpus actually occupies.

    Mean pairwise cosine says how similar documents are on average; effective rank (the
    perplexity of the singular-value distribution) says how many independent directions
    they span, and is the discriminating half -- same-genre prose already scores ~0.86
    cosine, while effective rank as a fraction of the sample runs 0.65 for a healthy
    corpus against 0.03 for an identical one. Both are SURFACE measures.
    """
    import numpy as np

    idx = c.sample(int(c.params["sample"]))
    X = hashed_features([c.texts[i] for i in idx])
    G = X @ X.T
    n = len(idx)
    off = (G.sum() - np.trace(G)) / max(n * (n - 1), 1)
    svals = np.linalg.svd(X, compute_uv=False)
    pk = svals / (float(svals.sum()) or 1.0)
    eff_rank = float(np.exp(-(pk * np.log(np.clip(pk, 1e-12, None))).sum()))
    frac = eff_rank / max(n, 1)

    metrics = {"sampled": n, "mean_pairwise_cosine": round(float(off), 4),
               "effective_rank": round(eff_rank, 2),
               "effective_rank_frac": round(frac, 4)}
    findings: list[Finding] = []
    flag(c, findings, "critical", "mean_pairwise_cosine", float(off), "mean_cosine_max",
         f"documents are {float(off):.0%} similar to each other on average",
         [c.ids[i] for i in idx])
    flag(c, findings, "critical", "effective_rank_frac", frac,
         "effective_rank_frac_min",
         f"{n} documents span only {eff_rank:.1f} effective dimensions",
         [c.ids[i] for i in idx], low=True)
    return CheckResult(metrics, findings)


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
    for inst in _instances(spec):
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
    from .core import (JUDGE_NO_REASONING, Checkpoint, call_tagged,
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
            extra={**(m.get("extra_body") or {}), **JUDGE_NO_REASONING})
        return {"record_id": item["record_id"], **parsed}

    done = run_items(items, one, c.workers, stage, ckpt,
                     max_fail_pct=float(c.params.get("max_fail_pct", 5.0)))
    labels = {r["record_id"]: {k: r[k] for k in tags if k in r} for r in done}
    stats = {"sampled": len(picked), "judged": len(labels),
             "unjudged": len(picked) - len(labels), "judge_model": m["model"]}
    return labels, stats


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
    """Collapse a discovered pattern's wording so two scans can be compared, using this
    module's own tokeniser so it cannot drift from every n-gram metric."""
    return " ".join(words(str(text)))[:120]


def check_pattern_scan(c: Corpus) -> CheckResult:
    """Discover the corpus's OWN recurring tics, then measure how far they spread.

    Three passes, after GDM's scan -> cluster -> autorate: SCAN batches and ask what
    keeps recurring (no rubric written in advance -- a corpus's characteristic tic is
    usually something nobody thought to look for); CLUSTER by keeping only patterns
    `min_scans` INDEPENDENT batches both named; AUTORATE a sample against the survivors.

    Report-only, always. The scan cache is keyed on BATCH CONTENT, never on a document
    or run id -- keying on an id that embeds the run is what made every sweep arm re-pay
    to scan identical documents last time this existed.
    """
    from .core import (JUDGE_NO_REASONING, Checkpoint, call_tagged, model_cfg,
                       run_items)

    p = c.params
    rub = _rubric(c)
    m = model_cfg(c.ctx.cfg, c.spec.get("model"))

    batch_size, n_batches = int(p["batch_size"]), int(p["batches"])
    rng = random.Random(c.seed)
    pool = list(range(len(c.records)))
    rng.shuffle(pool)
    batches = [sorted(pool[i * batch_size:(i + 1) * batch_size])
               for i in range(n_batches)]
    batches = [b for b in batches if len(b) >= 2]
    keys = [hashlib.sha256("\n\x00\n".join(c.texts[i] for i in b).encode()).hexdigest()
            for b in batches]
    # The same resume protocol every paid stage uses, not a second hand-rolled copy.
    ckpt = Checkpoint(Path(c.run_dir) / f"{c.stage}_{c.name}.scans.jsonl",
                      key="batch_key")

    def scan(item: dict) -> dict:
        idxs = item["_idxs"]
        docs = "\n\n---\n\n".join(f"[{c.ids[i]}]\n{c.texts[i]}" for i in idxs)
        parsed = call_tagged(
            c.ctx.client, c.ctx.usage, m["model"],
            [{"role": "system", "content": rub["scan_system"]},
             {"role": "user", "content": rub["scan_user"].format(
                 documents=docs, n=len(idxs),
                 seeded=", ".join(str(s) for s in (p.get("seed_patterns") or [])))}],
            0.0, int(p["scan_max_tokens"]), f"corpus:{c.name}:scan", ("patterns",),
            extra={**(m.get("extra_body") or {}), **JUDGE_NO_REASONING})
        return {"batch_key": item["batch_key"], "patterns": _listify(parsed["patterns"])}

    cached_before = sum(1 for k in keys if k in ckpt.done)
    done = run_items([{"batch_key": k, "_idxs": b} for k, b in zip(keys, batches)],
                     scan, c.workers, f"corpus:{c.name}:scan", ckpt,
                     max_fail_pct=float(p.get("max_fail_pct", 25.0)))

    # CLUSTER: a pattern survives only if independent scans found it.
    votes: dict[str, dict] = {}
    for r in done:
        for norm, original in {_norm_pattern(x): x for x in r["patterns"]}.items():
            if norm:
                v = votes.setdefault(norm, {"scans": 0, "label": original})
                v["scans"] += 1
    min_scans = int(p["min_scans"])
    survivors = [v["label"] for _, v in
                 sorted(votes.items(), key=lambda kv: -kv[1]["scans"])
                 if v["scans"] >= min_scans][:int(p["max_patterns"])]

    metrics: dict[str, Any] = {
        "batches_scanned": len(batches), "batches_from_cache": cached_before,
        "patterns_proposed": len(votes), "patterns_surviving": len(survivors),
        "survivors": survivors,
        "discarded_single_scan": sum(1 for v in votes.values()
                                     if v["scans"] < min_scans),
        "judge_model": m["model"],
    }
    if not survivors:
        return CheckResult(metrics, [Finding(
            c.name, "info", "patterns_surviving", 0, None, "",
            f"{len(votes)} candidate patterns proposed, none named by "
            f"{min_scans} independent scans")])

    # AUTORATE: how far does each survivor actually spread?
    listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(survivors))
    labels, stats = judge(c, ("matches", "why"),
                          lambda i: rub["rate_user"].format(
                              document=c.texts[i], patterns=listing),
                          max_tokens=int(p["rate_max_tokens"]),
                          system=rub["rate_system"])
    hits: dict[str, int] = {s: 0 for s in survivors}
    per_doc: list[int] = []
    by_norm = {_norm_pattern(s): s for s in survivors}
    for lab in labels.values():
        matched = {by_norm[_norm_pattern(x)] for x in _listify(lab.get("matches"))
                   if _norm_pattern(x) in by_norm}
        for s in matched:
            hits[s] += 1
        per_doc.append(len(matched))

    n = len(per_doc) or 1
    metrics.update({
        **stats,
        "pattern_hit_share": {s: round(v / n, 4) for s, v in
                              sorted(hits.items(), key=lambda kv: -kv[1])},
        "docs_matching_any": round(sum(1 for x in per_doc if x >= 1) / n, 4),
        "docs_matching_two_or_more": round(sum(1 for x in per_doc if x >= 2) / n, 4),
        "mean_patterns_per_doc": round(sum(per_doc) / n, 3),
    })
    top, top_n = max(hits.items(), key=lambda kv: kv[1])
    findings = [Finding(
        c.name, "info", "pattern_hit_share", round(top_n / n, 4), None, "",
        f"the corpus's most widespread recurring pattern -- {top!r} -- appears in "
        f"{top_n / n:.0%} of sampled documents ({len(survivors)} patterns survived "
        f"{min_scans}+ independent scans)",
        tuple(rid for rid, lab in labels.items()
              if any(_norm_pattern(x) == _norm_pattern(top)
                     for x in _listify(lab.get("matches"))))[:5])]
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
    "near_duplicates": CorpusCheck(
        "near_duplicates", check_near_duplicates,
        defaults={"jaccard_min": 0.70, "dup_share_max": 0.02,
                  "shingle": 5, "sketch": 64, "bands": 8, "max_bucket": 200},
        min_docs=10,
        doc="exact and near-duplicate documents via banded bottom-k shingle sketches"),
    "opening_collapse": CorpusCheck(
        "opening_collapse", check_opening_collapse,
        defaults={"opening_words": 8, "top_opener_share_max": 0.15},
        min_docs=10,
        doc="documents sharing their first N words -- the earliest collapse signal"),
    "length_profile": CorpusCheck(
        "length_profile", check_length_profile,
        defaults={"cv_min": 0.12, "group_mean_delta_max": 0.35},
        min_docs=10,
        doc="length distribution overall and per group; low variation is a template"),
    "field_balance": CorpusCheck(
        "field_balance", check_field_balance,
        roles=("id",),
        # Normalised entropy divides by ln(k), penalising a high-cardinality axis with a
        # natural long tail (the healthy 495-value `domain` axis scores 0.80). 0.75 keeps
        # the gate meaningful for categorical axes without flagging every free-form one.
        defaults={"min_normalized_entropy": 0.75, "max_share": 0.60,
                  "max_empty_cross_buckets": 0},
        min_docs=10,
        doc="per-axis value distribution and entropy; empty buckets of each cross"),
    "feature_diversity": CorpusCheck(
        "feature_diversity", check_feature_diversity,
        defaults={"sample": 400, "mean_cosine_max": 0.95,
                  "effective_rank_frac_min": 0.25},
        min_docs=20,
        doc="mean pairwise cosine and effective rank over hashed char n-grams"),
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
        defaults={"batches": 8, "batch_size": 12, "min_scans": 2, "max_patterns": 10,
                  "sample": 200, "scan_max_tokens": 900, "rate_max_tokens": 500,
                  "seed_patterns": []},
        min_docs=24,
        # One call per scan batch plus one per autorated document.
        est_calls=lambda p, n: int(p["batches"]) + min(int(p["sample"] or n), n),
        validate=needs_rubric("scan_system", "scan_user", "rate_system", "rate_user"),
        doc="GDM scan -> cluster -> autorate: the corpus's own recurring tics and how "
            "far each has spread"),
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


def is_paid(spec: dict) -> bool:
    """Whether any property in this stage entry calls a model."""
    return any(CORPUS_CHECKS[p["property"]].paid for p in _instances(spec)
               if p["property"] in CORPUS_CHECKS)


def corpus_check_calls(spec: dict, n_docs: int) -> int:
    """API calls a corpus_check stage entry will make, for `--estimate`."""
    total = 0
    for inst in _instances(spec):
        check = CORPUS_CHECKS.get(inst["property"])
        if check is None or not check.paid:
            continue
        assert check.est_calls is not None, (
            f"corpus property {check.name!r} is paid but declares no `est_calls`; "
            f"`--estimate` would price it at zero")
        total += int(check.est_calls({**check.defaults, **(inst.get("params") or {})},
                                     n_docs))
    return total


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
        entry: dict[str, Any] = {"property": check.name, "gate": gate, "params": params}

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
            err = Finding(alias, "error", "exception", f"{type(exc).__name__}: {exc}",
                          None, "", f"the check itself failed: {type(exc).__name__}: {exc}")
            entry.update({"status": "errored", "pass": not gate,
                          "metrics": {}, "findings": [err.as_dict()]})
            report["properties"][alias] = entry
            all_findings.append(err)
            ok = ok and not gate
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


def print_summary(report: dict) -> None:
    """Print one line per property plus the loudest findings."""
    for alias, entry in report["properties"].items():
        status = entry["status"]
        tag = {"skipped": "SKIP ", "errored": "ERROR",
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
