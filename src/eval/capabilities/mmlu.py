# ABOUTME: MMLU subset construction, prompting, answer parsing and paired accuracy
# ABOUTME: statistics — the shared core under the MMLU absolute-benchmark eval.

"""MMLU as an *absolute* capability check for the constitution-SFT arms.

`configs/capability_eval.yaml` gates on a pairwise preference judge, which has a known
blind spot the spec itself flags: a preference comparison cannot detect **both** arms
degrading together, and it rewards style over substance. MMLU closes exactly that gap —
it is scored against a fixed answer key, so every arm's number stands on its own and the
base model is a real anchor rather than another thing being compared.

Three design choices here are load-bearing:

1. **One fixed subset, shared by every arm.** The subset is derived deterministically
   from `seed`, so all arms answer literally the same questions. That makes the
   comparison *paired*: per-question outcomes line up across arms, which both tightens
   the interval on the difference and licenses McNemar. `subset_hash` exists so "every
   arm saw the same items" is something you can verify rather than assume.

2. **Stratified by subject, not sampled at random.** A uniform sample over the 14,042
   test rows would over-represent the large subjects (`professional_law` alone is 1,534
   rows) and give some subjects two questions. Taking a fixed count per subject keeps all
   57 present and makes the per-subject breakdown interpretable at small n.

3. **Choices are shuffled per question.** MMLU's answer key is not uniformly distributed
   over positions, and a model with a position bias (always answering "C") can score well
   above chance without knowing anything. The shuffle is seeded from the uid, so it is
   identical across arms *and* stable when the subset size changes — growing 10/subject to
   20/subject does not re-shuffle the first 10 or invalidate their cached generations.

Parsing is the other place this eval quietly goes wrong. On a thinking model the visible
answer arrives after a `<think>` block (CLAUDE.md gotcha 6), and a generation that runs
out of tokens mid-trace yields no answer at all — which scores as *wrong* and reads as
catastrophic capability loss when it is really a `max_tokens` bug. `parse_answer` reports
which tier matched so format compliance is measurable separately from correctness, and
the eval reports parse rate and truncation rate next to every accuracy number.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from src.eval.capabilities.capability_metrics import split_think

LETTERS = "ABCD"

# The standard 4-way MMLU grouping. Reported alongside the headline number because a
# uniform drop and a drop concentrated in STEM are different findings: the first suggests
# a format or decoding problem, the second an actual knowledge cost.
#
# `moral_scenarios`, `moral_disputes`, `philosophy` and `jurisprudence` are the subjects
# most plausibly *moved* by constitution SFT rather than merely damaged by it. Note this
# before looking: a rise there with flat everything else is a different story from noise.
SUBJECT_CATEGORY: dict[str, str] = {
    # STEM
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    # Humanities
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_law": "humanities",
    "world_religions": "humanities",
    # Social sciences
    "econometrics": "social_sciences",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # Other
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_medicine": "other",
    "global_facts": "other",
    "human_aging": "other",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "nutrition": "other",
    "professional_accounting": "other",
    "professional_medicine": "other",
    "virology": "other",
}

CATEGORIES = ("STEM", "humanities", "social_sciences", "other")


def _seed_for(seed: int, key: str) -> int:
    """Derive a stable 64-bit sub-seed from the run seed and a string key.

    Hashing rather than incrementing a counter is what makes per-question randomness
    independent of subset size: question `x`'s shuffle depends only on its uid, so
    enlarging the subset leaves every existing question — and its cached generation —
    untouched.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_split(split: str = "test", path: str = "cais/mmlu", name: str = "all") -> list[dict]:
    """Load one MMLU split into plain dicts, with a stable per-subject uid.

    Args:
        split: `test` (14,042 rows), `dev` (5 per subject, the canonical few-shot
            demonstrations), or `validation`.
        path: HuggingFace dataset id.
        name: Dataset config; `all` covers every subject.

    Returns:
        Rows of `{uid, subject, category, question, choices, answer_index}`, in file
        order. `uid` is `<subject>/<index within subject>`, so it is stable across
        subset sizes and independent of any sampling decision made later.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    seen: dict[str, int] = {}
    for rec in load_dataset(path, name, split=split):
        subject = str(rec["subject"])
        idx = seen.get(subject, 0)
        seen[subject] = idx + 1
        rows.append(
            {
                "uid": f"{subject}/{idx:05d}",
                "subject": subject,
                "category": SUBJECT_CATEGORY.get(subject, "other"),
                "question": str(rec["question"]).strip(),
                "choices": [str(c) for c in rec["choices"]],
                "answer_index": int(rec["answer"]),
            }
        )
    return rows


def build_subset(
    rows: Sequence[dict],
    per_subject: int,
    seed: int = 0,
    subjects: Sequence[str] | None = None,
    shuffle_choices: bool = True,
) -> list[dict]:
    """Draw a seeded, subject-stratified subset and fix each question's choice order.

    Args:
        rows: All candidate rows, as returned by `load_split`.
        per_subject: Questions to draw per subject. Subjects with fewer rows contribute
            all of them.
        seed: Run seed. Fixes both which questions are drawn and how choices are ordered.
        subjects: Restrict to these subjects; `None` uses all present.
        shuffle_choices: Permute each question's choices (recommended — see module
            docstring on position bias).

    Returns:
        Questions in `(subject, uid)` order, each with `choices` in presentation order
        and `answer_letter` pointing at the correct one *after* any shuffle.
    """
    if per_subject <= 0:
        raise ValueError(f"per_subject must be positive, got {per_subject}")

    wanted = set(subjects) if subjects else None
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        if wanted is not None and row["subject"] not in wanted:
            continue
        by_subject.setdefault(row["subject"], []).append(row)

    if wanted:
        missing = wanted - set(by_subject)
        if missing:
            raise ValueError(f"Unknown MMLU subjects: {sorted(missing)}")

    selected: list[dict] = []
    for subject in sorted(by_subject):
        pool = by_subject[subject]
        take = min(per_subject, len(pool))
        rng = np.random.default_rng(_seed_for(seed, f"select:{subject}"))
        # Draw as a PREFIX of a seeded permutation, not `rng.choice(size=take)`. The
        # permutation depends only on (seed, subject, pool size), so the draw at
        # per_subject=10 is a strict subset of the draw at 40 — which is what makes
        # enlarging the subset cache-safe. `choice(size=take)` consumes the RNG
        # differently per size and returns an unrelated set, silently re-drawing every
        # question and invalidating every cached generation.
        picks = sorted(rng.permutation(len(pool))[:take].tolist())
        for i in picks:
            selected.append(_present(pool[i], seed, shuffle_choices))
    return selected


def _present(row: dict, seed: int, shuffle_choices: bool) -> dict:
    """Fix one question's choice order and record where the correct answer landed."""
    choices = list(row["choices"])
    order = list(range(len(choices)))
    if shuffle_choices:
        rng = np.random.default_rng(_seed_for(seed, f"choices:{row['uid']}"))
        rng.shuffle(order)
    presented = [choices[i] for i in order]
    answer_position = order.index(row["answer_index"])
    return {
        "uid": row["uid"],
        "subject": row["subject"],
        "category": row["category"],
        "question": row["question"],
        "choices": presented,
        "answer_index": answer_position,
        "answer_letter": LETTERS[answer_position],
        "n_choices": len(presented),
    }


def subset_hash(questions: Iterable[dict]) -> str:
    """Content hash of a question subset, including presented choice order.

    Stamped into every arm's `run_meta.json`. Two arms whose hashes differ were not
    given the same exam, and no paired statistic over them means anything.
    """
    digest = hashlib.sha256()
    for q in sorted(questions, key=lambda r: r["uid"]):
        digest.update(q["uid"].encode())
        digest.update(b"\x00")
        digest.update("\x00".join(q["choices"]).encode())
        digest.update(b"\x00")
        digest.update(q["answer_letter"].encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def render_question(q: dict, include_answer: bool = False, cue: str = "Answer:") -> str:
    """Render one question block, optionally with its answer (for few-shot demos)."""
    lines = [q["question"]]
    lines += [f"{LETTERS[i]}. {choice}" for i, choice in enumerate(q["choices"])]
    block = "\n".join(lines)
    if include_answer:
        return f"{block}\n{cue} {q['answer_letter']}"
    return f"{block}\n{cue}"


def build_prompt(
    q: dict,
    shots: Sequence[dict] = (),
    instruction: str = "",
    cue: str = "Answer:",
) -> str:
    """Assemble the user-turn prompt for one question.

    The prompt deliberately carries *both* an explicit instruction and few-shot
    demonstrations in `Answer: X` form, because it has to serve two very different
    models under a parity constraint. The instruct-tuned arms follow the instruction;
    the raw base model ignores instructions but reliably continues the demonstrated
    pattern. Dropping either half would hand one arm a format advantage the other
    lacks, and a format advantage is indistinguishable from a capability advantage
    once it reaches the accuracy number.

    Args:
        q: The question, as produced by `build_subset`.
        shots: Few-shot demonstrations, ideally from the same subject's `dev` split.
        instruction: Leading instruction line(s).
        cue: The answer cue, repeated in demos and at the end of the prompt.

    Returns:
        The full user message.
    """
    parts: list[str] = []
    if instruction:
        parts.append(instruction.format(subject=q["subject"].replace("_", " ")).strip())
    parts += [render_question(s, include_answer=True, cue=cue) for s in shots]
    parts.append(render_question(q, include_answer=False, cue=cue))
    return "\n\n".join(parts)


def prompt_hash(prompt: str) -> str:
    """Short hash of a rendered prompt, used to invalidate stale cached generations."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


# Parsing tiers, strictest first. Each returns the LAST match, because a thinking model
# routinely considers several options before committing and the final statement is the
# answer. Tier names are recorded per question so a drop in format compliance can be read
# off the distribution rather than inferred.
_TIER_PATTERNS: tuple[tuple[str, str], ...] = (
    # `\boxed{C}` — the conventional final-answer marker in maths-style responses, and
    # what the Tulu-SFT arms actually emit. Checked FIRST, above the "Answer:" cue,
    # because a response that reasons "Answer: A ... no, \boxed{B}" has committed to B;
    # letting the earlier cue win would score the discarded candidate. Observed on the
    # live run: 139/570 of one arm's answers ended in \boxed{}, and before this tier
    # existed they were caught only incidentally by the last-resort `tail` rule.
    ("boxed", r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?\s*\}"),
    # "ANSWER: C", "Answer - C", "answer is **C**", "the correct answer is (C)"
    ("cue", r"(?i)\banswers?\b(?:\s+\w+){0,3}?[\s:\-–—]*[\*\(\[]{0,2}\s*([A-Z])\b"),
    # "Option C", "choice C"
    ("option", r"(?i)\b(?:option|choice)\s*[:\-]?\s*[\*\(\[]{0,2}([A-Z])\b"),
    # A bare letter standing alone as a line: "C", "**C**", "(C)", "C."
    ("bare", r"(?m)^\s*[\*\(\[]{0,2}\s*([A-Z])\s*[\)\].\*]{0,2}\s*$"),
)


def parse_answer(text: str, n_choices: int = 4) -> tuple[str | None, str]:
    """Extract the chosen letter from a model's visible answer.

    Args:
        text: The user-visible answer, with any `<think>` trace already stripped.
        n_choices: Number of valid options; letters beyond this are rejected rather
            than silently accepted (a model answering "E" has not answered).

    Returns:
        `(letter, tier)`. `letter` is `None` when nothing parseable was found, in which
        case `tier` is `"none"`. `tier` names which rule matched, so format compliance
        is measurable independently of correctness.
    """
    if not text or not text.strip():
        return None, "none"
    valid = set(LETTERS[:n_choices])

    for tier, pattern in _TIER_PATTERNS:
        matches = [m for m in re.findall(pattern, text) if m.upper() in valid]
        if matches:
            return matches[-1].upper(), tier

    # Last resort: a standalone letter anywhere in the final non-empty line. Restricted
    # to that line because a loose \b[A-D]\b over the whole text happily matches the "A"
    # in an enumerated restatement of the options and invents an answer that was never
    # given.
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        tail = [m for m in re.findall(r"\b([A-Z])\b", lines[-1]) if m in valid]
        if tail:
            return tail[-1], "tail"
    return None, "none"


def resolve_trace(content: str, reasoning: str | None) -> tuple[str, str]:
    """Split a completion into `(think, answer)` across every shape vLLM returns.

    Three shapes exist in the wild and the eval has to handle all of them, because
    getting this wrong reports a normally-reasoning model as having a collapsed
    `<think>` block (CLAUDE.md gotcha 2) — a false alarm on the exact failure mode this
    eval is supposed to detect:

    - **No reasoning parser configured.** The trace arrives inline in `content`, wrapped
      in `<think>` tags.
    - **Parser configured** (`--reasoning-parser qwen3`). The trace arrives out of band
      and `content` holds only the visible answer. The out-of-band field is named
      `reasoning_content` on vLLM 0.8.x and `reasoning` on 0.26 — the caller passes
      whichever it found.
    - **Thinking disabled.** No trace at all; `content` is the bare answer.

    Args:
        content: The `message.content` field, possibly `None`/empty.
        reasoning: The out-of-band trace, from whichever field carried it.

    Returns:
        `(think, answer)`, both stripped.
    """
    raw = content or ""
    think, answer = split_think(raw)
    if reasoning and not think:
        # An out-of-band trace means `content` was never a container for it, so the
        # whole of `content` is the answer — including the case where content is empty
        # because generation was cut off mid-trace, which must stay an empty answer so
        # it scores as unparseable rather than silently borrowing the trace text.
        return str(reasoning).strip(), raw.strip()
    return think, answer


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> dict[str, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than normal-approximation: at n≈500 and accuracy near 0.8 the two
    barely differ, but Wilson stays inside [0, 1] and keeps behaving at the small n of
    a single-subject breakdown, where the normal interval produces bounds above 1.

    Args:
        k: Successes.
        n: Trials.
        alpha: Two-sided level.

    Returns:
        `mean`, `ci_lower`, `ci_upper`, `n`.
    """
    if n <= 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}
    z = _z_for(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {
        "mean": p,
        "ci_lower": max(0.0, centre - half),
        "ci_upper": min(1.0, centre + half),
        "n": n,
    }


def _z_for(alpha: float) -> float:
    """Two-sided normal critical value, via the inverse error function."""
    # math.erfinv does not exist; use the standard relation through erf's inverse by
    # bisection on erf, which is exact enough for interval endpoints and avoids a scipy
    # dependency the rest of the repo does not carry.
    target = 1.0 - alpha
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on paired per-question outcomes.

    Both arms answered identical questions, so the informative evidence is the
    *discordant* pairs: questions one arm got right and the other got wrong. Questions
    both arms answered the same way carry no information about the difference, and a
    two-sample test that pools them throws away the pairing and widens the interval for
    nothing.

    The exact binomial form is used rather than the chi-square approximation because the
    discordant count can be small — at n=570 with two near-identical checkpoints, a
    dozen discordant pairs is a realistic outcome and chi-square is unreliable there.

    Args:
        a_correct: Reference arm's per-question correctness.
        b_correct: Test arm's per-question correctness, question-aligned with `a_correct`.

    Returns:
        `n_discordant`, `b01` (reference right, test wrong), `b10` (test right,
        reference wrong), and the two-sided exact `p_value`.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(
            f"paired test needs aligned outcomes, got {len(a_correct)} vs {len(b_correct)}"
        )
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        # Two-sided exact binomial at p=0.5: twice the smaller tail, capped at 1.
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
        p = min(1.0, 2.0 * tail)
    return {"n_discordant": n, "b01": b01, "b10": b10, "p_value": p}


def paired_bootstrap_diff(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    rounds: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI for the accuracy difference `b - a`, resampling questions.

    Resamples question *indices* and carries both arms' outcomes for the drawn question
    together. Between-question difficulty variance is common to both arms and cancels in
    the difference, so this interval is materially tighter than differencing two
    independent per-arm intervals — the same paired argument `capability_stats` makes
    for the preference eval.

    Args:
        a_correct: Reference arm's per-question correctness.
        b_correct: Test arm's per-question correctness, question-aligned.
        rounds: Bootstrap resamples.
        alpha: Two-sided level.
        seed: Fixed so a reported interval is reproducible.

    Returns:
        `diff` (b - a, in proportion units), `ci_lower`, `ci_upper`, `std_error`, `n`.
    """
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs aligned outcomes, got {a.shape} vs {b.shape}")
    n = len(a)
    if n == 0:
        raise ValueError("paired_bootstrap_diff called with no questions")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(rounds, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {
        "diff": float(b.mean() - a.mean()),
        "ci_lower": float(np.quantile(diffs, alpha / 2)),
        "ci_upper": float(np.quantile(diffs, 1 - alpha / 2)),
        "std_error": float(diffs.std(ddof=1)),
        "n": n,
    }


def score_records(records: Sequence[dict]) -> dict[str, Any]:
    """Summarise one arm's graded records into the numbers the report prints.

    `accuracy` counts an unparseable response as wrong, which is the honest headline:
    a model that cannot state an answer has not answered. `accuracy_parsed_only` is
    reported beside it because the two diverging is the signature of a *format* failure
    rather than a knowledge failure — the case where the fix is `max_tokens` or the chat
    template, not the training mixture.

    Args:
        records: Graded records carrying `correct`, `parsed`, `parse_tier`,
            `finish_reason`, `subject`, `category`, `think_words`.

    Returns:
        Overall block plus `by_category` and `by_subject` accuracy breakdowns.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    n_correct = sum(1 for r in records if r["correct"])
    parsed = [r for r in records if r["parsed"] is not None]
    truncated = sum(1 for r in records if r.get("finish_reason") == "length")

    def _group(key: str) -> dict[str, dict[str, float]]:
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(r[key]), []).append(r)
        return {
            name: wilson_ci(sum(1 for r in rows if r["correct"]), len(rows))
            for name, rows in sorted(groups.items())
        }

    tiers: dict[str, int] = {}
    for r in records:
        tiers[r.get("parse_tier", "none")] = tiers.get(r.get("parse_tier", "none"), 0) + 1

    return {
        "n": n,
        "n_correct": n_correct,
        **wilson_ci(n_correct, n),
        "accuracy_parsed_only": (
            sum(1 for r in parsed if r["correct"]) / len(parsed) if parsed else float("nan")
        ),
        "parse_rate": len(parsed) / n,
        "truncation_rate": truncated / n,
        "mean_think_words": sum(float(r.get("think_words", 0)) for r in records) / n,
        "empty_think_rate": sum(1 for r in records if not r.get("think_words")) / n,
        "parse_tiers": tiers,
        "by_category": _group("category"),
        "by_subject": _group("subject"),
    }
