# ABOUTME: Corpus validity checks for a model-eval-model run: properties of the data, not the
# ABOUTME: pipeline. Gated by config thresholds; report always written before gating.

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src.endpoints.openrouter import OpenRouterClient

from .model_eval_model_cells import CELLS, eval_response_text
from .constitution import full_text
from .stage_runtime import JUDGE_NO_REASONING, Usage, call_tagged, run_items
from .check_corpus import (CORPUS_CHECKS, Corpus, check_label_leakage,
                           check_ngram_diversity, words as _words)
from .hf_cache import read_jsonl

# Phrases that read as a settled judgement of the evaluated reply. Appearing in the
# opening stretch of a trace suggests the verdict came first and the considerations
# were back-filled -- the generator failure the corpus must not contain.
_VERDICT_MARKERS = (
    "was sound", "is sound", "was the right call", "is the right call", "was wrong",
    "was right to", "fails the", "falls short", "handled this well", "handled it well",
    "got it right", "got this right", "got it wrong", "missed the", "my verdict",
    "verdict:",
)

# Below these document counts the corresponding gate is reported but not enforced --
# the statistics are binomial noise at smoke scale. The per-group and per-class ones
# now live in the registry (`ngram_diversity.min_group_docs`,
# `label_leakage.min_per_class`), so the adapters below read them from there.
_MIN_DOCS_FOR_VERDICT = 20
_MIN_DOCS_FOR_FLAWID = 10

# The verdict each arm should mostly (but never degenerately) reach: good responses are
# mostly sound/held, flawed ones mostly caught. 100% either way means the verdict carries
# no signal (or, in the self arms, trains reflexive capitulation/defensiveness). Keyed by
# whatever the config groups on -- `cell` for the archived cell-based configs, a plain
# label field for the ones built out of generic operators. A config may override or extend
# it with `checks.expected_majority`.
_EXPECTED_MAJORITY = {
    "m4_other_good": "sound",
    "m3_other_flawed": "issue_found",
    "m2_self_good": "held",
    "m1_self_flawed": "revised",
    "good": "held",
    "flawed": "revised",
}

_MIN_DOCS_FOR_DIVERSITY = 20

# Which record fields the checks read. Defaults are the cell-based vocabulary the
# archived configs use; a config overrides any of them under `checks.fields`, which is
# what lets one check suite serve both a celled corpus and one whose arms are just a
# label an `assign` stage stamped on.
FIELD_DEFAULTS = {
    "group": "cell",            # the arm a document belongs to
    "id": "record_id",          # unique per document
    "quality": "response_kind", # good | flawed, for the surface-shortcut classifier
    # The reply under evaluation. Empty means "ask the cell", which is the only way to
    # get it for a celled corpus (three provenances, resolved by `eval_response_text`).
    # A de-celled config names the field directly.
    "evaluated": "",
    "verdict": "assessment",
}


def _fields(cfg: dict) -> dict[str, str]:
    """Resolve the record-field names the checks should read for this config."""
    named = dict((cfg.get("checks") or {}).get("fields") or {})
    unknown = sorted(set(named) - set(FIELD_DEFAULTS))
    assert not unknown, (
        f"checks.fields: unknown key(s) {unknown}. Known: {sorted(FIELD_DEFAULTS)}")
    return {**FIELD_DEFAULTS, **named}


def _doc_text(r: dict) -> str:
    """The generated prose of one document (reasoning plus reply where present)."""
    return r["reasoning"] + "\n" + r.get("response", "")


def check_gate_yield(plan: list[dict], entered: list[dict], group: str = "cell") -> dict:
    """Report how many planned documents survived to generation, per cell.

    Report-only: a `filter` stage already fails the run outright on a drop rate that
    means the recipe is broken. What this adds is the number a config needs in order to
    stop guessing -- the gates' real yield, which is what its `expected_keep` priors and
    its cell counts should be re-sized from after a smoke run.
    """
    def count(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r[group]] = out.get(r[group], 0) + 1
        return out

    planned, kept = count(plan), count(entered)
    if planned == kept:
        return {"pass": True, "gated_any": False,
                "note": "no stage between planning and generation drops records"}
    return {"pass": True, "gated_any": True, "planned": len(plan),
            "entered_generation": len(entered),
            "by_arm": {c: {"planned": n, "kept": kept.get(c, 0),
                           "yield": round(kept.get(c, 0) / n, 3)}
                       for c, n in sorted(planned.items())}}


def check_coverage(plan: list[dict], generated: list[dict], group: str = "cell") -> dict:
    """Compare generated counts against the records that ENTERED generation, over
    cell x trait and, for flawed cells, cell x flaw-type x severity.

    A bucket that entered generation and came out empty means every one of its documents
    failed -- systematic, so it gates. The baseline is deliberately the last snapshot
    before generation rather than the plan: in a found-lapse recipe a gate legitimately
    empties a bucket, and calling that a generation failure would gate on the recipe
    working. `check_gate_yield` reports that attrition separately.
    """
    def buckets(r: dict) -> list[str]:
        keys = [f"{r[group]}/{r['trait_id']}"]
        if r.get("flaw"):
            keys.append(f"{r[group]}/{r['flaw']['type']}/{r['flaw']['severity']}")
        return keys

    def count(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            for k in buckets(r):
                out[k] = out.get(k, 0) + 1
        return out

    planned, got = count(plan), count(generated)
    empty = sorted(k for k in planned if got.get(k, 0) == 0)
    styles: dict[str, dict[str, int]] = {}
    for r in generated:
        c = styles.setdefault(r[group], {})
        c[r.get("explicitness", "")] = c.get(r.get("explicitness", ""), 0) + 1
    flaw_grid = {k: v for k, v in sorted(got.items()) if k.count("/") == 2}
    return {"pass": not empty, "planned": len(plan), "generated": len(generated),
            "empty_buckets": empty, "explicitness_by_arm": styles,
            "flaw_grid": flaw_grid}


def _corpus_of(generated: list[dict], text, group: str, seed: int, **fields) -> Corpus:
    """Build a Corpus view over model-eval-model records for a registry check.

    The cell/flaw/verdict vocabulary stops here: the registry check below sees only
    roles, which is what lets the same implementation serve any document type.
    """
    return Corpus(records=generated, params={}, seed=seed, name="",
                  fields={"id": "record_id", "text": text, "group": group, **fields})


def check_template_collapse(generated: list[dict], max_8gram_share: float,
                            seed: int = 0, max_4gram_jaccard: float | None = None,
                            group: str = "cell") -> dict:
    """Detect critique-shape collapse: repeated long n-grams and high pairwise overlap.

    Generators converge on one critique shape quickly; a single 8-gram appearing in a
    large share of a cell's documents is the fingerprint.

    A thin adapter over the `ngram_diversity` corpus property -- same numbers, same
    seed, same per-cell keys -- so the in-pipeline stage and this post-hoc verb cannot
    drift on what "8-gram share" means.

    Args:
        generated: The generated-document snapshot.
        max_8gram_share: The gated threshold, per cell.
        seed: Jaccard sampling seed.
        max_4gram_jaccard: Optional second gate. The mean pairwise 4-gram Jaccard has
            always been computed here and never enforced; a config opts in by setting
            `checks.gates.template_4gram_jaccard_max`. Left None it is reported only,
            so an existing config's verdict is unchanged.

    Returns:
        `{pass, max_8gram_share, cells: {cell: {...}}}` -- the historical shape.
    """
    c = _corpus_of(generated, ["reasoning", "response"], group, seed)
    c.params = {**CORPUS_CHECKS["ngram_diversity"].defaults,
                "top_8gram_share_max": max_8gram_share,
                "mean_jaccard_max": (float("inf") if max_4gram_jaccard is None
                                     else float(max_4gram_jaccard)),
                "distinct_2_min": 0.0}
    result = check_ngram_diversity(c)

    cells_out: dict[str, dict] = {}
    ok = True
    for cell, m in result.metrics["by_group"].items():
        # `gated` is the registry's own small-group guard, not a second copy of it.
        cell_ok = (not m["gated"]) or (
            m["top_8gram_share"] <= max_8gram_share
            and (max_4gram_jaccard is None
                 or m["mean_pairwise_4gram_jaccard"] <= float(max_4gram_jaccard)))
        ok = ok and cell_ok
        cells_out[cell] = {"docs": m["docs"], "top_8gram_share": m["top_8gram_share"],
                           "top_8gram": m["top_8gram"],
                           "mean_pairwise_4gram_jaccard": m["mean_pairwise_4gram_jaccard"],
                           "gated": m["gated"], "pass": cell_ok}
    return {"pass": ok, "max_8gram_share": max_8gram_share,
            "max_4gram_jaccard": max_4gram_jaccard, "cells": cells_out}


def _last(messages: list[dict], role: str) -> dict:
    """The last message with this role (the trained turn, and the turn it answers)."""
    return next(m for m in reversed(messages) if m["role"] == role)


def check_structural_diversity(sft: list[dict], unique_min: float, cv_min: float,
                               opening_max: float, group: str = "cell") -> dict:
    """Corpus-level check for repeated structure in what actually trains.

    `check_template_collapse` looks for shared long n-grams anywhere in a document.
    This looks for the artifacts a config's SHAPE produces rather than its generator's
    voice, per cell, on the assembled records:

    * `user_turn_unique_share` -- distinct final user turns. A fixed reflection prompt
      or a handful of transcript wrappers reused across thousands of documents makes
      this collapse toward `1/len(variants)`, and the model can key the whole behaviour
      on that phrasing.
    * `length_cv` -- coefficient of variation of the trained turn's length. Uniform
      lengths mean the generator settled into one output size regardless of what the
      situation needed.
    * `top_opening_5gram_share` -- share of documents whose trained turn opens with the
      same five words: the scaffold tell.

    Reported for every cell, gated only where a cell has enough documents for the
    statistic to mean anything.
    """
    import statistics

    by_cell: dict[str, list[dict]] = {}
    for rec in sft:
        by_cell.setdefault(rec["metadata"][group], []).append(rec)

    cells_out: dict[str, dict] = {}
    ok = True
    for cell, rows in sorted(by_cell.items()):
        users = [_last(r["messages"], "user")["content"].strip() for r in rows]
        trained = [_last(r["messages"], "assistant") for r in rows]
        lengths = [len(_words((m.get("reasoning_content") or "") + " "
                              + (m.get("content") or ""))) for m in trained]
        openings = [" ".join(_words(m.get("content") or "")[:5]) for m in trained]

        n = len(rows)
        counts: dict[str, int] = {}
        for o in openings:
            counts[o] = counts.get(o, 0) + 1
        top_open = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
        user_counts: dict[str, int] = {}
        for u in users:
            user_counts[u] = user_counts.get(u, 0) + 1
        mean_len = statistics.fmean(lengths) if lengths else 0.0
        cv = (statistics.pstdev(lengths) / mean_len) if mean_len else 0.0
        unique_share = len(set(users)) / n

        gated = n >= _MIN_DOCS_FOR_DIVERSITY
        cell_ok = (not gated) or (unique_share >= unique_min and cv >= cv_min
                                  and top_open[1] / n <= opening_max)
        ok = ok and cell_ok
        cells_out[cell] = {
            "docs": n,
            "user_turn_unique_share": round(unique_share, 3),
            "top_user_turn_share": round(
                max(user_counts.values(), default=0) / n, 3),
            "trained_turn_mean_words": round(mean_len, 1),
            "length_cv": round(cv, 3),
            "top_opening_5gram": top_open[0],
            "top_opening_5gram_share": round(top_open[1] / n, 3),
            "gated": gated, "pass": cell_ok,
        }
    return {"pass": ok, "thresholds": {"user_turn_unique_share_min": unique_min,
                                       "length_cv_min": cv_min,
                                       "opening_5gram_share_max": opening_max},
            "cells": cells_out}


def check_verdict_distribution(generated: list[dict], majority_min: float,
                               majority_max: float, group: str = "cell",
                               verdict: str = "assessment",
                               expected: dict | None = None) -> dict:
    """Gate each cell's verdicts on a non-degenerate split around its expected majority.

    m4/m2 evaluate good responses (mostly sound/held), m3/m1 flawed ones (mostly
    caught/revised) -- but 100% either way is a failure: the verdict then carries no
    signal, and in the self cells it trains reflexive capitulation (every pushback ->
    revision) or defensiveness (every pushback -> hold).
    """
    majority = {**_EXPECTED_MAJORITY, **(expected or {})}
    judged = [r for r in generated if verdict in r]
    if not judged:
        return {"pass": True, "judged": 0,
                "note": "no verdict-bearing arms in this run; nothing to gate"}
    cells_out: dict[str, dict] = {}
    ok = True
    for cell in sorted({r[group] for r in judged}):
        rows = [r for r in judged if r[group] == cell]
        dist: dict[str, int] = {}
        for r in rows:
            dist[r[verdict]] = dist.get(r[verdict], 0) + 1
        want = majority.get(cell)
        share = dist.get(want, 0) / len(rows) if want else None
        gated = want is not None and len(rows) >= _MIN_DOCS_FOR_VERDICT
        cell_ok = (not gated) or majority_min <= share <= majority_max
        ok = ok and cell_ok
        cells_out[cell] = {"n": len(rows), "distribution": dist, "expected": want,
                           "expected_share": None if share is None else round(share, 3),
                           "gated": gated, "pass": cell_ok}
    return {"pass": ok, "judged": len(judged),
            "band": [majority_min, majority_max], "cells": cells_out}


def check_post_hoc_heuristic(generated: list[dict], verdict: str = "assessment") -> dict:
    """Flag traces whose earliest verdict phrase sits in the opening 20%.

    A cheap screen, report-only; the judged sample (`post_hoc_judge`) is the
    authoritative read. Applies to critique cells only -- the control has no verdict.
    """
    judged = [r for r in generated if verdict in r]
    early = flagged = 0
    for r in judged:
        text = r["reasoning"].lower()
        positions = [m for m in (text.find(k) for k in _VERDICT_MARKERS) if m != -1]
        if positions:
            flagged += 1
            if min(positions) < 0.2 * len(text):
                early += 1
    return {"pass": True, "judged": len(judged), "with_verdict_phrase": flagged,
            "early_verdict": early,
            "early_share": round(early / len(judged), 3) if judged else 0.0}


def check_blindness(generated: list[dict], sft: list[dict], constitution: str,
                    P: dict, group: str = "cell", ident: str = "record_id") -> dict:
    """Prove the flaw label never reaches a generation prompt or a training record.

    For every flawed document, the generation messages rebuilt with the flaw metadata
    stripped must be byte-identical to the real ones (the evaluated response text
    itself stays -- that is the input, not the label), and the perturbation stage's
    `change_summary` must not appear verbatim in any training message.

    A config that defines `prompts.known_flaw_note` deliberately unblinds the
    GENERATOR (the reflect prompt carries the change summary as scaffolding); the
    prompt-identity half is then skipped and reported as such -- the training-record
    half still gates, since the summary must never train regardless.
    """
    # The prompt-identity half rebuilds a document's generation messages, which only a
    # cell knows how to do. A config built from generic operators has no such builder --
    # its prompts live in the config -- so that half is skipped and reported as skipped.
    # The half that actually matters is unconditional: the summary must never train.
    rebuildable = all(r.get(group) in CELLS for r in generated)
    generator_blind = not P.get("known_flaw_note")
    flawed = [r for r in generated if r.get("flaw") or r.get("change_summary")]
    prompt_leaks = []
    if generator_blind and rebuildable:
        for r in flawed:
            build = CELLS[r[group]].build_messages
            stripped = {**r, "flaw": None, "change_summary": ""}
            if build(r, constitution, P) != build(stripped, constitution, P):
                prompt_leaks.append(r[ident])

    summaries = {r[ident]: r["change_summary"]
                 for r in flawed if r.get("change_summary")}
    sft_leaks = []
    for rec in sft:
        s = summaries.get(rec["metadata"].get(ident))
        if s and any(s in m["content"] for m in rec["messages"]):
            sft_leaks.append(rec["metadata"][ident])
    return {"pass": not prompt_leaks and not sft_leaks, "flawed_docs": len(flawed),
            "generator_blind": generator_blind,
            "prompt_identity_checked": generator_blind and rebuildable,
            "prompt_leaks": prompt_leaks, "sft_leaks": sft_leaks}


def check_gold_validation(source: list[dict], client: OpenRouterClient,
                          usage: Usage, model: str, judges: dict, n: int, seed: int,
                          workers: int, max_below_3: float) -> dict:
    """Judge a sample of good responses against their target trait, 1-5.

    A wrong "good" response poisons every document built on it (a confabulated
    endorsement in m4, a broken minimal pair in m1/m3), so this gates before scaling.

    `source` is whichever population supplies the corpus's *known-good* replies: a
    completed run's gold responses where there is one, and otherwise the first turns the
    pipeline generated and its own rater passed as sound. The second case is the one
    that matters most -- it is an INDEPENDENT read on the rater's "no genuine shortfall"
    verdict, which in a found-lapse recipe is the only thing separating the good cell
    from the flawed one.
    """
    if not source:
        return {"pass": True, "sampled": 0,
                "note": "no known-good replies to judge in this run"}
    rng = random.Random(seed)
    picked = rng.sample(source, min(n, len(source)))

    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": judges["gold_system"]},
             {"role": "user", "content": judges["gold_user"].format(
                 trait_name=r["trait_name"], trait_text=r["trait_text"],
                 user=r["user"], response=r["response"])}],
            0.3, 800, "check:gold", ("score", "why"), extra=JUDGE_NO_REASONING)
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")
        return {"scenario_id": r["scenario_id"], "score": score, "why": parsed["why"]}

    scored = run_items(picked, one, workers, "check:gold")
    scores = [s["score"] for s in scored]
    below = sum(1 for s in scores if s < 3)
    share_below = below / len(scores) if scores else 1.0
    dist = {str(v): scores.count(v) for v in range(1, 6)}
    worst = sorted(scored, key=lambda s: s["score"])[:3]
    return {"pass": share_below <= max_below_3, "sampled": len(scores),
            "share_below_3": round(share_below, 3), "max_below_3": max_below_3,
            "distribution": dist, "worst": worst}


def check_flaw_identification(generated: list[dict], client: OpenRouterClient,
                              usage: Usage, model: str, judges: dict, n: int,
                              seed: int, workers: int, min_hit_clear: float,
                              ident: str = "record_id") -> dict:
    """Judge whether each blind critique landed on the shortfall actually present.

    A low rate means the lapses are too subtle or the critiques are confabulated -- either
    way every flawed-response document is suspect.

    Which bucket gates depends on where the lapse came from. A PLANTED flaw carries a
    severity, and only the `clear` band gates (grey flaws are legitimately arguable). A
    FOUND flaw -- an autorater naming a genuine shortfall in a reply nobody perturbed --
    has no severity to band by, so every judged document counts toward one overall rate.
    Reporting a severity these documents never had would be fiction; skipping the gate
    entirely would leave the recipe's central claim unmeasured.
    """
    flawed = [r for r in generated if r.get("change_summary") and "reasoning" in r]
    if not flawed:
        return {"pass": True, "sampled": 0, "note": "no flawed cells in this run"}
    rng = random.Random(seed)
    picked = rng.sample(flawed, min(n, len(flawed)))

    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": judges["flawid_system"]},
             {"role": "user", "content": judges["flawid_user"].format(
                 change_summary=r["change_summary"], critique=_doc_text(r))}],
            0.3, 500, "check:flawid", ("hit", "why"), extra=JUDGE_NO_REASONING)
        v = parsed["hit"].strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"unrecognised hit answer: {v!r}")
        flaw = r.get("flaw") or {}
        return {"record_id": r[ident], "hit": v == "yes",
                "severity": flaw.get("severity") or "unrated",
                "type": flaw.get("type") or "unrated",
                "why": parsed["why"]}

    results = run_items(picked, one, workers, "check:flawid")
    by_sev: dict[str, dict] = {}
    by_type: dict[str, dict] = {}
    for r in results:
        for key, bucket in ((r["severity"], by_sev), (r["type"], by_type)):
            b = bucket.setdefault(key, {"n": 0, "hits": 0})
            b["n"] += 1
            b["hits"] += r["hit"]
    for bucket in (by_sev, by_type):
        for b in bucket.values():
            b["rate"] = round(b["hits"] / b["n"], 3)
    overall = {"n": len(results), "hits": sum(r["hit"] for r in results)}
    overall["rate"] = round(overall["hits"] / overall["n"], 3) if results else 0.0
    # Planted flaws band by severity and only `clear` gates; found flaws have no
    # severity, so the whole judged sample is the band.
    band = "clear" if "clear" in by_sev else "overall"
    gate_on = by_sev["clear"] if band == "clear" else overall
    gated = gate_on["n"] >= _MIN_DOCS_FOR_FLAWID
    return {"pass": (not gated) or gate_on["rate"] >= min_hit_clear,
            "sampled": len(results), "gated": gated, "gated_on": band,
            "gated_rate": gate_on["rate"], "min_hit_clear": min_hit_clear,
            "by_severity": by_sev, "by_type": by_type,
            "missed": [r for r in results if not r["hit"]][:5]}


def check_surface_shortcut(generated: list[dict], max_auc: float, seed: int,
                           quality: str = "response_kind",
                           evaluated: str = "", group: str = "cell") -> dict:
    """Can a cheap classifier predict good vs flawed from the evaluated response alone?

    If yes, the perturbations have a surface tell (length, phrasing, a stray hedge) and
    the model will learn the tell instead of the reasoning. A label-shuffle baseline
    anchors what "chance" looks like at this sample size.

    A thin adapter over the `label_leakage` corpus property; `flawed` is the positive
    class, so the reported AUC keeps its historical direction.
    """
    rows = [r for r in generated if r.get(quality) in ("good", "flawed")]
    # A de-celled config names the evaluated field outright; a celled one resolves it per
    # arm, since which reply a cell evaluates depends on the cell.
    text_spec = evaluated or {"by": quality,
                              "cases": {"flawed": "flawed_response",
                                        "good": "gold_response"}}
    c = _corpus_of(rows, text_spec, group, seed, label=quality)
    c.params = {"surface_auc_max": max_auc, "positive": "flawed",
                "min_per_class": CORPUS_CHECKS["label_leakage"].defaults[
                    "min_per_class"]}
    m = check_label_leakage(c).metrics

    n_pos, n_neg = int(m.get("flawed", 0)), int(m.get("good", 0))
    if not m.get("gated"):
        return {"pass": True, "flawed": n_pos, "good": n_neg, "gated": False,
                "note": m.get("note", "not enough documents per class")}
    return {"pass": m["auc"] is not None and m["auc"] <= max_auc,
            "gated": True, "flawed": n_pos, "good": n_neg,
            "auc": m["auc"], "max_auc": max_auc,
            "auc_label_shuffled": m["auc_label_shuffled"],
            # Reported, never gated: a leak pointing the other way is real, but this
            # estimator's separability sits near 0.65 on a null corpus of this size.
            "separability": m["separability"],
            "ambiguous_texts_dropped": m["ambiguous_texts_dropped"],
            "mean_word_delta_flawed_minus_good": m["mean_word_delta"]}


def check_post_hoc_judge(generated: list[dict], client: OpenRouterClient,
                         usage: Usage, model: str, judges: dict, n: int, seed: int,
                         workers: int, verdict: str = "assessment",
                         ident: str = "record_id") -> dict:
    """LLM-judged post-hoc-reasoning rate on a small sample of critique traces."""
    judged = [r for r in generated if verdict in r]
    if not judged:
        return {"pass": True, "sampled": 0, "note": "no verdict-bearing arms in this run"}
    rng = random.Random(seed)
    picked = rng.sample(judged, min(n, len(judged)))

    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": judges["posthoc_system"]},
             {"role": "user", "content": judges["posthoc_user"].format(
                 reasoning=r["reasoning"])}],
            0.3, 500, "check:posthoc", ("posthoc", "why"), extra=JUDGE_NO_REASONING)
        v = parsed["posthoc"].strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"unrecognised posthoc answer: {v!r}")
        return {"record_id": r[ident], "posthoc": v == "yes", "why": parsed["why"]}

    results = run_items(picked, one, workers, "check:posthoc")
    share = sum(1 for r in results if r["posthoc"]) / len(results) if results else 0.0
    return {"pass": True, "sampled": len(results), "post_hoc_share": round(share, 3),
            "flagged": [r for r in results if r["posthoc"]][:5]}


def _gold_rows(source: list[dict], generated: list[dict],
               quality: str = "response_kind",
               evaluated: str = "first_turn") -> list[dict]:
    """The known-good replies gold validation should judge.

    A source run supplies them directly. Without one, the corpus's claim to have any
    known-good reply rests entirely on its own rater having passed a generated first turn
    as sound -- so those are what an independent judge must read, shaped into the same
    fields the gold judge template expects.
    """
    if source:
        return source
    return [{"scenario_id": r["scenario_id"], "trait_name": r["trait_name"],
             "trait_text": r["trait_text"], "user": r["user"],
             "response": str(r[evaluated])}
            for r in generated
            if r.get(quality) == "good" and r.get(evaluated)]


def run_checks(run_dir: str | Path, cfg: dict,
               sample: int | None = None) -> tuple[dict, bool]:
    """Run every applicable validity check over a model-eval-model run and write the report.

    The report is always written in full before any gating, so a failed run is still
    inspectable. Checks marked report-only always pass; gated checks read their
    thresholds from `cfg["checks"]["gates"]`.

    Args:
        run_dir: The model-eval-model run directory (stage snapshots + manifest).
        cfg: The model-eval-model run config.
        sample: Override for the LLM-judged sample size.

    Returns:
        (the report dict, whether every gated check passed).
    """
    from .pipeline import build_stages, snapshot_positions

    run_dir = Path(run_dir)
    # Snapshot positions and names come from the config's `stages:` list, so the
    # checks stay correct when a stage is added (e.g. the `final` rewrite pass) --
    # and observer stages take no position, so adding a corpus check anywhere in the
    # list leaves every one of these paths where it was.
    names = [s["name"] for s in cfg["stages"]]
    kinds = {s["name"]: s["kind"] for s in cfg["stages"]}
    positions = snapshot_positions(build_stages(cfg))

    def _snap_path(name: str) -> Path:
        return run_dir / f"stage_{positions[name]}_{name}.jsonl"

    def snap(name: str) -> list[dict]:
        p = _snap_path(name)
        assert p.exists(), (
            f"{p.name} not found in {run_dir} -- was this run produced by an older "
            f"stage layout? Check it with the config version recorded in its manifest.")
        return read_jsonl(p)

    ccfg = cfg.get("checks", {})
    # Which snapshots to read. A cell-based config is found by operator kind, since its
    # kinds ARE its roles. A config assembled from generic operators has no such
    # signature -- `llm_tagged` appears six times and means something different each time
    # -- so it names the three roles explicitly under `checks.stages`.
    named = dict(ccfg.get("stages") or {})
    unknown = sorted(set(named) - {"plan", "drafted", "generated", "sft"})
    assert not unknown, (
        f"checks.stages: unknown role(s) {unknown}. Expected "
        f"plan/drafted/generated/sft.")
    bad = sorted(v for v in named.values() if v not in names)
    assert not bad, (
        f"checks.stages names stage(s) {bad} that are not in this config's pipeline "
        f"({names}).")

    def by_kind(kind: str) -> str:
        return next(n for n in names if kinds[n] == kind)

    def by_kind_opt(kind: str) -> str | None:
        return next((n for n in names if kinds[n] == kind), None)

    def role(name: str, kind: str) -> str:
        return named[name] if name in named else by_kind(kind)

    # A config that generates its own scenarios has no source run and therefore no gold
    # replies; `_gold_rows` below picks the right population for gold validation either
    # way, so every other check is unaffected.
    src_stage = by_kind_opt("load_source_run")
    source = snap(src_stage) if src_stage else []
    plan = snap(role("plan", "plan_cells"))
    # Judge the documents that actually train: the LAST snapshot carrying them -- the
    # rewrite stage's when the pipeline has one and it ran (a skipped stage writes no
    # snapshot), else the generation stage's.
    if "generated" in named:
        doc_stages = [named["generated"]]
    else:
        doc_stages = [n for n in names if kinds[n] in ("generate_cells", "revise_cells")
                      and _snap_path(n).exists()]
    assert doc_stages, f"no generated-document snapshot found in {run_dir}"
    generated = snap(doc_stages[-1])
    # What generation was actually handed: the last snapshot before the FIRST stage that
    # produced documents. Equal to the plan unless a `filter` stage sits in between.
    # `drafted` names that first stage where it differs from the last one -- a pipeline
    # that drafts a turn and then rewrites it has two document stages, and measuring
    # attrition from the rewrite would report a yield of 100% every time.
    first_doc = named.get("drafted") or named.get("generated") \
        or by_kind("generate_cells")
    gen_ix = names.index(first_doc)
    entered = next((snap(nm) for nm in reversed(names[:gen_ix])
                    if _snap_path(nm).exists()), plan)
    sft = snap(role("sft", "assemble_cells"))
    constitution = full_text(cfg["constitution"])

    F = _fields(cfg)
    group, ident = F["group"], F["id"]
    quality, evaluated, verdict = F["quality"], F["evaluated"], F["verdict"]
    gates = ccfg.get("gates", {})
    seed = int(cfg.get("seed", 0))
    workers = int(cfg.get("workers", 8))
    n = int(sample or ccfg.get("sample", 100))

    report: dict = {"run_dir": str(run_dir), "generated": len(generated)}
    report["gate_yield"] = check_gate_yield(plan, entered, group)
    report["coverage"] = check_coverage(entered, generated, group)
    jac = gates.get("template_4gram_jaccard_max")
    report["template_collapse"] = check_template_collapse(
        generated, float(gates.get("template_8gram_share_max", 0.2)), seed,
        max_4gram_jaccard=None if jac is None else float(jac), group=group)
    report["verdict_distribution"] = check_verdict_distribution(
        generated, float(gates.get("verdict_majority_min", 0.6)),
        float(gates.get("verdict_majority_max", 0.98)), group, verdict,
        ccfg.get("expected_majority"))
    report["structural_diversity"] = check_structural_diversity(
        sft, float(gates.get("user_turn_unique_share_min", 0.90)),
        float(gates.get("length_cv_min", 0.15)),
        float(gates.get("opening_5gram_share_max", 0.25)), group)
    report["post_hoc_heuristic"] = check_post_hoc_heuristic(generated, verdict)
    report["blindness"] = check_blindness(generated, sft, constitution,
                                          cfg["prompts"], group, ident)
    report["surface_shortcut"] = check_surface_shortcut(
        generated, float(gates.get("surface_auc_max", 0.65)), seed, quality,
        evaluated if _fields(cfg) is not FIELD_DEFAULTS else "")

    judge_model = ccfg.get("judge_model")
    if judge_model:
        judges = ccfg["judges"]
        client = OpenRouterClient()
        usage = Usage()
        report["gold_validation"] = check_gold_validation(
            _gold_rows(source, generated, quality, evaluated), client, usage,
            judge_model, judges, n, seed, workers,
            float(gates.get("gold_below_3_max", 0.10)))
        report["flaw_identification"] = check_flaw_identification(
            generated, client, usage, judge_model, judges, n, seed, workers,
            float(gates.get("flaw_id_clear_min", 0.70)), ident)
        report["post_hoc_judge"] = check_post_hoc_judge(
            generated, client, usage, judge_model, judges, min(n, 30), seed, workers,
            verdict, ident)
        report["judge_spend_usd"] = round(usage.usd, 4)
    else:
        print("!!! checks.judge_model not set -- skipping the LLM-judged checks")

    ok = all(sec.get("pass", True) for sec in report.values() if isinstance(sec, dict))
    report["pass"] = ok
    report["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "checks_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    for name, sec in report.items():
        if isinstance(sec, dict) and "pass" in sec:
            keys = {k: v for k, v in sec.items()
                    if k not in ("pass",) and not isinstance(v, (dict, list))}
            print(f"{'PASS' if sec['pass'] else 'FAIL'}  {name}: {keys}")
    print(f">>> checks {'PASSED' if ok else 'FAILED'} -- "
          f"report at {run_dir / 'checks_report.json'}")
    return report, ok
