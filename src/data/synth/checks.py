# ABOUTME: Corpus validity checks for a model-eval-model run: properties of the data, not the
# ABOUTME: pipeline. Gated by config thresholds; report always written before gating.

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src.endpoints.openrouter import OpenRouterClient

from .cells import CELLS
from .constitution import full_text
from .core import Usage, call_tagged, run_items
from .corpus import Corpus, check_label_leakage, check_ngram_diversity
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
# the statistics are binomial noise at smoke scale.
# Judges return a tag and one sentence inside tight max_tokens; Sonnet 5's hidden
# extended thinking otherwise eats that budget and returns EMPTY content with
# finish_reason=length (observed 2026-08-05: 500-token cap, 95+ reasoning tokens,
# content=None). Same OpenRouter control the self_reflection stages use.
_JUDGE_NO_REASONING = {"reasoning": {"enabled": False}}

_MIN_DOCS_FOR_COLLAPSE = 5
_MIN_DOCS_FOR_VERDICT = 20
_MIN_DOCS_PER_CLASS_FOR_SHORTCUT = 20
_MIN_DOCS_FOR_FLAWID = 10

# The verdict each cell should mostly (but never degenerately) reach: good responses
# are mostly sound/held, flawed ones mostly caught. 100% either way means the verdict
# carries no signal (or, in the self cells, trains reflexive capitulation/defensiveness).
_EXPECTED_MAJORITY = {
    "m4_other_good": "sound",
    "m3_other_flawed": "issue_found",
    "m2_self_good": "held",
    "m1_self_flawed": "revised",
}


def _doc_text(r: dict) -> str:
    """The generated prose of one document (reasoning plus reply where present)."""
    return r["reasoning"] + "\n" + r.get("response", "")


def check_coverage(plan: list[dict], generated: list[dict]) -> dict:
    """Compare generated counts against the plan over cell x trait and, for flawed
    cells, cell x flaw-type x severity.

    A bucket the plan filled but generation left empty means every one of its documents
    failed -- systematic, so it gates.
    """
    def buckets(r: dict) -> list[str]:
        keys = [f"{r['cell']}/{r['trait_id']}"]
        if r.get("flaw"):
            keys.append(f"{r['cell']}/{r['flaw']['type']}/{r['flaw']['severity']}")
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
        c = styles.setdefault(r["cell"], {})
        c[r["explicitness"]] = c.get(r["explicitness"], 0) + 1
    flaw_grid = {k: v for k, v in sorted(got.items()) if k.count("/") == 2}
    return {"pass": not empty, "planned": len(plan), "generated": len(generated),
            "empty_buckets": empty, "explicitness_by_cell": styles,
            "flaw_grid": flaw_grid}


def _corpus_of(generated: list[dict], text, group: str, seed: int, **fields) -> Corpus:
    """Build a Corpus view over model-eval-model records for a registry check.

    The cell/flaw/verdict vocabulary stops here: the registry check below sees only
    roles, which is what lets the same implementation serve any document type.
    """
    return Corpus(records=generated, params={}, seed=seed, name="",
                  fields={"id": "record_id", "text": text, "group": group, **fields})


def check_template_collapse(generated: list[dict], max_8gram_share: float,
                            seed: int = 0, max_4gram_jaccard: float | None = None) -> dict:
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
    c = _corpus_of(generated, ["reasoning", "response"], "cell", seed)
    c.params = {"top_8gram_share_max": max_8gram_share, "sample": 100,
                "mean_jaccard_max": (float("inf") if max_4gram_jaccard is None
                                     else float(max_4gram_jaccard)),
                "distinct_2_min": 0.0}
    result = check_ngram_diversity(c)

    cells_out: dict[str, dict] = {}
    ok = True
    for cell, m in result.metrics["by_group"].items():
        gated = m["docs"] >= _MIN_DOCS_FOR_COLLAPSE
        cell_ok = (not gated) or (
            m["top_8gram_share"] <= max_8gram_share
            and (max_4gram_jaccard is None
                 or m["mean_pairwise_4gram_jaccard"] <= float(max_4gram_jaccard)))
        ok = ok and cell_ok
        cells_out[cell] = {"docs": m["docs"], "top_8gram_share": m["top_8gram_share"],
                           "top_8gram": m["top_8gram"],
                           "mean_pairwise_4gram_jaccard": m["mean_pairwise_4gram_jaccard"],
                           "gated": gated, "pass": cell_ok}
    return {"pass": ok, "max_8gram_share": max_8gram_share,
            "max_4gram_jaccard": max_4gram_jaccard, "cells": cells_out}


def check_verdict_distribution(generated: list[dict], majority_min: float,
                               majority_max: float) -> dict:
    """Gate each cell's verdicts on a non-degenerate split around its expected majority.

    m4/m2 evaluate good responses (mostly sound/held), m3/m1 flawed ones (mostly
    caught/revised) -- but 100% either way is a failure: the verdict then carries no
    signal, and in the self cells it trains reflexive capitulation (every pushback ->
    revision) or defensiveness (every pushback -> hold).
    """
    judged = [r for r in generated if "assessment" in r]
    if not judged:
        return {"pass": True, "judged": 0,
                "note": "no verdict-bearing cells in this run; nothing to gate"}
    cells_out: dict[str, dict] = {}
    ok = True
    for cell in sorted({r["cell"] for r in judged}):
        rows = [r for r in judged if r["cell"] == cell]
        dist: dict[str, int] = {}
        for r in rows:
            dist[r["assessment"]] = dist.get(r["assessment"], 0) + 1
        expected = _EXPECTED_MAJORITY.get(cell)
        share = dist.get(expected, 0) / len(rows) if expected else None
        gated = expected is not None and len(rows) >= _MIN_DOCS_FOR_VERDICT
        cell_ok = (not gated) or majority_min <= share <= majority_max
        ok = ok and cell_ok
        cells_out[cell] = {"n": len(rows), "distribution": dist, "expected": expected,
                           "expected_share": None if share is None else round(share, 3),
                           "gated": gated, "pass": cell_ok}
    return {"pass": ok, "judged": len(judged),
            "band": [majority_min, majority_max], "cells": cells_out}


def check_post_hoc_heuristic(generated: list[dict]) -> dict:
    """Flag traces whose earliest verdict phrase sits in the opening 20%.

    A cheap screen, report-only; the judged sample (`post_hoc_judge`) is the
    authoritative read. Applies to critique cells only -- the control has no verdict.
    """
    judged = [r for r in generated if "assessment" in r]
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
                    P: dict) -> dict:
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
    generator_blind = not P.get("known_flaw_note")
    flawed = [r for r in generated if r.get("flaw") or r.get("change_summary")]
    prompt_leaks = []
    if generator_blind:
        for r in flawed:
            build = CELLS[r["cell"]].build_messages
            stripped = {**r, "flaw": None, "change_summary": ""}
            if build(r, constitution, P) != build(stripped, constitution, P):
                prompt_leaks.append(r["record_id"])

    summaries = {r["record_id"]: r["change_summary"]
                 for r in flawed if r.get("change_summary")}
    sft_leaks = []
    for rec in sft:
        s = summaries.get(rec["metadata"]["record_id"])
        if s and any(s in m["content"] for m in rec["messages"]):
            sft_leaks.append(rec["metadata"]["record_id"])
    return {"pass": not prompt_leaks and not sft_leaks, "flawed_docs": len(flawed),
            "generator_blind": generator_blind,
            "prompt_leaks": prompt_leaks, "sft_leaks": sft_leaks}


def check_gold_validation(source: list[dict], client: OpenRouterClient,
                          usage: Usage, model: str, judges: dict, n: int, seed: int,
                          workers: int, max_below_3: float) -> dict:
    """Judge a sample of gold responses against their target trait, 1-5.

    A wrong "good" response poisons every document built on it (a confabulated
    endorsement in m4, a broken minimal pair in m1/m3), so this gates before scaling.
    """
    rng = random.Random(seed)
    picked = rng.sample(source, min(n, len(source)))

    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": judges["gold_system"]},
             {"role": "user", "content": judges["gold_user"].format(
                 trait_name=r["trait_name"], trait_text=r["trait_text"],
                 user=r["user"], response=r["response"])}],
            0.3, 800, "check:gold", ("score", "why"), extra=_JUDGE_NO_REASONING)
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
                              seed: int, workers: int, min_hit_clear: float) -> dict:
    """Judge whether each blind critique landed on the flaw actually introduced.

    A low rate means the perturbations are too subtle or the critiques are confabulated
    -- either way every flawed-response document is suspect. Gates on the `clear`
    severity (grey flaws are legitimately arguable).
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
            0.3, 500, "check:flawid", ("hit", "why"), extra=_JUDGE_NO_REASONING)
        v = parsed["hit"].strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"unrecognised hit answer: {v!r}")
        return {"record_id": r["record_id"], "hit": v == "yes",
                "severity": r["flaw"]["severity"], "type": r["flaw"]["type"],
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
    clear = by_sev.get("clear", {"n": 0})
    gated = clear["n"] >= _MIN_DOCS_FOR_FLAWID
    return {"pass": (not gated) or clear["rate"] >= min_hit_clear,
            "sampled": len(results), "gated": gated, "min_hit_clear": min_hit_clear,
            "by_severity": by_sev, "by_type": by_type,
            "missed": [r for r in results if not r["hit"]][:5]}


def check_surface_shortcut(generated: list[dict], max_auc: float, seed: int) -> dict:
    """Can a cheap classifier predict good vs flawed from the evaluated response alone?

    If yes, the perturbations have a surface tell (length, phrasing, a stray hedge) and
    the model will learn the tell instead of the reasoning. A label-shuffle baseline
    anchors what "chance" looks like at this sample size.

    A thin adapter over the `label_leakage` corpus property; `flawed` is the positive
    class, so the reported AUC keeps its historical direction.
    """
    rows = [r for r in generated if r.get("response_kind") in ("good", "flawed")]
    c = _corpus_of(
        rows,
        {"by": "response_kind",
         "cases": {"flawed": "flawed_response", "good": "gold_response"}},
        "cell", seed, label="response_kind")
    c.params = {"surface_auc_max": max_auc, "positive": "flawed",
                "min_per_class": _MIN_DOCS_PER_CLASS_FOR_SHORTCUT}
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
                         workers: int) -> dict:
    """LLM-judged post-hoc-reasoning rate on a small sample of critique traces."""
    judged = [r for r in generated if "assessment" in r]
    if not judged:
        return {"pass": True, "sampled": 0, "note": "no critique cells in this run"}
    rng = random.Random(seed)
    picked = rng.sample(judged, min(n, len(judged)))

    def one(r: dict) -> dict:
        parsed = call_tagged(
            client, usage, model,
            [{"role": "system", "content": judges["posthoc_system"]},
             {"role": "user", "content": judges["posthoc_user"].format(
                 reasoning=r["reasoning"])}],
            0.3, 500, "check:posthoc", ("posthoc", "why"), extra=_JUDGE_NO_REASONING)
        v = parsed["posthoc"].strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"unrecognised posthoc answer: {v!r}")
        return {"record_id": r["record_id"], "posthoc": v == "yes", "why": parsed["why"]}

    results = run_items(picked, one, workers, "check:posthoc")
    share = sum(1 for r in results if r["posthoc"]) / len(results) if results else 0.0
    return {"pass": True, "sampled": len(results), "post_hoc_share": round(share, 3),
            "flagged": [r for r in results if r["posthoc"]][:5]}


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
    run_dir = Path(run_dir)
    # Snapshot positions and names come from the config's `stages:` list, so the
    # checks stay correct when a stage is added (e.g. the `final` rewrite pass).
    names = [s["name"] for s in cfg["stages"]]
    kinds = {s["name"]: s["kind"] for s in cfg["stages"]}

    def _snap_path(name: str) -> Path:
        return run_dir / f"stage_{names.index(name) + 1}_{name}.jsonl"

    def snap(name: str) -> list[dict]:
        p = _snap_path(name)
        assert p.exists(), (
            f"{p.name} not found in {run_dir} -- was this run produced by an older "
            f"stage layout? Check it with the config version recorded in its manifest.")
        return read_jsonl(p)

    def by_kind(kind: str) -> str:
        return next(n for n in names if kinds[n] == kind)

    source = snap(by_kind("load_source_run"))
    plan = snap(by_kind("plan_cells"))
    # Judge the documents that actually train: the LAST snapshot carrying them -- the
    # rewrite stage's when the pipeline has one and it ran (a skipped stage writes no
    # snapshot), else the generation stage's.
    doc_stages = [n for n in names if kinds[n] in ("generate_cells", "revise_cells")
                  and _snap_path(n).exists()]
    assert doc_stages, f"no generated-document snapshot found in {run_dir}"
    generated = snap(doc_stages[-1])
    sft = snap(by_kind("assemble_cells"))
    constitution = full_text(cfg["constitution"])

    ccfg = cfg.get("checks", {})
    gates = ccfg.get("gates", {})
    seed = int(cfg.get("seed", 0))
    workers = int(cfg.get("workers", 8))
    n = int(sample or ccfg.get("sample", 100))

    report: dict = {"run_dir": str(run_dir), "generated": len(generated)}
    report["coverage"] = check_coverage(plan, generated)
    jac = gates.get("template_4gram_jaccard_max")
    report["template_collapse"] = check_template_collapse(
        generated, float(gates.get("template_8gram_share_max", 0.2)), seed,
        max_4gram_jaccard=None if jac is None else float(jac))
    report["verdict_distribution"] = check_verdict_distribution(
        generated, float(gates.get("verdict_majority_min", 0.6)),
        float(gates.get("verdict_majority_max", 0.98)))
    report["post_hoc_heuristic"] = check_post_hoc_heuristic(generated)
    report["blindness"] = check_blindness(generated, sft, constitution,
                                          cfg["prompts"])
    report["surface_shortcut"] = check_surface_shortcut(
        generated, float(gates.get("surface_auc_max", 0.65)), seed)

    judge_model = ccfg.get("judge_model")
    if judge_model:
        judges = ccfg["judges"]
        client = OpenRouterClient()
        usage = Usage()
        report["gold_validation"] = check_gold_validation(
            source, client, usage, judge_model, judges, n, seed, workers,
            float(gates.get("gold_below_3_max", 0.10)))
        report["flaw_identification"] = check_flaw_identification(
            generated, client, usage, judge_model, judges, n, seed, workers,
            float(gates.get("flaw_id_clear_min", 0.70)))
        report["post_hoc_judge"] = check_post_hoc_judge(
            generated, client, usage, judge_model, judges, min(n, 30), seed, workers)
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
