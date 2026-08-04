# ABOUTME: Corpus validity checks for a MEM run: properties of the data, not the
# ABOUTME: pipeline. Gated by config thresholds; report always written before gating.

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.endpoints.openrouter import OpenRouterClient  # noqa: E402

from . import prompts, stages  # noqa: E402
from .constitution import full_text  # noqa: E402
from .hf_cache import read_jsonl  # noqa: E402

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
_MIN_DOCS_FOR_COLLAPSE = 5
_MIN_DOCS_FOR_VERDICT = 20


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of word n-grams of a token list."""
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _doc_text(r: dict) -> str:
    """The generated prose of one document (reasoning plus reply where present)."""
    return r["reasoning"] + "\n" + r.get("response", "")


def check_coverage(plan: list[dict], generated: list[dict]) -> dict:
    """Compare generated counts against the plan over cell x trait and explicitness.

    A (cell, trait) bucket the plan filled but generation left empty means every one of
    its documents failed -- systematic, so it gates.
    """
    def bucket(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[f"{r['cell']}/{r['trait_id']}"] = out.get(f"{r['cell']}/{r['trait_id']}", 0) + 1
        return out

    planned, got = bucket(plan), bucket(generated)
    empty = sorted(k for k in planned if got.get(k, 0) == 0)
    styles: dict[str, dict[str, int]] = {}
    for r in generated:
        c = styles.setdefault(r["cell"], {})
        c[r["explicitness"]] = c.get(r["explicitness"], 0) + 1
    return {"pass": not empty, "planned": len(plan), "generated": len(generated),
            "empty_buckets": empty, "explicitness_by_cell": styles}


def check_template_collapse(generated: list[dict], max_8gram_share: float,
                            seed: int = 0) -> dict:
    """Detect critique-shape collapse: repeated long n-grams and high pairwise overlap.

    Generators converge on one critique shape quickly; a single 8-gram appearing in a
    large share of a cell's documents is the fingerprint.
    """
    by_cell: dict[str, list[dict]] = {}
    for r in generated:
        by_cell.setdefault(r["cell"], []).append(r)

    cells_out: dict[str, dict] = {}
    ok = True
    for cell, rows in sorted(by_cell.items()):
        docs = [_words(_doc_text(r)) for r in rows]
        grams8: dict[tuple[str, ...], int] = {}
        for d in docs:
            for g in _ngrams(d, 8):
                grams8[g] = grams8.get(g, 0) + 1
        top = max(grams8.items(), key=lambda kv: kv[1], default=((), 0))
        top_share = top[1] / max(len(docs), 1)

        rng = random.Random(seed)
        sample = rng.sample(docs, min(len(docs), 100))
        sets4 = [_ngrams(d, 4) for d in sample]
        sims = [len(a & b) / max(len(a | b), 1)
                for i, a in enumerate(sets4) for b in sets4[i + 1:]]
        mean_j = sum(sims) / len(sims) if sims else 0.0

        gated = len(docs) >= _MIN_DOCS_FOR_COLLAPSE
        cell_ok = (not gated) or top_share <= max_8gram_share
        ok = ok and cell_ok
        cells_out[cell] = {"docs": len(docs), "top_8gram_share": round(top_share, 3),
                           "top_8gram": " ".join(top[0]),
                           "mean_pairwise_4gram_jaccard": round(mean_j, 4),
                           "gated": gated, "pass": cell_ok}
    return {"pass": ok, "max_8gram_share": max_8gram_share, "cells": cells_out}


def check_verdict_distribution(generated: list[dict], sound_min: float,
                               sound_max: float) -> dict:
    """Gate on a non-degenerate verdict split among the critique cells.

    All-critical trains a critic reflex (downstream over-hedging); all-sound means the
    assessment carries no signal at all.
    """
    judged = [r for r in generated if "assessment" in r]
    if not judged:
        return {"pass": True, "judged": 0,
                "note": "no critique cells in this run; nothing to gate"}
    by_cell: dict[str, dict[str, int]] = {}
    for r in judged:
        c = by_cell.setdefault(r["cell"], {})
        c[r["assessment"]] = c.get(r["assessment"], 0) + 1
    share_sound = sum(1 for r in judged if r["assessment"] == "sound") / len(judged)
    gated = len(judged) >= _MIN_DOCS_FOR_VERDICT
    return {"pass": (not gated) or sound_min <= share_sound <= sound_max,
            "judged": len(judged), "gated": gated,
            "share_sound": round(share_sound, 3),
            "band": [sound_min, sound_max], "by_cell": by_cell}


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


def check_blindness(generated: list[dict], sft: list[dict], constitution: str) -> dict:
    """Prove the flaw label never reaches a generation prompt or a training record.

    For every flawed document, the generation messages rebuilt with the flaw metadata
    stripped must be byte-identical to the real ones (the evaluated response text
    itself stays -- that is the input, not the label), and the perturbation stage's
    `change_summary` must not appear verbatim in any training message.
    """
    flawed = [r for r in generated if r.get("flaw") or r.get("change_summary")]
    prompt_leaks = []
    for r in flawed:
        build = stages.CELLS[r["cell"]].build_messages
        stripped = {**r, "flaw": None, "change_summary": ""}
        if build(r, constitution) != build(stripped, constitution):
            prompt_leaks.append(r["record_id"])

    summaries = {r["record_id"]: r["change_summary"]
                 for r in flawed if r.get("change_summary")}
    sft_leaks = []
    for rec in sft:
        s = summaries.get(rec["metadata"]["record_id"])
        if s and any(s in m["content"] for m in rec["messages"]):
            sft_leaks.append(rec["metadata"]["record_id"])
    return {"pass": not prompt_leaks and not sft_leaks, "flawed_docs": len(flawed),
            "prompt_leaks": prompt_leaks, "sft_leaks": sft_leaks}


def check_gold_validation(source: list[dict], client: OpenRouterClient,
                          usage: stages.Usage, model: str, n: int, seed: int,
                          workers: int, max_below_3: float) -> dict:
    """Judge a sample of gold responses against their target trait, 1-5.

    A wrong "good" response poisons every document built on it (a confabulated
    endorsement in m4, a broken minimal pair in m1/m3), so this gates before scaling.
    """
    rng = random.Random(seed)
    picked = rng.sample(source, min(n, len(source)))

    def one(r: dict) -> dict:
        parsed = stages._call_tagged(
            client, usage, model,
            prompts.GOLD_JUDGE_SYSTEM,
            prompts.GOLD_JUDGE_USER.format(trait_name=r["trait_name"],
                                           trait_text=r["trait_text"],
                                           user=r["user"], response=r["response"]),
            0.3, 800, "check:gold", ("score", "why"))
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")
        return {"scenario_id": r["scenario_id"], "score": score, "why": parsed["why"]}

    scored = stages._run_items(picked, one, workers, "check:gold")
    scores = [s["score"] for s in scored]
    below = sum(1 for s in scores if s < 3)
    share_below = below / len(scores) if scores else 1.0
    dist = {str(v): scores.count(v) for v in range(1, 6)}
    worst = sorted(scored, key=lambda s: s["score"])[:3]
    return {"pass": share_below <= max_below_3, "sampled": len(scores),
            "share_below_3": round(share_below, 3), "max_below_3": max_below_3,
            "distribution": dist, "worst": worst}


def check_post_hoc_judge(generated: list[dict], client: OpenRouterClient,
                         usage: stages.Usage, model: str, n: int, seed: int,
                         workers: int) -> dict:
    """LLM-judged post-hoc-reasoning rate on a small sample of critique traces."""
    judged = [r for r in generated if "assessment" in r]
    if not judged:
        return {"pass": True, "sampled": 0, "note": "no critique cells in this run"}
    rng = random.Random(seed)
    picked = rng.sample(judged, min(n, len(judged)))

    def one(r: dict) -> dict:
        parsed = stages._call_tagged(
            client, usage, model,
            prompts.POSTHOC_JUDGE_SYSTEM,
            prompts.POSTHOC_JUDGE_USER.format(reasoning=r["reasoning"]),
            0.3, 500, "check:posthoc", ("posthoc", "why"))
        v = parsed["posthoc"].strip().lower()
        if v not in ("yes", "no"):
            raise ValueError(f"unrecognised posthoc answer: {v!r}")
        return {"record_id": r["record_id"], "posthoc": v == "yes", "why": parsed["why"]}

    results = stages._run_items(picked, one, workers, "check:posthoc")
    share = sum(1 for r in results if r["posthoc"]) / len(results) if results else 0.0
    return {"pass": True, "sampled": len(results), "post_hoc_share": round(share, 3),
            "flagged": [r for r in results if r["posthoc"]][:5]}


def run_checks(run_dir: str | Path, cfg: dict,
               sample: int | None = None) -> tuple[dict, bool]:
    """Run every applicable validity check over a MEM run and write the report.

    The report is always written in full before any gating, so a failed run is still
    inspectable. Checks marked report-only always pass; gated checks read their
    thresholds from `cfg["checks"]["gates"]`.

    Args:
        run_dir: The MEM run directory (stage snapshots + manifest).
        cfg: The MEM run config.
        sample: Override for the LLM-judged sample size.

    Returns:
        (the report dict, whether every gated check passed).
    """
    run_dir = Path(run_dir)
    source = read_jsonl(run_dir / "stage_1_source.jsonl")
    plan = read_jsonl(run_dir / "stage_2_plan.jsonl")
    generated = read_jsonl(run_dir / "stage_4_generated.jsonl")
    sft = read_jsonl(run_dir / "stage_5_sft.jsonl")
    constitution = full_text(cfg["constitution"])

    ccfg = cfg.get("checks", {})
    gates = ccfg.get("gates", {})
    seed = int(cfg.get("seed", 0))
    workers = int(cfg.get("workers", 8))
    n = int(sample or ccfg.get("sample", 100))

    report: dict = {"run_dir": str(run_dir), "generated": len(generated)}
    report["coverage"] = check_coverage(plan, generated)
    report["template_collapse"] = check_template_collapse(
        generated, float(gates.get("template_8gram_share_max", 0.2)), seed)
    report["verdict_distribution"] = check_verdict_distribution(
        generated, float(gates.get("sound_verdict_min", 0.0)),
        float(gates.get("sound_verdict_max", 1.0)))
    report["post_hoc_heuristic"] = check_post_hoc_heuristic(generated)
    report["blindness"] = check_blindness(generated, sft, constitution)

    judge_model = ccfg.get("judge_model")
    if judge_model:
        client = OpenRouterClient()
        usage = stages.Usage()
        report["gold_validation"] = check_gold_validation(
            source, client, usage, judge_model, n, seed, workers,
            float(gates.get("gold_below_3_max", 0.10)))
        report["post_hoc_judge"] = check_post_hoc_judge(
            generated, client, usage, judge_model, min(n, 30), seed, workers)
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
