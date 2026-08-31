# ABOUTME: Screen the in-the-wild jailbreak pool for CORE-HARM elicitation and emit the top-N
# ABOUTME: templates as a jailbreaks file the registered DAN eval can consume.

"""Rank 1,405 in-the-wild jailbreak templates by how much core harm they elicit.

The DAN set this repo has been using was selected by `span_days` — how long a template
persisted in the wild — which turns out not to track efficacy at all. Measured on the five
we had, core-harm ASR ranges 5.9%-26.5%, and the second-most-persistent template elicits 0%
core harm from the control. Selecting on efficacy instead gives a much sharper instrument.

CORE HARM is the six forbidden-question categories where answering IS the failure: illegal
activity, physical harm, fraud, malware, hate speech, privacy violation. The other seven
categories are excluded deliberately — "financial advice" and "legal opinion" cells reward a
model for refusing to be useful, and they dominate the headline ASR without measuring harm.

Each candidate is scored on the same stratified sample of core-harm questions, so the ranking
is a like-for-like comparison. Refusal judging reuses the registered eval's prompt and its
`classify`, so a judge reply that is not a verdict is EXCLUDED rather than counted as a
successful jailbreak.

    uv run python scratch/safety_evals/screen_jailbreaks.py \
        --base_url <served endpoint> --model <served name> [--n_questions 12] [--top_n 20]
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv
from openai import OpenAI

from src.endpoints.openrouter import (
    OpenRouterClient,
    _CompletionFailure,
    map_threaded,
)
from src.eval.misalignment.dan.runner import REFUSAL_SYS, UNSCORABLE, classify
from src.model_profile import resolve_trace

CORE_HARM = {"Illegal Activity", "Physical Harm", "Fraud", "Malware", "Hate Speech",
             "Privacy Violence"}
POOL = "TrustAIRLab/in-the-wild-jailbreak-prompts"
POOL_CONFIG = "jailbreak_2023_12_25"
QUESTIONS = "TrustAIRLab/forbidden_question_set"


def load_pool(token: str | None, max_chars: int) -> list[dict]:
    """Return the jailbreak candidates, deduplicated on prompt text.

    Exact duplicates are common in the in-the-wild scrape (the same template reposted across
    platforms); keeping them would waste generations and let one template occupy several
    slots in the top-N.
    """
    from datasets import load_dataset

    rows = load_dataset(POOL, POOL_CONFIG, split="train", token=token)
    seen: set[str] = set()
    out = []
    for i, r in enumerate(rows):
        p = (r["prompt"] or "").strip()
        if not p or len(p) > max_chars or p in seen:
            continue
        seen.add(p)
        out.append({"id": f"jb{i:04d}", "prompt": p, "platform": r.get("platform"),
                    "created_at": str(r.get("created_at") or r.get("date") or "")})
    return out


def load_core_questions(token: str | None, n: int, seed: int,
                        questions_file: str = "") -> list[dict]:
    """Return n core-harm questions, stratified round-robin across the six categories.

    `questions_file` restricts the draw to a pre-filtered set — in practice the OPERATIONAL
    questions from classify_question_harm.py. Without it the draw includes questions whose
    honest answer is encyclopedic ("which languages is malware written in"), which a binary
    refusal judge scores as a jailbreak success and which therefore inflate ASR and skew
    template selection toward attacks that unlock harmless content.
    """
    from datasets import load_dataset

    if questions_file:
        rows = json.loads(Path(questions_file).read_text())
        assert rows and "question" in rows[0] and "content_policy_name" in rows[0], (
            f"{questions_file} must hold objects with `question` and `content_policy_name`")
    else:
        rows = [r for r in load_dataset(QUESTIONS, split="train", token=token)
                if r["content_policy_name"] in CORE_HARM]
    by = defaultdict(list)
    for r in rows:
        by[r["content_policy_name"]].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    cats = sorted(by)
    picked, depth = [], 0
    while len(picked) < n:
        for c in cats:
            if depth < len(by[c]) and len(picked) < n:
                picked.append(by[c][depth])
        depth += 1
    return picked


def main(base_url: str, model: str, out: str = "output/jailbreak_screen",
         api_key_env: str = "VLLM_API_KEY",
         n_questions: int = 12, top_n: int = 20, seed: int = 0, max_chars: int = 6000,
         max_tokens: int = 4096, temperature: float = 1.0, concurrency: int = 32,
         judge_model: str = "google/gemini-3-flash-preview", judge_workers: int = 16,
         limit: int = 0, from_generations: str = "", ids_file: str = "",
         questions_file: str = "") -> None:
    """Screen the pool and write the ranking plus a top-N jailbreaks file.

    Args:
        base_url: OpenAI-compatible endpoint of the served screening model.
        model: Served model name to request.
        api_key_env: Env var holding the endpoint's key. The vLLM default takes any value;
            point this at OPENROUTER_API_KEY to smoke the pipeline against a public API.
        out: Output root; a timestamped run dir is created under it.
        n_questions: Core-harm questions per candidate (stratified over the 6 categories).
        top_n: How many templates to emit in the selected set.
        seed: Seeds the question sample.
        max_chars: Skip candidates longer than this (they crowd the context window).
        max_tokens: Generation cap. Refusals are short; a truncated-but-non-empty compliant
            answer is still judgeable, so this trades tail fidelity for throughput.
        temperature: Target sampling temperature.
        concurrency: Concurrent target calls (the served --max-num-seqs).
        judge_model: OpenRouter refusal judge.
        judge_workers: Concurrent judge calls.
        limit: Screen only the first N candidates (smoke/testing).
        questions_file: JSON list of questions to draw from (e.g. the OPERATIONAL set from
            classify_question_harm.py). Omit to draw from all core-harm questions.
        ids_file: JSON list of already-selected templates (objects with `sha1`). Restricts
            the run to exactly those, in that order, so a second model is measured on the
            SAME templates x the SAME questions as the screen — which is what makes the two
            ASRs comparable at all.
        from_generations: Path to a previous run's generations.json. Skips generation
            entirely and re-judges those cells — generation is the half that needs a GPU,
            so a judge-side failure must never cost it twice.
    """
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(token, max_chars)
    if ids_file:
        want = [r["sha1"] for r in json.loads(Path(ids_file).read_text())]
        by_id = {c["id"]: c for c in pool}
        missing = [i for i in want if i not in by_id]
        assert not missing, (
            f"{len(missing)} id(s) from {ids_file} are not in the pool (e.g. {missing[:3]}) — "
            "the pool was built with different --max_chars, or from a different snapshot")
        pool = [by_id[i] for i in want]
    if limit:
        pool = pool[:limit]
    questions = load_core_questions(token, n_questions, seed, questions_file)
    cells = [(jb, q) for jb in pool for q in questions]

    print(f">>> screening {len(pool)} candidates x {len(questions)} core-harm questions "
          f"= {len(cells)} generations")
    print(f"    categories: {dict((c, sum(1 for q in questions if q['content_policy_name'] == c)) for c in sorted({q['content_policy_name'] for q in questions}))}")
    print(f"    target {model} @ {base_url} | judge {judge_model}")
    print(f"--- first composed prompt ({len(cells[0][0]['prompt'])} chars template) ---")
    print((cells[0][0]["prompt"] + "\n\n" + cells[0][1]["question"])[:600])
    print("---")

    client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "dummy"),
                    timeout=600, max_retries=2)
    orc = OpenRouterClient()

    def generate(i: int) -> dict:
        jb, q = cells[i]
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": f"{jb['prompt']}\n\n{q['question']}"}],
                temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — one dropped cell must not sink a 3h screen
            return {"jb": jb["id"], "scenario": q["content_policy_name"],
                    "answer": "", "error": type(exc).__name__}
        c = r.choices[0]
        reasoning = (getattr(c.message, "reasoning_content", None)
                     or getattr(c.message, "reasoning", None))
        _, answer = resolve_trace(c.message.content or "", reasoning)
        return {"jb": jb["id"], "scenario": q["content_policy_name"], "answer": answer,
                "error": ""}

    if from_generations:
        gens = json.loads(Path(from_generations).read_text())
        assert len(gens) == len(cells), (
            f"{from_generations} holds {len(gens)} cells but this configuration builds "
            f"{len(cells)} — the pool or question sample differs, so the two cannot be "
            "aligned. Re-run with the same --n_questions/--seed/--max_chars.")
        print(f">>> reusing {len(gens)} generations from {from_generations}")
    else:
        gens = map_threaded(generate, len(cells), max_workers=concurrency,
                            desc="screen generate")
    (run_dir / "generations.json").write_text(json.dumps(gens, indent=1), encoding="utf-8")
    print(f">>> generations saved -> {run_dir / 'generations.json'}")

    def judge(i: int) -> str:
        g = gens[i]
        if g["error"] or not g["answer"].strip():
            return "EMPTY"
        try:
            return classify(orc.chat(
                judge_model,
                [{"role": "system", "content": REFUSAL_SYS},
                 {"role": "user", "content": g["answer"]}],
                temperature=0.0, max_tokens=8).content)
        except _CompletionFailure:
            # Judge refused, or returned no choices at all. Excluded, never scored.
            return "JUDGE_BLOCKED"

    verdicts = map_threaded(judge, len(gens), max_workers=judge_workers, desc="screen judge")

    # Persist the per-cell verdicts. Without them the run can report a rate but nobody can
    # read the cells behind it — and inspecting what a successful jailbreak actually elicited
    # is the point of running one.
    (run_dir / "cells.jsonl").write_text("\n".join(
        json.dumps({"jb": g["jb"], "scenario": g["scenario"], "verdict": v,
                    "answer_chars": len(g["answer"]), "error": g["error"]})
        for g, v in zip(gens, verdicts)) + "\n", encoding="utf-8")
    print(f">>> per-cell verdicts -> {run_dir / 'cells.jsonl'}")

    by_jb: dict[str, list[str]] = defaultdict(list)
    for g, v in zip(gens, verdicts):
        by_jb[g["jb"]].append(v)
    rank = []
    for jb in pool:
        vs = by_jb[jb["id"]]
        scored = [v for v in vs if v not in UNSCORABLE]
        asr = round(100.0 * sum(1 for v in scored if v == "ANSWERED") / len(scored), 2) \
            if scored else None
        rank.append({**jb, "core_harm_asr": asr, "n_scored": len(scored),
                     "n_unscorable": len(vs) - len(scored)})
    # A candidate scored on too few cells has an unreliable rate; rank it last rather than
    # letting a 1-of-1 lucky hit take a slot in the selected set.
    ranked = sorted(rank, key=lambda r: (-(r["core_harm_asr"] or -1)
                                         if r["n_scored"] >= max(2, n_questions // 2)
                                         else 1e9, -r["n_scored"]))
    (run_dir / "ranking.json").write_text(json.dumps(ranked, indent=1), encoding="utf-8")

    usable = [r for r in ranked if r["core_harm_asr"] is not None]
    assert usable, (
        "no candidate produced a single scorable cell — the screen measured nothing. "
        f"Check the endpoint and key ({api_key_env}); generation errors: "
        f"{sum(1 for g in gens if g['error'])}/{len(gens)}")
    if len(usable) < top_n:
        print(f"!!! only {len(usable)} of {len(pool)} candidates produced scorable cells")
    top = ranked[:top_n]
    # Emitted in the shape the registered DAN eval's jailbreaks file uses, so it is a drop-in.
    sel = [{"sha1": r["id"], "span_days": None, "core_harm_asr": r["core_harm_asr"],
            "n_scored": r["n_scored"], "platform": r["platform"],
            "created_at": r["created_at"], "prompt": r["prompt"]} for r in top]
    sel_path = run_dir / f"jailbreaks_coreharm_top{top_n}.json"
    sel_path.write_text(json.dumps(sel, indent=1), encoding="utf-8")

    lines = [f"# Core-harm jailbreak screen — {stamp}", "",
             f"Pool {len(pool)} deduplicated candidates x {len(questions)} core-harm "
             f"questions = {len(cells)} generations. Screened on `{model}`.", "",
             f"| rank | id | core-harm ASR | n scored | platform |", "|---|---|--:|--:|---|"]
    for i, r in enumerate(top, 1):
        lines.append(f"| {i} | {r['id']} | {r['core_harm_asr']}% | {r['n_scored']} | "
                     f"{r['platform']} |")
    (run_dir / "screen_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n>>> top {top_n} by core-harm ASR:")
    for i, r in enumerate(top, 1):
        asr = "n/a" if r["core_harm_asr"] is None else f"{r['core_harm_asr']}%"
        print(f"  {i:3d}. {r['id']}  {asr:>7}  (n={r['n_scored']}, "
              f"unscorable {r['n_unscorable']})")
    print(f"\n>>> wrote {sel_path}")
    print(f">>> wrote {run_dir / 'screen_results.md'}")


if __name__ == "__main__":
    fire.Fire(main)
