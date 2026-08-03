# ABOUTME: Run the MMLU subset against one or more served arms and grade the answers.
# ABOUTME: Run: uv run python src/eval/capabilities/mmlu_eval.py --arms all --endpoint <url>

"""MMLU generation + grading for the capability regression eval.

One command evaluates every arm, because all arms are served concurrently as LoRA modules
off a single vLLM process (`scripts/gpu/runpod_capability.py`). That is not just convenient:
it means every arm is measured by the same process, on the same GPU, with the same build
and flags, so decoding parity is a property of the setup rather than something we have to
trust across separate boots.

Three things this does that a generic MMLU harness does not, all of them because the
models under test are thinking models:

- **Splits and measures the `<think>` trace.** Accepts it inline *or* out of band under
  either field name vLLM has used for it (`reasoning_content` on 0.8.x, `reasoning` on
  0.26), grades only the visible answer, and reports trace length so CLAUDE.md gotcha 2 —
  the empty-`<think>` collapse that plain SFT induces — stays checkable per arm.
- **Reports format compliance separately from correctness.** An unparseable answer scores
  wrong, but `parse_rate`, the parse-tier distribution and `truncation_rate` are printed
  next to every accuracy number, because the difference between "lost knowledge" and "ran
  out of tokens mid-trace" is invisible in accuracy alone and they demand opposite fixes.
- **Caches on prompt content, not just uid.** Cached generations carry the hash of the
  prompt that produced them, so editing the prompt template invalidates them instead of
  silently mixing two prompt formats into one accuracy number.

    # every trained arm, against a RunPod-served endpoint
    uv run python src/eval/capabilities/mmlu_eval.py --arms all \
        --endpoint https://<pod>-8000.proxy.runpod.net/v1

    # one arm, quick wiring check (2 questions per subject)
    uv run python src/eval/capabilities/mmlu_eval.py --arms arm_base --smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fire
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.endpoints.openrouter import map_threaded  # noqa: E402
from src.eval.capabilities.mmlu import (  # noqa: E402
    build_prompt,
    build_subset,
    load_split,
    parse_answer,
    prompt_hash,
    resolve_trace,
    score_records,
    subset_hash,
)
from src.utils import read_jsonl, timestamp, write_run_meta  # noqa: E402


def resolve_arms(cfg: DictConfig) -> list[dict]:
    """Load the arm ladder from the capability-eval config and attach served names.

    The ladder is not restated in the MMLU config on purpose (see the `arms_from` note
    there): one source of truth means a newly-trained arm cannot be missing from this
    eval while appearing in the other.

    Returns:
        One dict per arm with `name`, `served`, `adapter`, `synthetic_fraction`, `role`
        and `trained`.
    """
    source = OmegaConf.load(str(cfg.arms_from))
    served_names = OmegaConf.to_container(cfg.get("served_names") or {}, resolve=True)
    arms = []
    for arm in source.arms:
        name = str(arm.name)
        adapter = None if arm.adapter is None else str(arm.adapter)
        role = str(arm.get("role", ""))
        arms.append(
            {
                "name": name,
                "served": str(served_names.get(name, name)),
                "adapter": adapter,
                "synthetic_fraction": arm.get("synthetic_fraction"),
                "role": role,
                # `adapter: null` means the checkpoint does not exist yet, EXCEPT for the
                # floor arm, which is the bare base model and has no adapter by design.
                "trained": adapter is not None or role == "floor",
            }
        )
    return arms


def _shots_by_subject(cfg: DictConfig, seed: int) -> dict[str, list[dict]]:
    """Build the few-shot demonstrations for each subject from the dev split.

    Demo uids are namespaced with a `dev:` prefix before shuffling, so a demo and a test
    question with the same subject-relative index do not share a choice permutation.
    """
    n_shot = int(cfg.prompt.n_shot)
    if n_shot <= 0:
        return {}
    rows = load_split(str(cfg.prompt.shot_split), str(cfg.subset.dataset), str(cfg.subset.name))
    for row in rows:
        row["uid"] = f"dev:{row['uid']}"
    # per_subject is capped at the dev split's 5-per-subject by build_subset itself.
    demos = build_subset(
        rows,
        per_subject=n_shot,
        seed=seed,
        shuffle_choices=bool(cfg.subset.shuffle_choices),
    )
    by_subject: dict[str, list[dict]] = {}
    for demo in demos:
        by_subject.setdefault(demo["subject"], []).append(demo)
    return by_subject


def _grade(record: dict, question: dict) -> dict:
    """Attach parsing and correctness to one raw generation."""
    parsed, tier = parse_answer(record["answer"], question["n_choices"])
    return {
        **record,
        "subject": question["subject"],
        "category": question["category"],
        "answer_letter": question["answer_letter"],
        "parsed": parsed,
        "parse_tier": tier,
        "correct": parsed == question["answer_letter"],
        "think_words": len(record["think"].split()),
    }


def run_arm(
    arm: dict,
    questions: list[dict],
    shots: dict[str, list[dict]],
    cfg: DictConfig,
    endpoint: str,
    out_root: Path,
) -> dict[str, Any]:
    """Generate, grade and summarise one arm over the shared question subset.

    Returns:
        The arm's score block, as produced by `score_records`.
    """
    gen = cfg.generation
    mode = "think" if bool(gen.enable_thinking) else "nothink"
    arm_dir = out_root / mode / arm["name"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    records_file = arm_dir / "records.jsonl"

    prompts = {
        q["uid"]: build_prompt(
            q,
            shots.get(q["subject"], []),
            instruction=str(cfg.prompt.instruction),
            cue=str(cfg.prompt.cue),
        )
        for q in questions
    }

    # Resume, but only for generations produced by the CURRENT prompt. Reusing an answer
    # produced under a different prompt template silently mixes two formats into one
    # accuracy number, which is exactly the kind of error that survives review because
    # the result still looks plausible.
    cached: dict[str, dict] = {}
    stale = 0
    retry_failed = 0
    if records_file.exists():
        for rec in read_jsonl(records_file):
            if rec.get("prompt_hash") != prompt_hash(prompts.get(rec["uid"], "")):
                stale += 1
            elif rec.get("finish_reason") == "timeout":
                # A recorded failure is not a result. Re-running must retry it rather
                # than treating a dropped connection as a wrong answer forever.
                retry_failed += 1
            else:
                cached[rec["uid"]] = rec

    todo = [q for q in questions if q["uid"] not in cached]
    print(f"\n>>> arm {arm['name']}  (served as {arm['served']!r}, adapter={arm['adapter']})")
    print(f"    questions: {len(questions)} total, {len(cached)} cached, {len(todo)} to generate"
          + (f", {stale} stale (prompt changed)" if stale else "")
          + (f", {retry_failed} retrying (previously timed out)" if retry_failed else ""))

    client = OpenAI(
        base_url=endpoint,
        api_key=str(gen.api_key),
        timeout=float(gen.get("request_timeout", 180)),
        max_retries=int(gen.get("max_retries", 2)),
    )

    def generate(i: int) -> dict:
        q = todo[i]
        prompt = prompts[q["uid"]]
        try:
            return _one(q, prompt)
        except Exception as exc:  # noqa: BLE001
            # A request that times out or errors after the SDK's own retries must not
            # take the whole run down with it — `map_threaded` is fail-fast, and losing
            # four arms of finished work to one dropped connection is a bad trade.
            # Recorded as an empty answer so it scores unparseable and shows up in
            # `parse_rate`, and marked `timeout` so the cache below refuses it and the
            # next run retries just these rather than baking a failure into the result.
            print(f"    !! {q['uid']}: {type(exc).__name__} — recorded as timeout")
            return {
                "uid": q["uid"],
                "prompt_hash": prompt_hash(prompt),
                "raw": "",
                "think": "",
                "answer": "",
                "finish_reason": "timeout",
            }

    def _one(q: dict, prompt: str) -> dict:
        resp = client.chat.completions.create(
            model=arm["served"],
            messages=[{"role": "user", "content": prompt}],
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            max_tokens=int(gen.max_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(gen.enable_thinking)}},
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        # The out-of-band trace field is NOT stable across vLLM versions: 0.8.x used
        # `reasoning_content`, 0.26 returns `reasoning`. Reading only one of them
        # reports every trace as empty and trips the gotcha-2 alarm on a model that is
        # reasoning normally. Verified against the live endpoint: vllm-0.26.0 populates
        # `reasoning` and leaves `reasoning_content` unset.
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
            choice.message, "reasoning", None
        )
        think, answer = resolve_trace(raw, reasoning)
        return {
            "uid": q["uid"],
            "prompt_hash": prompt_hash(prompt),
            "raw": raw,
            "think": think,
            "answer": answer,
            "finish_reason": choice.finish_reason or "",
        }

    fresh = (
        map_threaded(generate, len(todo), max_workers=int(gen.parallel), desc=f"mmlu {arm['name']}")
        if todo
        else []
    )

    by_uid = {q["uid"]: q for q in questions}
    graded = [_grade(r, by_uid[r["uid"]]) for r in fresh]
    graded += [_grade(r, by_uid[uid]) for uid, r in cached.items()]
    graded.sort(key=lambda r: r["uid"])

    with records_file.open("w", encoding="utf-8") as fh:
        for rec in graded:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    scores = score_records(graded)
    scores |= {
        "arm": arm["name"],
        "served_model": arm["served"],
        "adapter": arm["adapter"],
        "synthetic_fraction": arm["synthetic_fraction"],
        "role": arm["role"],
        "mode": mode,
    }

    run_dir = arm_dir / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(scores, indent=2))

    # The cheapest defence against a chat-template mismatch, which reads as catastrophic
    # capability loss but is purely a serving bug. Look at these before believing a
    # surprising number.
    dump = graded[: int(gen.raw_sample_dump)]
    (run_dir / "raw_samples.md").write_text(
        f"# MMLU raw generations — {arm['name']} (served as `{arm['served']}`, {mode})\n\n"
        "Eyeball these before trusting the accuracy number. Look for: special tokens or\n"
        "role markers leaking into the text, the model continuing the few-shot pattern\n"
        "instead of answering, empty or unterminated `<think>` blocks, and generations\n"
        "cut off mid-trace (`finish_reason=length`).\n\n"
        + "\n\n---\n\n".join(
            f"## {r['uid']} — gold `{r['answer_letter']}`, parsed `{r['parsed']}` "
            f"(tier `{r['parse_tier']}`), finish_reason `{r['finish_reason']}`\n\n"
            f"**Think ({r['think_words']} words)**\n\n{r['think'][:1200]}\n\n"
            f"**Answer**\n\n{r['answer'][:1200]}"
            for r in dump
        )
    )
    write_run_meta(
        run_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={
            "arm": arm["name"],
            "served_model": arm["served"],
            "adapter": arm["adapter"],
            "endpoint": endpoint,
            "subset_hash": subset_hash(questions),
            "n_questions": len(questions),
            "records_file": str(records_file),
        },
    )

    print(
        f"    accuracy {scores['mean']:.1%} "
        f"[{scores['ci_lower']:.1%}, {scores['ci_upper']:.1%}]  "
        f"parse {scores['parse_rate']:.1%}  trunc {scores['truncation_rate']:.1%}  "
        f"think {scores['mean_think_words']:.0f}w"
    )
    return scores


def main(
    config: str = "configs/eval/mmlu.yaml",
    arms: str = "all",
    endpoint: str = "",
    per_subject: int = 0,
    parallel: int = 0,
    nothink: bool = False,
    smoke: bool = False,
) -> None:
    """Evaluate the MMLU subset against one or more served arms.

    Args:
        config: Path to the MMLU eval config.
        arms: `all` for every trained arm in the ladder, or a comma-separated list of
            arm names.
        endpoint: Override the serving base URL (e.g. a RunPod proxy). Decoding params
            still come from the config, so parity cannot be broken by this flag.
        per_subject: Override the subset size per subject; 0 uses the config value.
        parallel: Override concurrent requests; 0 uses the config value. Turn this DOWN
            when another job shares the endpoint. Observed on a live pod: this eval at
            16 workers alongside an Arena-Hard sweep at 16, over four different LoRA
            adapters, made vLLM's adapter scheduling thrash — arm names started
            returning 404 and whole arms came back as 0.0% accuracy at 0% parse rate.
            That is a serving artefact, not capability collapse, but it costs a re-run.
        nothink: Evaluate with thinking disabled. Results land under a separate
            `nothink/` tree — never compare a nothink arm against a thinking baseline
            (CLAUDE.md gotcha 5).
        smoke: 2 questions per subject, to validate wiring before spending GPU time.
    """
    cfg = OmegaConf.load(config)
    if nothink:
        cfg.generation.enable_thinking = False
    if per_subject:
        cfg.subset.per_subject = per_subject
    if parallel:
        cfg.generation.parallel = parallel
    if smoke:
        cfg.subset.per_subject = 2

    ladder = resolve_arms(cfg)
    known = {a["name"]: a for a in ladder}

    if arms == "all":
        selected = [a for a in ladder if a["trained"]]
        skipped = [a["name"] for a in ladder if not a["trained"]]
        if skipped:
            # Loud, not silent: a half-trained ladder must not read as a full result.
            print(f"!!! SKIPPING untrained arms (no adapter in {cfg.arms_from}): {skipped}")
    else:
        selected = []
        for name in [a.strip() for a in arms.split(",") if a.strip()]:
            if name not in known:
                raise SystemExit(f"Unknown arm {name!r}. Known: {', '.join(known)}")
            if not known[name]["trained"]:
                raise SystemExit(
                    f"Arm {name!r} has no adapter in {cfg.arms_from}: that checkpoint has "
                    f"not been trained yet. Train it, publish it, set `adapter:`, re-run."
                )
            selected.append(known[name])
    if not selected:
        raise SystemExit("No arms to evaluate.")

    rows = load_split(str(cfg.subset.split), str(cfg.subset.dataset), str(cfg.subset.name))
    raw_subjects = cfg.subset.get("subjects")
    subjects = OmegaConf.to_container(raw_subjects, resolve=True) if raw_subjects else None
    questions = build_subset(
        rows,
        per_subject=int(cfg.subset.per_subject),
        seed=int(cfg.seed),
        subjects=subjects,
        shuffle_choices=bool(cfg.subset.shuffle_choices),
    )
    shots = _shots_by_subject(cfg, int(cfg.seed))
    url = endpoint or str(cfg.generation.endpoint)

    print(f">>> subset:   {len(questions)} questions, "
          f"{len({q['subject'] for q in questions})} subjects, hash {subset_hash(questions)}")
    print(f">>> prompt:   {int(cfg.prompt.n_shot)}-shot, cue {str(cfg.prompt.cue)!r}")
    print(f">>> endpoint: {url}")
    print(f">>> mode:     {'thinking' if cfg.generation.enable_thinking else 'nothink'}, "
          f"temp={cfg.generation.temperature}, max_tokens={cfg.generation.max_tokens}")
    print(f">>> arms:     {', '.join(a['name'] for a in selected)}")

    out_root = Path(str(cfg.output_dir))
    results = [run_arm(arm, questions, shots, cfg, url, out_root) for arm in selected]

    print("\n=== summary ===")
    for r in results:
        print(
            f"  {r['arm']:18} acc {r['mean']:6.1%} "
            f"[{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]  n={r['n']}  "
            f"parse {r['parse_rate']:.1%}  trunc {r['truncation_rate']:.1%}"
        )
    print("\nBuild the comparison report with:")
    print(f"  uv run python src/eval/capabilities/mmlu_report.py --config {config}"
          + (" --nothink" if nothink else ""))


if __name__ == "__main__":
    fire.Fire(main)
