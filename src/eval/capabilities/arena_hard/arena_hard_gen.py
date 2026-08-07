# ABOUTME: Generate Arena-Hard answers for one arena-hard-eval arm from a served vLLM
# ABOUTME: endpoint. Run: uv run python src/eval/capabilities/arena_hard_gen.py --arm arm_a_synth00

"""Candidate answer generation for the capability regression eval.

Writes arena-hard-format answer files that the vendored `gen_judgment.py` consumes, plus
the instrumentation the vendored harness does not produce: degeneracy counters, style
drift, `<think>`-block health, and a raw-generation dump for the chat-template check.

Why this is ours rather than the vendored `gen_answer.py`:

- **Thinking models.** Qwen3 emits a `<think>` block. It must be stripped before judging
  (the judge should score the answer, not the scratchpad) but retained for measurement,
  because a collapsed `<think>` is CLAUDE.md gotcha 2 and invalidates everything
  downstream. Upstream has no concept of this.
- **Degeneracy counters.** Spec §8 wants truncation, repetition, refusal and length-shape
  instrumentation over exactly these generations. Capturing `finish_reason` at generation
  time is the only place truncation is observable.
- **Decoding parity.** Spec §4 requires provably identical decoding across arms. Reading
  it from one config field and stamping it into `run_meta.json` per arm makes that
  auditable instead of assumed.

Answers are written to the vendored tree (where the judge looks) *and* mirrored under
`output/`. The tree is TRACKED in git (patched + pruned; see its VENDORED_FROM.txt).

Run one arm at a time, against whatever that arm's adapter is currently served as:

    uv run python src/eval/capabilities/arena_hard_gen.py \
        --config configs/eval/arena_hard.yaml --arm arm_a_synth00 --served_model arm_a_synth00
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time

import tiktoken
from pathlib import Path
from typing import Any

import fire
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.eval.capabilities.arena_hard.arena_hard_metrics import (  # noqa: E402
    degeneracy_metrics,
    pattern_frequencies,
    style_features,
)
from src.endpoints.openrouter import map_threaded  # noqa: E402
from src.model_profile import split_think
from src.utils import read_jsonl, timestamp, write_run_meta  # noqa: E402


def _arm(cfg: DictConfig, name: str) -> DictConfig:
    """Look up an arm by name, failing loudly if it is not in the config."""
    for arm in cfg.arms:
        if arm.name == name:
            return arm
    known = ", ".join(a.name for a in cfg.arms)
    raise SystemExit(f"Unknown arm {name!r}. Known arms: {known}")


def _select_questions(
    questions: list[dict],
    arm: DictConfig,
    stage: int | None = None,
    creative: int | None = None,
) -> list[dict]:
    """Take each category's first-N prefix for this arm.

    The prefix (rather than a random sample) is deliberate: it is the same prefix the
    staged-sampling limits in `gen_judgment.py` use, so stage 1's 150 questions are a
    strict subset of stage 2's 300 and the cache actually hits.

    Args:
        questions: All benchmark questions, in file order.
        arm: The arm config, carrying `n_hard_prompt` / `n_creative_writing`.
        stage: Optional override for the `hard_prompt` count, so a stage can be run
            without editing the arm's canonical target in the config.
        creative: Optional override for the `creative_writing` count.

    Returns:
        The selected questions, in file order.
    """
    limits = {
        "hard_prompt": min(int(stage or arm.n_hard_prompt), int(arm.n_hard_prompt)),
        "creative_writing": int(
            arm.n_creative_writing if creative is None else creative
        ),
    }
    seen: dict[str, int] = {}
    kept = []
    for q in questions:
        cat = q["category"]
        used = seen.get(cat, 0)
        if used >= limits.get(cat, 0):
            continue
        seen[cat] = used + 1
        kept.append(q)
    return kept


def main(
    config: str = "configs/eval/arena_hard.yaml",
    arm: str = "",
    served_model: str = "",
    api_key: str = "",
    endpoint: str = "",
    stage: int = 0,
    creative: int = -1,
    smoke: bool = False,
) -> None:
    """Generate one arm's answers over the Arena-Hard question set.

    Args:
        config: Path to the arena-hard-eval config.
        arm: Arm name from the config (e.g. `arm_a_synth00`).
        served_model: Model name as served by vLLM. Defaults to `arm`.
        endpoint: Override the serving base URL (e.g. a RunPod proxy URL) without
            editing the config. Decoding params are still read from the config, so
            parity across arms cannot be broken by this flag.
        stage: Override the `hard_prompt` count for this run; 0 uses the arm's target.
        creative: Override the `creative_writing` count; -1 uses the arm's target.
        smoke: If True, generate 4 questions per category to validate wiring.
    """
    if not arm:
        raise SystemExit("--arm is required (e.g. --arm arm_a_synth00)")

    cfg = OmegaConf.load(config)
    arm_cfg = _arm(cfg, arm)
    served = served_model or arm

    # An arm with no adapter and no floor role is a checkpoint that does not exist yet.
    # Fail here rather than producing an answer file that silently holds the wrong
    # model's generations — a half-trained ladder must not look like a full result.
    if arm_cfg.adapter is None and arm_cfg.role != "floor":
        raise SystemExit(
            f"Arm {arm!r} has no adapter in {config}: that checkpoint has not been "
            f"trained yet. Train it, publish it, set `adapter:`, then re-run."
        )

    vendor = Path(cfg.vendor_dir)
    question_file = vendor / "data" / cfg.bench_name / "question.jsonl"
    if not question_file.exists():
        raise SystemExit(
            f"No question set at {question_file} — the vendored tree is TRACKED in "
            f"git (patched, pruned). Restore it: git checkout -- {vendor}"
        )

    questions = read_jsonl(question_file)
    selected = _select_questions(
        questions, arm_cfg, stage or None, None if creative < 0 else creative
    )
    if smoke:
        by_cat: dict[str, int] = {}
        smoked = []
        for q in selected:
            n = by_cat.get(q["category"], 0)
            if n >= 4:
                continue
            by_cat[q["category"]] = n + 1
            smoked.append(q)
        selected = smoked

    answer_file = vendor / "data" / cfg.bench_name / "model_answer" / f"{served}.jsonl"
    answer_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip uids already generated. Same contract as the judgment cache, so an
    # interrupted run costs only what it had not yet finished.
    existing: dict[str, dict] = {}
    if answer_file.exists():
        for rec in read_jsonl(answer_file):
            existing[rec["uid"]] = rec
    todo = [q for q in selected if q["uid"] not in existing]

    gen = cfg.generation
    print(f">>> arm:          {arm}  (served as {served!r})")
    print(f">>> adapter:      {arm_cfg.adapter}")
    print(f">>> questions:    {len(selected)} selected, {len(todo)} to generate")
    print(f">>> endpoint:     {endpoint or gen.endpoint}")
    print(f">>> decoding:     temp={gen.temperature} top_p={gen.top_p} max_tokens={gen.max_tokens}")

    base_url = endpoint or str(gen.endpoint)
    client = OpenAI(base_url=base_url, api_key=api_key or str(gen.api_key))

    # The output budget must be sized PER PROMPT against the server's context window.
    # A fixed max_tokens large enough for a hard reasoning prompt (the <think> trace is
    # generated inside this budget) will exceed the window on long prompts, and vLLM
    # rejects the whole request with a 400 rather than clamping it. Because map_threaded
    # is fail-fast, a single such prompt aborts the entire arm — which cost a full 62
    # minute run of 150 answers. Ask the server for its real limit instead of assuming.
    context_limit = 8192
    try:
        for entry in client.models.list().data:
            if getattr(entry, "max_model_len", None):
                context_limit = int(entry.max_model_len)
                break
    except Exception as exc:  # noqa: BLE001 - fall back to the conservative default
        print(f">>> could not read max_model_len ({type(exc).__name__}); assuming {context_limit}")

    _enc = tiktoken.encoding_for_model("gpt-4o")

    def output_budget(prompt: str) -> int:
        """Largest safe completion budget for this prompt.

        The gpt-4o tokenizer only approximates Qwen's, so a margin absorbs the
        disagreement; without it a prompt that tokenizes longer than estimated still
        400s.
        """
        approx_prompt = len(_enc.encode(prompt, disallowed_special=()))
        # A fixed 512 margin is not enough: on one arena-hard prompt the gpt-4o estimate
        # undercounted Qwen's tokenizer by 26% (1,999 est vs 2,512 actual) and the request
        # exceeded the window by 1 token — a deterministic 400 on every arm. Scale the
        # margin with prompt length so the divergence cannot outgrow it.
        margin = 512 + approx_prompt // 3
        return max(512, min(int(gen.max_tokens), context_limit - approx_prompt - margin))

    def generate(i: int) -> dict:
        q = todo[i]
        # STREAMING IS REQUIRED, not a preference. Serving through RunPod's HTTPS proxy
        # puts Cloudflare in the path, which enforces a 120s read timeout: a
        # non-streaming request that spends four minutes generating a long <think> block
        # sends zero bytes until it finishes and is killed with a 524 mid-run. Streaming
        # resets that timer on every token, so a slow generation is fine and only a
        # genuinely hung one fails. Token-for-token identical output either way.
        stream = client.chat.completions.create(
            model=served,
            messages=[{"role": "user", "content": q["prompt"]}],
            temperature=float(gen.temperature),
            top_p=float(gen.top_p),
            max_tokens=output_budget(q["prompt"]),
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(gen.enable_thinking)}},
        )
        parts: list[str] = []
        reasoning_parts: list[str] = []
        finish = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    parts.append(delta.content)
                # Some vLLM builds stream the trace out-of-band instead of inline. The
                # field name is NOT stable: vLLM 0.26 emits `reasoning`, other builds and
                # OpenRouter emit `reasoning_content`. Checking only one silently reports
                # every trace as empty, which is indistinguishable from CLAUDE.md gotcha 2
                # (the empty-<think> collapse) and would have us "discover" that training
                # destroyed the model's reasoning when it is reasoning fine.
                extra = getattr(delta, "model_extra", None) or {}
                trace = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or extra.get("reasoning_content")
                    or extra.get("reasoning")
                )
                if trace:
                    reasoning_parts.append(trace)
            if choice.finish_reason:
                finish = choice.finish_reason
        raw = "".join(parts)
        reasoning = "".join(reasoning_parts)
        think, answer = split_think(raw)
        if reasoning and not think:
            think = reasoning.strip()
        return {
            "uid": q["uid"],
            "category": q["category"],
            "prompt": q["prompt"],
            "raw": raw,
            "think": think,
            "answer": answer,
            "finish_reason": finish,
        }

    # Checkpoint every answer to disk as it lands, rather than holding 150 in memory and
    # writing once at the end. A 40-minute arm that dies at minute 39 previously lost
    # everything; now a rerun re-reads what completed and pays only for the remainder.
    # The file is rewritten in sorted, de-duplicated form below, so partial lines here
    # cost nothing.
    checkpoint = out_partial = answer_file.with_suffix(".partial.jsonl")
    lock = threading.Lock()

    def generate_and_checkpoint(i: int) -> dict:
        rec = generate(i)
        with lock:
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # Resume from a previous interrupted run of this same arm.
    resumed: dict[str, dict] = {}
    if out_partial.exists():
        for rec in read_jsonl(out_partial):
            resumed[rec["uid"]] = rec
        if resumed:
            print(f">>> resuming: {len(resumed)} answers recovered from a prior run")
    todo = [q for q in todo if q["uid"] not in resumed]

    generated = (
        map_threaded(
            generate_and_checkpoint,
            len(todo),
            max_workers=int(gen.parallel),
            desc=f"gen {arm}",
        )
        if todo
        else []
    )
    generated = list(resumed.values()) + generated

    # Arena-hard's answer schema. `messages[-1]["content"]["answer"]` is the field both
    # the judge and the style-control regression read, and it holds the think-stripped
    # answer: the judge must score the response, not the scratchpad.
    with answer_file.open("a", encoding="utf-8") as fh:
        for rec in generated:
            visible = rec["answer"] if gen.strip_think_for_judging else rec["raw"]
            ans_id = hashlib.sha256(f"{served}:{rec['uid']}".encode()).hexdigest()[:22]
            fh.write(
                json.dumps(
                    {
                        "uid": rec["uid"],
                        "ans_id": ans_id,
                        "model": served,
                        "messages": [
                            {"role": "user", "content": rec["prompt"]},
                            {"role": "assistant", "content": {"answer": visible}},
                        ],
                        "tstamp": time.time(),
                        "metadata": style_features(visible),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Sort and de-duplicate, matching the vendored `reorg_answer_file` contract.
    rows = {
        rec["uid"]: json.dumps(rec, ensure_ascii=False) for rec in read_jsonl(answer_file)
    }
    answer_file.write_text("\n".join(rows[k] for k in sorted(rows)) + "\n")

    # --- Instrumentation ------------------------------------------------------------
    out_dir = Path(cfg.output_dir) / arm / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = generated + [
        {
            "uid": uid,
            "category": next(q["category"] for q in selected if q["uid"] == uid),
            "prompt": rec["messages"][0]["content"],
            "raw": rec["messages"][-1]["content"]["answer"],
            "think": "",
            "answer": rec["messages"][-1]["content"]["answer"],
            "finish_reason": "",
        }
        for uid, rec in existing.items()
        if uid in {q["uid"] for q in selected}
    ]

    metrics: dict[str, Any] = {
        "arm": arm,
        "served_model": served,
        "adapter": arm_cfg.adapter,
        "base_model": str(cfg.base_model),
        "synthetic_fraction": arm_cfg.synthetic_fraction,
        "n_generated_this_run": len(generated),
        "decoding": {
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
            "max_tokens": int(gen.max_tokens),
            "enable_thinking": bool(gen.enable_thinking),
        },
        "by_slice": {},
    }
    percentiles = [int(p) for p in cfg.degeneracy.length_distribution_percentiles]
    patterns = {k: str(v) for k, v in cfg.style_drift.corpus_patterns.items()}

    for category in ("hard_prompt", "creative_writing"):
        subset = [r for r in all_records if r["category"] == category]
        if not subset:
            continue
        answers = [r["answer"] for r in subset]
        feats = [style_features(a) for a in answers]
        metrics["by_slice"][category] = {
            "degeneracy": degeneracy_metrics(subset, percentiles),
            "style": {
                "mean_token_len": sum(f["token_len"] for f in feats) / len(feats),
                "mean_header_count": sum(sum(f["header_count"].values()) for f in feats) / len(feats),
                "mean_list_count": sum(sum(f["list_count"].values()) for f in feats) / len(feats),
                "mean_bold_count": sum(sum(f["bold_count"].values()) for f in feats) / len(feats),
            },
            "corpus_patterns": pattern_frequencies(answers, patterns),
        }

    (out_dir / "gen_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Footgun §10.1: the cheapest check for a chat-template mismatch is to look at ten
    # raw generations. A mismatch reads as catastrophic capability loss but is purely a
    # serving bug, so this must be eyeballed BEFORE anything is judged.
    dump = generated[: int(gen.raw_sample_dump)] or all_records[: int(gen.raw_sample_dump)]
    (out_dir / "raw_samples.md").write_text(
        f"# Raw generations — {arm} (served as `{served}`)\n\n"
        "Eyeball these before judging. Look for: role markers or special tokens leaking\n"
        "into the text, answers that continue the prompt instead of responding, empty or\n"
        "unterminated `<think>` blocks, and run-on or abruptly cut generations.\n\n"
        + "\n\n---\n\n".join(
            f"## {r['uid']} ({r['category']}) — finish_reason=`{r['finish_reason']}`\n\n"
            f"**Prompt**\n\n{r['prompt'][:800]}\n\n"
            f"**Think ({len(r['think'].split())} words)**\n\n{r['think'][:800]}\n\n"
            f"**Answer**\n\n{r['answer'][:2000]}"
            for r in dump
        )
    )

    # The durable, sorted answer file now exists, so the checkpoint has served its
    # purpose. Leaving it would make a later rerun resume from a stale partial written
    # under different decoding settings.
    out_partial.unlink(missing_ok=True)

    write_run_meta(
        out_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"arm": arm, "served_model": served, "answer_file": str(answer_file)},
    )

    print(f"\n>>> answers:  {answer_file} ({len(rows)} total)")
    print(f">>> metrics:  {out_dir / 'gen_metrics.json'}")
    print(f">>> EYEBALL:  {out_dir / 'raw_samples.md'}  <- do this before judging")
    for category, block in metrics["by_slice"].items():
        deg = block["degeneracy"]
        print(
            f"    {category:18} n={deg['n']:<4} trunc={deg['truncation_rate']:.1%} "
            f"refusal={deg['refusal_rate_benign']:.1%} rep={deg['repetition_rate']:.1%} "
            f"len={deg['mean_output_words']:.0f}w think={deg['mean_think_words']:.0f}w"
        )


if __name__ == "__main__":
    fire.Fire(main)
