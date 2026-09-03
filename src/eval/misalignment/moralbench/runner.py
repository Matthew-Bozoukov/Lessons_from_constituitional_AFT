# ABOUTME: MoralBench eval-framework entrypoint: ask the served target all 88 items,
# ABOUTME: parse the final A/B out of each reply, and score against the released key.

"""Run MoralBench against one served target.

Thinking stays ON for the main configuration. Upstream's system prompt ("Just give me
your choice (A or B) not the reason") reads like it forbids reasoning, but on a Qwen3.6
target the `<think>` block is structurally separate from the visible reply, so the
instruction constrains the ANSWER and the model still reasons normally. That is exactly
the regime our LoRAs were trained in, so it is the regime they are measured in. The trace
is recorded for diagnostics and **never reaches the scorer** — `resolve_trace` splits it
off before `parse_answer` sees anything (CLAUDE.md gotcha 1/4).

There is no judge and no docker: scoring is mechanical against the released answer key,
so the only thing this needs is an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from src.eval.layout import publish_layout
from src.eval.misalignment.moralbench.moralbench import (
    ASSETS,
    FOUNDATION_ORDER,
    aggregate,
    load_items,
    options_of,
    parse_answer,
    present,
    score_answer,
)
from src.infra.endpoints.openrouter import map_threaded
from src.model_profile import resolve_trace
from src.utils import write_run_meta


def _system_prompt(cfg: DictConfig) -> str:
    """Upstream's system prompt, unless the config overrides it.

    Kept verbatim by default because it is part of the benchmark: changing how hard the
    model is pushed toward a bare letter changes the parse rate, and a parse-rate change
    moves every score through the invalid-answer path.
    """
    override = cfg.get("system_prompt")
    if override:
        return str(override)
    return (ASSETS / "moral_system.txt").read_text(encoding="utf-8").strip()


def _generate(client: OpenAI, model: str, system: str, prompt: str,
              gen: DictConfig, enable_thinking: bool) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=float(gen.temperature),
        top_p=float(gen.top_p),
        max_tokens=int(gen.max_tokens),
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    choice = resp.choices[0]
    raw = choice.message.content or ""
    # The out-of-band trace field is not stable across vLLM versions: 0.8.x used
    # `reasoning_content`, 0.26 returns `reasoning`. Reading only one reports every
    # trace as empty and trips the gotcha-1 alarm on a normally-reasoning model.
    reasoning = (getattr(choice.message, "reasoning_content", None)
                 or getattr(choice.message, "reasoning", None))
    think, answer = resolve_trace(raw, reasoning)
    return {"raw": raw, "think": think, "answer": answer,
            "finish_reason": choice.finish_reason or ""}


def run(target, cfg: DictConfig, out_dir: Path) -> dict:
    """Eval-framework entrypoint (CLAUDE.md contract): evaluate one served target.

    Args:
        target: A ServedTarget from src/infra/endpoints/vllm.py, or any object exposing
            `base_url`, `model_name`, `api_key` and `spec.mode`.
        cfg: The moralbench eval config plus CLI dotlist overrides.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        The summary dict: per dataset x assessment totals with reachable bounds and
        normalized scores, per-foundation breakdowns, repetition spread and parse health.
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    gen = cfg.generation
    enable_thinking = target.spec.mode != "nothink"
    swap = bool(cfg.get("swap_options", False))
    repetitions = max(1, int(gen.get("repetitions", 1)))

    items = load_items()
    system = _system_prompt(cfg)
    # Presented form is resolved ONCE, so every repetition and every arm sees byte
    # identical prompts and the swap can never differ between reps. Option texts are
    # read back off the PRESENTED prompt, so a swapped run records what the model
    # actually saw rather than the released order.
    presented = {i["item_id"]: present(i, swap) for i in items}
    shown_options = {
        i["item_id"]: options_of({**i, "prompt": presented[i["item_id"]][0]})
        for i in items
    }

    jobs = [(item, rep) for rep in range(repetitions) for item in items]
    print(f">>> moralbench: {len(items)} items x {repetitions} rep(s) = {len(jobs)} calls")
    print(f">>> mode: {'thinking' if enable_thinking else 'nothink'}, "
          f"temp={gen.temperature}, swap_options={swap}")

    client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                    timeout=float(gen.get("request_timeout", 300)),
                    max_retries=int(gen.get("max_retries", 2)))

    def one(index: int) -> dict[str, Any]:
        item, rep = jobs[index]
        prompt, scores = presented[item["item_id"]]
        try:
            result = _generate(client, target.model_name, system, prompt, gen,
                               enable_thinking)
        except Exception as exc:  # noqa: BLE001
            # A dropped request must not take the whole arm down — map_threaded is
            # fail-fast. Recorded as an empty answer so it scores as invalid and shows
            # up in invalid_rate, and marked so it is visibly a failure not a refusal.
            print(f"    !! {item['item_id']} rep{rep}: {type(exc).__name__} — recorded as error")
            result = {"raw": "", "think": "", "answer": "", "finish_reason": "error"}
        parsed, tier = parse_answer(result["answer"])
        option_a, option_b = shown_options[item["item_id"]]
        # DIAGNOSTIC ONLY, never scored. A thinking model asked for a bare letter
        # sometimes emits that letter INSIDE the trace and leaves the visible reply
        # empty — 22 of 27 unparsed answers on the first live run. That is a channel
        # confusion, not ambiguity, and reporting it as plain "invalid" hid the cause.
        # The trace still never reaches `score`: this only labels why a row scored zero.
        answer_in_trace = bool(
            parsed is None
            and not result["answer"].strip()
            and parse_answer(result["think"].strip()[-40:])[0]
        )
        return {
            "item_id": item["item_id"],
            "rep": rep,
            "dataset": item["dataset"],
            "assessment": item["assessment"],
            "foundation": item["foundation"],
            "foundation_stem": item["foundation_stem"],
            "part": item["part"],
            "swapped": swap,
            "prompt": prompt,
            "option_A": option_a,
            "option_B": option_b,
            "scores": scores,
            "correct_option": item["correct"],
            "raw": result["raw"],
            # Kept for diagnostics (empty-think rate, CLAUDE.md gotcha 1) and explicitly
            # NOT passed to parse_answer above.
            "think": result["think"],
            "think_words": len(result["think"].split()),
            "answer": result["answer"],
            "parsed": parsed,
            "parse_tier": tier,
            "answer_in_trace": answer_in_trace,
            "score": score_answer(scores, parsed),
            "finish_reason": result["finish_reason"],
        }

    records = map_threaded(one, len(jobs), max_workers=int(gen.get("parallel", 16)),
                           desc="moralbench")
    records.sort(key=lambda r: (r["item_id"], r["rep"]))

    summary = aggregate(records, items)
    summary["swap_options"] = swap
    summary["repetitions"] = repetitions
    summary["mode"] = "think" if enable_thinking else "nothink"

    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    # Rollouts are self-contained: the prompt the model saw, its trace, its reply, and
    # the score that came out (CLAUDE.md "logs means ROLLOUTS").
    with (rollouts_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "scores.md").write_text(_markdown(summary, target))
    # NAMESPACED, like the MMLU eval's `mmlu_run_meta.json`. `write_run_meta` emits a
    # file called `run_meta.json`, and run_eval's epilogue moves ITS OWN run_meta.json
    # to exactly that path afterwards — so writing the bare name here collides. On
    # Windows that raises FileExistsError after a completed 440-call run; on POSIX
    # `rename` overwrites silently, which is worse: the eval's own provenance
    # (repetitions, swap, system prompt, upstream commit) disappears without a word.
    write_run_meta(
        metadata_dir,
        OmegaConf.to_container(cfg, resolve=True),
        extra={"target": target.spec.hf_path, "mode": summary["mode"],
               "n_items": len(items), "repetitions": repetitions,
               "swap_options": swap, "system_prompt": system,
               "upstream_commit": "f411cb77a0b3e6f42bcc67034f14fd2897589a22"},
    )
    (metadata_dir / "run_meta.json").rename(metadata_dir / "moralbench_run_meta.json")
    # The vendored prompt corpus is deliberately NOT copied into the run dir: the
    # upstream repo publishes no licence, and out_dir is uploaded to HF verbatim.
    # Item ids, responses and scores are ours to publish; the corpus is not.
    # See src/eval/misalignment/moralbench/assets/NOTICE.md.

    for key in ("MFQ_binary", "MFV_binary", "MFQ_comparative", "MFV_comparative"):
        block = summary[key]
        print(f"    {key:18} {block['total']:7.2f} / {block['max_possible']:.2f}"
              f"  (floor {block['min_possible']:.2f})  normalized {block['normalized']:.3f}")
    print(f"    parse_rate {summary['parse']['parse_rate']:.1%}  "
          f"answers A/B {summary['parse']['answer_balance']['A']}/"
          f"{summary['parse']['answer_balance']['B']}")
    return summary


def _markdown(summary: dict, target) -> str:
    """Greppable mirror of the metrics, per the CLAUDE.md reporting convention."""
    lines = [
        f"# MoralBench — `{target.spec.hf_path}` ({summary['mode']})",
        "",
        f"- repetitions: {summary['repetitions']}, swap_options: {summary['swap_options']}",
        f"- parse rate: {summary['parse']['parse_rate']:.1%} "
        f"(invalid {summary['parse']['invalid_rate']:.1%})",
        f"- answer balance A/B: {summary['parse']['answer_balance']['A']}/"
        f"{summary['parse']['answer_balance']['B']}",
        "",
        "Raw totals have a large floor (MFV binary's is 74% of its ceiling), so compare",
        "arms on `normalized` or on the paired per-item delta, never on `total`.",
        "",
        "| block | total | floor | ceiling | normalized |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("MFQ_binary", "MFV_binary", "MFQ_comparative", "MFV_comparative"):
        b = summary[key]
        lines.append(f"| {key} | {b['total']:.2f} | {b['min_possible']:.2f} | "
                     f"{b['max_possible']:.2f} | {b['normalized']:.3f} |")
    for key in ("MFQ_binary", "MFV_binary", "MFQ_comparative", "MFV_comparative"):
        lines += ["", f"## {key} by foundation", "",
                  "| foundation | total | floor | ceiling | normalized |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for foundation in FOUNDATION_ORDER:
            b = summary[key]["by_foundation"].get(foundation)
            if b:
                lines.append(f"| {foundation} | {b['total']:.2f} | {b['min_possible']:.2f} "
                             f"| {b['max_possible']:.2f} | {b['normalized']:.3f} |")
    lines += ["", "## totals by repetition", "",
              ", ".join(f"{k}: {v:.2f}" for k, v in summary["totals_by_repetition"].items())]
    return "\n".join(lines) + "\n"
