# ABOUTME: MoReBench runner — generate one reasoned response per dilemma on the served target,
# ABOUTME: judge every rubric criterion on OpenRouter, score with the paper's own arithmetic.
"""MoReBench (arXiv:2510.16380) against one served checkpoint.

    uv run evals --name morebench --target <hf> --server root@<ip>:<port>

The paper's pipeline in three stages, reimplemented against this repo's eval contract:

  1. generate  one response per dilemma from the target, in its thinking mode
  2. judge     one call per (response x criterion) -- 11,450 for the public 500-dilemma
               set -- asking the paper's yes/no question, on the paper's judge
  3. score     share of criterion weight earned (scoring.py), overall and per category

Upstream's driver scripts are not vendored: their clients are hardcoded to five hosted APIs,
so a checkpoint we serve ourselves could not be a target at all. The two prompt strings ARE
vendored verbatim (third_party/prompts.py) because they are the measurement.

Judging dominates: stage 2 is ~23x the API calls of stage 1 and all of the dollar cost, so
the judge model is pinned in config and its spend is reported in the summary. Changing it
makes a run incomparable to the paper's numbers, exactly as with MASK.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

from src.eval.layout import publish_layout
from src.eval.misalignment.morebench.scoring import summarize
from src.eval.misalignment.morebench.source import load_dilemmas, smoke_subset
from src.eval.misalignment.morebench.third_party.prompts import JUDGE_PROMPT, REASONING_PROMPT
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded


def await_model(client, model_name: str, timeout_s: float = 300.0) -> None:
    """Block until the server lists `model_name`, not merely until it is healthy.

    VllmServer waits on /health, which vLLM answers before its `--lora-modules` adapters
    finish registering. This eval then fires `generation.parallel` requests at once, so the
    first of them can arrive in that window and come back 404 "model does not exist" --
    which map_threaded turns into a failed run and, with --terminate-pod, a discarded pod
    (caught 2026-09-29). Polling the model list closes the gap for the cost of one call.
    """
    deadline = time.time() + timeout_s
    seen: list[str] = []
    while time.time() < deadline:
        try:
            seen = [m.id for m in client.models.list().data]
            if model_name in seen:
                return
        except Exception:  # noqa: BLE001 — the server may not be answering yet at all
            pass
        time.sleep(5)
    raise TimeoutError(f"{model_name!r} never appeared in the served model list; saw {seen}")


def generate(client, model: str, dilemma: dict, gen) -> dict:
    """One reasoned response to one dilemma, with its thinking trace when the server emits one."""
    prompt = f"{REASONING_PROMPT}{dilemma['dilemma']}"
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=float(gen.temperature), top_p=float(gen.top_p),
        max_tokens=int(gen.max_tokens))
    message = completion.choices[0].message
    # vLLM's reasoning parser returns the trace as `reasoning_content`; hosted APIs that
    # this may one day run against call it `reasoning`. Neither is required.
    trace = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", "") or ""
    return {
        **{k: v for k, v in dilemma.items() if k != "criteria"},
        "prompt": prompt,
        "response": message.content or "",
        "thinking_trace": trace,
        "finish_reason": completion.choices[0].finish_reason,
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
    }


def run(target, cfg, out_dir: Path) -> dict:
    """Run MoReBench against one served target (CLAUDE.md contract).

    Args:
        target: The ServedTarget from run_eval (base_url, model_name, api_key, spec).
        cfg: Merged OmegaConf config (configs/eval/morebench.yaml + CLI overrides).
        out_dir: This target's output directory; all artifacts go under it.

    Returns:
        Summary metrics: overall rubric score, the ai_advisor/ai_agent split, per dilemma
        type and source, per-dimension criterion pass rates, plus generation and judge health.
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    rollout_dir, results_dir, _ = publish_layout(out_dir)
    dilemmas = load_dilemmas(cfg)
    if bool(cfg.get("smoke", False)):
        dilemmas = smoke_subset(dilemmas)

    gen = cfg.generation
    n_criteria = sum(len(d["criteria"]) for d in dilemmas)
    print(f">>> morebench: {len(dilemmas)} dilemmas | {n_criteria} criteria to judge "
          f"| target {target.model_name} | judge {cfg.judge.model}", flush=True)

    client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                    timeout=float(gen.request_timeout), max_retries=int(gen.max_retries))
    await_model(client, target.model_name)
    responses = map_threaded(
        lambda i: generate(client, target.model_name, dilemmas[i], gen),
        len(dilemmas), max_workers=int(gen.parallel), desc="morebench/generate")
    (rollout_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf-8")
    first = responses[0]
    print(f">>> first response ({first['role_domain']}, {first['dilemma_type']}, "
          f"{first['output_tokens']} tok): {first['response'][:300]!r}", flush=True)

    empty = [r for r in responses if not r["response"].strip()]
    truncated = [r for r in responses if r["finish_reason"] == "length"]
    if empty:
        print(f"!!! {len(empty)} of {len(responses)} responses are EMPTY — they score 0 and "
              f"are reported, not dropped", flush=True)

    jobs = [(r, c) for r, d in zip(responses, dilemmas) for c in d["criteria"]]
    judge = OpenRouterClient()
    jcfg = cfg.judge

    def judge_one(i: int) -> dict:
        """Ask the paper's yes/no question for one (response, criterion) pair."""
        response, criterion = jobs[i]
        prompt = (f"Reasoning Response:{response['response']}\n\n{JUDGE_PROMPT}\n\n"
                  f"Rubric Criterion:{criterion['title']}")
        result = judge.chat(model=jcfg.model, messages=[{"role": "user", "content": prompt}],
                            temperature=float(jcfg.temperature),
                            max_tokens=int(jcfg.max_tokens))
        return {
            "task_id": response["task_id"], "criterion_id": criterion["id"],
            "criterion": criterion["title"], "criterion_weight": criterion["weight"],
            "criterion_dimension": criterion["dimension"],
            "role_domain": response["role_domain"], "dilemma_type": response["dilemma_type"],
            "dilemma_source": response["dilemma_source"], "response": response["response"],
            "judgement": result.content, "judge_usd": result.cost or 0.0,
        }

    judged = map_threaded(judge_one, len(jobs), max_workers=int(jcfg.workers),
                          desc="morebench/judge")
    (rollout_dir / "judgements.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judged) + "\n", encoding="utf-8")

    unparsed = [j for j in judged
                if "yes" not in str(j["judgement"]).lower()
                and "no" not in str(j["judgement"]).lower()]
    if unparsed:
        print(f"!!! {len(unparsed)} of {len(judged)} judgements parse as NEITHER yes nor no; "
              f"they score as a miss", flush=True)

    summary = summarize(judged)
    summary.update(
        target=target.spec.hf_path, model=target.model_name, mode=target.spec.mode,
        judge_model=jcfg.model, judge_usd=round(sum(j["judge_usd"] for j in judged), 4),
        n_dilemmas=len(dilemmas), empty_responses=len(empty),
        truncated_responses=len(truncated), unparsed_judgements=len(unparsed),
        source={"repo": cfg.source.repo, "file": cfg.source.file,
                "theory": cfg.source.get("theory", "neutral")},
    )
    (results_dir / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    rows = ["| split | n | score |", "|---|---|---|"]
    rows += [f"| {k} | {v['n']} | {v['score']} |" for k, v in summary["by_role_domain"].items()]
    rows += [f"| {k} | {v['n']} | {v['score']} |" for k, v in summary["by_dilemma_type"].items()]
    (results_dir / "results.md").write_text(
        f"# MoReBench — {target.spec.hf_path}\n\n"
        f"Overall **{summary['overall']}** over {summary['n_tasks']} dilemmas "
        f"({summary['n_criteria']} criteria), judge `{jcfg.model}`, ${summary['judge_usd']}.\n\n"
        + "\n".join(rows) + "\n", encoding="utf-8")
    print(f">>> morebench overall {summary['overall']} | "
          f"advisor/agent {summary['by_role_domain']} | judge ${summary['judge_usd']}", flush=True)
    return summary
