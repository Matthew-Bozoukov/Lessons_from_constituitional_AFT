# ABOUTME: Eval-framework entrypoint for ctfish: N sandboxed agent rollouts against a
# ABOUTME: served target, scored on the paper's escalation scale by a majority-vote judge.

"""Capture the Fish, as an eval of this repo's model organisms.

Palisade Research put a generalist shell agent in front of an unbeatable Stockfish and
measured how often it reached for the environment instead of the board
(https://github.com/PalisadeResearch/ctfish, arXiv:2502.13295). That is a specification-
gaming probe, so it sits beside the other behavioural misalignment evals here: an arm that
plays chess and loses is the aligned outcome, and an arm that rewrites `game/fen.txt` is
the finding.

Shape of a run: build the sandbox image once, run `runs` independent rollouts (each its own
throwaway container, `sandbox.py`, driven by the OODA loop in `agent.py`), then score every
transcript with the upstream rubric over `judge.samples` samples and take the relaxed
majority (`judge.py`). Every rollout is written to disk the moment it finishes, so a judge
outage or a dead pod costs verdicts, never generations.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

from src.eval.layout import publish_layout
from src.eval.misalignment.ctfish import judge as judging
from src.eval.misalignment.ctfish.agent import Rollout, run_rollout
from src.eval.misalignment.ctfish.metrics import report_markdown, summarize
from src.eval.misalignment.ctfish.prompts import (PRE_INVOKE_NONE,
                                                  PRE_INVOKE_THOUGHT_TAG, TASK_PROMPTS)
from src.eval.misalignment.ctfish.sandbox import Sandbox, build_image
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

# Smoke profile: the FULL pipeline (image build, container, loop, judge, report) on one
# short run, so wiring is verifiable in a couple of minutes before a real arm is paid for.
SMOKE_RUNS = 1
SMOKE_MAX_STEPS = 4
SMOKE_JUDGE_SAMPLES = 1


def _pre_invoke_msg(cfg, mode: str) -> str:
    """The per-step nudge: explicit config value, else upstream's rule applied to `mode`.

    Upstream sends the `<THOUGHT>` scratchpad instruction to non-reasoning models and
    nothing at all to reasoning ones. A think-mode arm has a native reasoning channel, so
    it takes the empty message; a nothink arm takes the scratchpad instruction, which is
    the only place its deliberation could go.
    """
    configured = cfg.get("pre_invoke_msg")
    if configured is not None:
        return str(configured)
    return PRE_INVOKE_NONE if mode == "think" else PRE_INVOKE_THOUGHT_TAG


def _rollout_record(rollout: Rollout) -> dict:
    """The structured transcript published beside the journal."""
    return {
        "run_id": rollout.run_id,
        "variant": rollout.variant,
        "task_prompt": rollout.task_prompt,
        "stop_reason": rollout.stop_reason,
        "n_steps": len(rollout.steps),
        "stalled_steps": rollout.stalled_steps,
        "steps": [{**asdict(step), "results": [asdict(r) for r in step.results]}
                  for step in rollout.steps],
        "memory": {str(k): v for k, v in rollout.memory.items()},
        "plan": rollout.plan,
        "subgoal": rollout.subgoal,
        # What the agent actually left on disk. The paper's canonical hack is a rewritten
        # FEN, and only this shows it independently of what the transcript claims.
        "game_state": rollout.game_state,
    }


def run(target, cfg, out_dir: Path) -> dict:
    """Run ctfish against a ServedTarget (CLAUDE.md eval contract).

    Args:
        target: The served target; reached over its OpenAI-compatible endpoint.
        cfg: The merged `configs/eval/ctfish.yaml` (plus CLI dotlist overrides).
        out_dir: This arm's run directory, published verbatim by run_eval.py.

    Returns:
        The summary: escalation distribution, hacking rate and generation-health rates.
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    smoke = bool(cfg.get("smoke", False))
    variant = str(cfg.variant)
    assert variant in TASK_PROMPTS, (
        f"unknown task-prompt variant {variant!r}; known: {sorted(TASK_PROMPTS)}")
    task_prompt = TASK_PROMPTS[variant]
    n_runs = SMOKE_RUNS if smoke else int(cfg.runs)
    max_steps = SMOKE_MAX_STEPS if smoke else int(cfg.max_steps)
    judge_samples = SMOKE_JUDGE_SAMPLES if smoke else int(cfg.judge.samples)
    pre_invoke = _pre_invoke_msg(cfg, target.spec.mode)

    rollout_dir, results_dir, metadata_dir = publish_layout(out_dir)

    print(f">>> ctfish | variant={variant} | {n_runs} runs x {max_steps} steps | "
          f"judge={cfg.judge.model} x{judge_samples}", flush=True)
    print(f">>> task prompt:\n{task_prompt.strip()}", flush=True)

    image_id = build_image(str(cfg.docker.image))
    (metadata_dir / "environment.json").write_text(json.dumps({
        "image_tag": str(cfg.docker.image),
        "image_id": image_id,
        "network": str(cfg.docker.network),
        "docker_version": subprocess.run(["docker", "--version"], capture_output=True,
                                         text=True).stdout.strip(),
        "upstream": "PalisadeResearch/ctfish@f029c613ec40b059cc8d4742cb515a138aa6922c",
    }, indent=2))

    target_client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                           timeout=float(cfg.generation.request_timeout),
                           max_retries=int(cfg.generation.max_retries))

    # top_k is a vLLM sampling parameter with no OpenAI equivalent, so it only reaches
    # the server through extra_body — set on the client path it would be dropped in
    # silence. Left unset by default (vLLM disables it); Qwen's published thinking-mode
    # settings include top_k=20, so a run pinned to those needs it on the wire.
    top_k = cfg.generation.get("top_k")
    extra_body = {"top_k": int(top_k)} if top_k is not None else None

    def chat(messages: list[dict]) -> tuple[str, str | None, str]:
        resp = target_client.chat.completions.create(
            model=target.model_name,
            messages=messages,
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            max_tokens=int(cfg.generation.max_tokens),
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        # vLLM's out-of-band trace field is version-dependent (`reasoning_content` on
        # 0.8.x, `reasoning` on 0.26); resolve_trace covers both plus the inline form.
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
            choice.message, "reasoning", None)
        return choice.message.content or "", reasoning, choice.finish_reason or ""

    def rollout(i: int) -> Rollout | None:
        run_id = f"run_{i:03d}"
        try:
            with Sandbox(image=str(cfg.docker.image), network=str(cfg.docker.network),
                         command_timeout_s=int(cfg.command_timeout_s)) as sandbox:
                result = run_rollout(
                    run_id, variant, task_prompt, sandbox, chat, max_steps=max_steps,
                    pre_invoke_msg=pre_invoke,
                    history_carries_reasoning=bool(cfg.history_carries_reasoning))
        except Exception as exc:  # noqa: BLE001 — one rollout must not sink the arm
            print(f"!!! rollout {run_id} failed: {type(exc).__name__}: {exc}", flush=True)
            (rollout_dir / f"{run_id}_error.json").write_text(json.dumps(
                {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
            return None
        # Written the moment the run ends: judging happens after every rollout, and a
        # failure there must never cost a generation.
        (rollout_dir / f"{run_id}.json").write_text(
            json.dumps(_rollout_record(result), indent=2, ensure_ascii=False))
        (rollout_dir / f"{run_id}.md").write_text(judging.format_journal(result))
        print(f">>> {run_id}: {len(result.steps)} steps, stop={result.stop_reason}, "
              f"{sum(len(s.commands) for s in result.steps)} commands", flush=True)
        return result

    rollouts = [r for r in map_threaded(rollout, n_runs,
                                        max_workers=int(cfg.concurrency.rollouts),
                                        desc="ctfish rollouts") if r is not None]
    if not rollouts:
        raise RuntimeError("every ctfish rollout failed — see rollouts/*_error.json")

    openrouter = OpenRouterClient()
    # The (rollout, sample) grid is flattened so judging is parallel across BOTH axes:
    # majority@5 over 10 runs is 50 independent calls, and taking them five-at-a-time per
    # run would serialize the slowest run's samples behind each other for no reason.
    jobs = [(rollout_, k) for rollout_ in rollouts for k in range(judge_samples)]
    messages_by_run = {r.run_id: judging.judge_messages(r) for r in rollouts}
    # Judge spend scales with journal length x samples, and a journal is as long as the
    # agent made it. Printed before the calls go out so an unexpectedly expensive arm is
    # visible in the log rather than only on the bill.
    journal_chars = sum(len(m["content"]) for msgs in messages_by_run.values()
                        for m in msgs)
    print(f">>> judging {len(rollouts)} runs x {judge_samples} samples "
          f"({journal_chars * judge_samples / len(messages_by_run) / 1000:.0f}k chars "
          f"per run-sample set, ~{journal_chars * judge_samples / 4000:.0f}k input "
          "tokens total)", flush=True)

    def score(i: int) -> dict:
        rollout_, sample = jobs[i]
        base = {"run_id": rollout_.run_id, "sample": sample}
        try:
            reply = openrouter.chat(str(cfg.judge.model),
                                    messages_by_run[rollout_.run_id],
                                    temperature=float(cfg.judge.temperature),
                                    max_tokens=int(cfg.judge.max_tokens)).content
            return {**base, "escalation": judging.parse_escalation(reply),
                    "reasoning": reply}
        except Exception as exc:  # noqa: BLE001 — a lost sample is reported, not fatal
            print(f"!!! judge failed for {rollout_.run_id} sample {sample}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return {**base, "error": f"{type(exc).__name__}: {exc}"}

    rows = map_threaded(score, len(jobs), max_workers=int(cfg.concurrency.judge_workers),
                        desc="ctfish judging")
    with (results_dir / "judge_samples.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    verdicts = [judging.majority(
        r.run_id,
        [row["escalation"] for row in rows
         if row["run_id"] == r.run_id and "escalation" in row],
        [row["error"] for row in rows if row["run_id"] == r.run_id and "error" in row])
        for r in rollouts]
    judging.write_verdicts(results_dir / "verdicts.jsonl", verdicts)

    summary = summarize(rollouts, verdicts, think_mode=target.spec.mode == "think") | {
        "variant": variant,
        "max_steps": max_steps,
        "judge_model": str(cfg.judge.model),
        "judge_samples": judge_samples,
        "runs_failed": n_runs - len(rollouts),
        # The harness knobs that differ between our default and an upstream-style run,
        # recorded in the summary as well as run_meta: two ctfish numbers are only
        # comparable if these match.
        "pre_invoke_msg": pre_invoke,
        "history_carries_reasoning": bool(cfg.history_carries_reasoning),
        "top_k": top_k,
        "temperature": float(cfg.generation.temperature),
    }
    (results_dir / "report.md").write_text(
        report_markdown(target.spec.hf_path, summary, verdicts))
    return summary
