# ABOUTME: Eval-framework entrypoint for the AI-psychosis eval: run every persona's
# ABOUTME: escalation arc against the served target, judge each turn, aggregate.

"""run() per the CLAUDE.md eval contract.

Two phases, both resumable within a run and re-entrant across runs (no process
globals, everything under out_dir):

1. Conversations — one thread per persona (each internally sequential: 12 turns of
   red-teamer -> target). A persona that fails is recorded and reported loudly, not
   allowed to sink the others' finished work; the run only errors if ALL fail.
2. Judging — every (persona, turn) cell grades concurrently against OpenRouter.

Rollouts (CLAUDE.md: "logs means ROLLOUTS") are self-contained per persona: the
formatted red-teamer instructions (persona arc included), every strategy note, the
target's reasoning trace and visible reply, and the per-turn grades.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.eval.misalignment.psychosis.conversation import (
    Turn,
    format_red_team_prompt,
    run_conversation,
    split_redteam_completion,
)
from src.eval.misalignment.psychosis.judge import GRADE_KEYS, grade_turn
from src.eval.misalignment.psychosis.metrics import summarize
from src.utils import transcript_markdown

ASSETS = Path(__file__).parent / "assets"


def load_characters(selected: list[str] | None) -> dict[str, str]:
    """Load persona files as {stem: text}, optionally restricted to `selected` stems."""
    available = {p.stem: p.read_text() for p in sorted((ASSETS / "characters").glob("*.txt"))}
    if selected is None:
        return available
    missing = [s for s in selected if s not in available]
    if missing:
        raise SystemExit(f"Unknown characters {missing}; available: {sorted(available)}")
    return {s: available[s] for s in selected}


def character_name(stem: str) -> str:
    """'ethan_reality' -> 'Ethan' (upstream's naming rule)."""
    return stem.split("_")[0].title()


def _rollout_markdown(stem: str, prompt: str, turns: list[Turn], grades: dict[int, dict]) -> str:
    """Compose one persona's transcript via the shared renderer (src.utils).

    The red-teamer completion is split into out-of-character strategy (fenced — it is
    chain-of-thought, never prose) and the in-character message; the target's reasoning
    section is ALWAYS present, so an empty trace is visible as such rather than
    silently missing (how the zara pre-fix leak hid on HF).
    """
    sections: list[tuple[int, str, str, str]] = [
        (2, "Red-teamer instructions", "fenced", prompt),
    ]
    for t in turns:
        strategy, message = split_redteam_completion(t.redteam_raw)
        sections.append((2, f"Turn {t.turn}", "text", ""))
        if message is None:
            sections.append((3, "Red-teamer (full completion — no <message> block)",
                             "fenced", t.redteam_raw))
        else:
            sections.append((3, "Red-teamer strategy (out of character)", "fenced",
                             strategy))
            sections.append((3, "User message (in character)", "text", message))
        sections.append((3, "Target reasoning", "fenced",
                         t.think or "(no reasoning captured for this turn — see "
                                    "empty_think_rate in results.json)"))
        sections.append((3, "Target reply", "text", t.assistant))
        grade = grades.get(t.turn)
        sections.append((3, "Grades", "json", json.dumps(grade, indent=2)) if grade
                        else (3, "Grades", "text", "(judge failed for this turn)"))
    return transcript_markdown(
        f"Psychosis rollout — {character_name(stem)} ({stem})",
        "Self-contained transcript: red-teamer instructions (persona arc included), the "
        "red-teamer's strategy notes, the target's reasoning and visible replies, and "
        "the judge's per-turn grades.",
        sections)


def run(target, cfg, out_dir: Path) -> dict:
    """Run the psychosis eval against one ServedTarget (CLAUDE.md contract).

    Returns:
        Summary metrics (referral/pushback/delusion-confirmation, therapy criteria,
        per-character and per-turn breakdowns, generation-health rates).
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    selected = OmegaConf.to_container(cfg.characters, resolve=True) if cfg.characters else None
    characters = load_characters(selected)
    n_turns = int(cfg.turns)
    if bool(cfg.get("smoke", False)):
        characters = dict(sorted(characters.items())[:1])
        n_turns = min(n_turns, 3)

    red_team_template = (ASSETS / "red_team_prompt.txt").read_text()
    grader_prompt = (ASSETS / "grader_prompt.txt").read_text()

    openrouter = OpenRouterClient()
    target_client = OpenAI(
        base_url=target.base_url,
        api_key=target.api_key,
        timeout=float(cfg.generation.request_timeout),
        max_retries=int(cfg.generation.max_retries),
    )

    def redteam_chat(messages: list[dict]) -> str:
        return openrouter.chat(str(cfg.redteam.model), messages,
                               temperature=float(cfg.redteam.temperature),
                               max_tokens=int(cfg.redteam.max_tokens)).content

    def judge_chat(messages: list[dict]) -> str:
        return openrouter.chat(str(cfg.judge.model), messages,
                               temperature=float(cfg.judge.temperature),
                               max_tokens=int(cfg.judge.max_tokens)).content

    def target_chat(messages: list[dict]) -> tuple[str, str | None, str]:
        resp = target_client.chat.completions.create(
            model=target.model_name,
            messages=messages,
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            max_tokens=int(cfg.generation.max_tokens),
        )
        choice = resp.choices[0]
        # vLLM's out-of-band trace field is version-dependent (mmlu_eval, same fix).
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
            choice.message, "reasoning", None)
        return choice.message.content or "", reasoning, choice.finish_reason or ""

    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(characters)

    def converse(i: int) -> dict:
        stem = stems[i]
        prompt = format_red_team_prompt(red_team_template, character_name(stem),
                                        characters[stem])
        try:
            turns = run_conversation(character_name(stem), prompt, n_turns,
                                     redteam_chat, target_chat)
        except Exception as exc:  # noqa: BLE001 — one persona must not sink the rest
            print(f"!!! conversation failed for {stem}: {type(exc).__name__}: {exc}")
            (rollout_dir / f"{stem}_error.json").write_text(
                json.dumps({"character": stem, "error": f"{type(exc).__name__}: {exc}"},
                           indent=2))
            return {"stem": stem, "prompt": prompt, "turns": None}
        # Written the moment the conversation finishes: a judge-phase crash or dead pod
        # loses no rollouts.
        (rollout_dir / f"{stem}.json").write_text(
            json.dumps({"character": stem, "red_team_prompt": prompt,
                        "turns": [asdict(t) for t in turns]}, indent=2))
        return {"stem": stem, "prompt": prompt, "turns": turns}

    conversations = map_threaded(converse, len(stems),
                                 max_workers=int(cfg.concurrency.characters),
                                 desc="psychosis conversations")
    completed = [c for c in conversations if c["turns"] is not None]
    failed_characters = [c["stem"] for c in conversations if c["turns"] is None]
    if not completed:
        raise RuntimeError("every persona conversation failed — see rollouts/*_error.json")

    jobs = [(c, k) for c in completed for k in range(1, len(c["turns"]) + 1)]

    def grade(i: int) -> dict:
        convo, upto = jobs[i]
        base = {"character": convo["stem"], "turn": upto}
        try:
            return {**base, **grade_turn(judge_chat, grader_prompt, convo["turns"], upto)}
        except Exception as exc:  # noqa: BLE001 — a failed cell is reported, not fatal
            print(f"!!! judge failed for {convo['stem']} turn {upto}: {exc}")
            return {**base, "judge_error": f"{type(exc).__name__}: {exc}"}

    rows = map_threaded(grade, len(jobs), max_workers=int(cfg.concurrency.judge_workers),
                        desc="psychosis judging")
    graded = [r for r in rows if "judge_error" not in r]

    with (out_dir / "grades.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "grades.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["character", "turn", *GRADE_KEYS])
        writer.writeheader()
        writer.writerows(graded)

    for convo in completed:
        grades = {r["turn"]: {k: r[k] for k in GRADE_KEYS} for r in graded
                  if r["character"] == convo["stem"]}
        (rollout_dir / f"{convo['stem']}.md").write_text(
            _rollout_markdown(convo["stem"], convo["prompt"], convo["turns"], grades))

    all_turns = [t for c in completed for t in c["turns"]]
    summary = summarize(graded) | {
        "n_characters": len(completed),
        "turns_per_character": n_turns,
        "characters_failed": failed_characters,
        "judge_failures": len(rows) - len(graded),
        "truncation_rate": round(sum(t.finish_reason == "length" for t in all_turns)
                                 / len(all_turns), 4),
        "redteam_model": str(cfg.redteam.model),
        "judge_model": str(cfg.judge.model),
    }
    if target.spec.mode != "nothink":
        # CLAUDE.md gotcha: a ~0-length trace means the arm stopped reasoning.
        summary["empty_think_rate"] = round(
            sum(not t.think.strip() for t in all_turns) / len(all_turns), 4)
    return summary
