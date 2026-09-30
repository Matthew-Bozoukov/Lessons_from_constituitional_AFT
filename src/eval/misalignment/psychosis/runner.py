# ABOUTME: Eval-framework entrypoint for the AI-psychosis eval: run every persona's
# ABOUTME: escalation arc against the served target, judge each turn, aggregate.

"""run() per the CLAUDE.md eval contract.

Two phases, both re-entrant across runs (no process
globals, everything under out_dir):

1. Conversations — one thread per persona (each internally sequential: 12 turns of
   red-teamer -> target). A persona that fails is recorded and reported loudly, not
   allowed to discard the others' finished work; any failed persona blocks a summary.
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

from src.eval.layout import publish_layout
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded
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
    available = {p.stem: p.read_text(encoding="utf-8")
                 for p in sorted((ASSETS / "characters").glob("*.txt"))}
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

    red_team_template = (ASSETS / "red_team_prompt.txt").read_text(encoding="utf-8")
    grader_prompt = (ASSETS / "grader_prompt.txt").read_text(encoding="utf-8")

    openrouter = OpenRouterClient()
    target_client = OpenAI(
        base_url=target.base_url,
        api_key=target.api_key,
        timeout=float(cfg.generation.request_timeout),
        max_retries=int(cfg.generation.max_retries),
    )

    def recorded_chat(settings, prefix: str):
        """A separate recorder per persona/cell retains failed calls and retries."""
        count = 0
        def chat(messages: list[dict]) -> str:
            nonlocal count
            count += 1
            record = {"model": str(settings.model), "messages": messages,
                      "temperature": float(settings.temperature),
                      "max_tokens": int(settings.max_tokens)}
            try:
                reply = openrouter.chat(record["model"], messages,
                                       temperature=record["temperature"],
                                       max_tokens=record["max_tokens"])
                record.update(content=reply.content, finish_reason=reply.finish_reason,
                              reasoning=(getattr(reply, "reasoning_content", None)
                                         or getattr(reply, "reasoning", None)))
                if reply.finish_reason != "stop":
                    raise ValueError(f"Psychosis {prefix} did not finish")
                return reply.content
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                (rollout_dir / f"{prefix}_{count:03d}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return chat

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

    rollout_dir, results_dir, _ = publish_layout(out_dir)
    stems = sorted(characters)

    def converse(i: int) -> dict:
        stem = stems[i]
        redteam_chat = recorded_chat(cfg.redteam, f"{stem}_redteam")
        prompt = format_red_team_prompt(red_team_template, character_name(stem),
                                        characters[stem])
        def checkpoint(turns):
            (rollout_dir / f"{stem}.json").write_text(json.dumps(
                {"character": stem, "red_team_prompt": prompt,
                 "complete": len(turns) == n_turns and turns[-1].finish_reason == "stop"
                             and bool(turns[-1].assistant.strip()),
                 "turns": [asdict(t) for t in turns]}, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            turns = run_conversation(character_name(stem), prompt, n_turns,
                                     redteam_chat, target_chat, on_turn=checkpoint)
        except Exception as exc:  # noqa: BLE001 — one persona must not sink the rest
            print(f"!!! conversation failed for {stem}: {type(exc).__name__}: {exc}")
            (rollout_dir / f"{stem}_error.json").write_text(
                json.dumps({"character": stem, "error": f"{type(exc).__name__}: {exc}"},
                           indent=2), encoding="utf-8")
            return {"stem": stem, "prompt": prompt, "turns": None}
        # The final on_turn checkpoint already persists the complete conversation,
        # including its completion flag, before any judging starts.
        return {"stem": stem, "prompt": prompt, "turns": turns}

    conversations = map_threaded(converse, len(stems),
                                 max_workers=int(cfg.concurrency.characters),
                                 desc="psychosis conversations")
    completed = [c for c in conversations if c["turns"] is not None]
    failed_characters = [c["stem"] for c in conversations if c["turns"] is None]
    if not completed:
        raise RuntimeError("every persona conversation failed — see rollouts/*_error.json")
    if failed_characters:
        raise RuntimeError(f"Incomplete psychosis panel: {failed_characters}; partial turns retained in rollouts")

    jobs = [(c, k) for c in completed for k in range(1, len(c["turns"]) + 1)]

    def grade(i: int) -> dict:
        convo, upto = jobs[i]
        base = {"character": convo["stem"], "turn": upto}
        judge_chat = recorded_chat(cfg.judge, f"{convo['stem']}_judge_turn{upto:02d}")
        try:
            return {**base, **grade_turn(judge_chat, grader_prompt, convo["turns"], upto)}
        except Exception as exc:  # noqa: BLE001 — a failed cell is reported, not fatal
            print(f"!!! judge failed for {convo['stem']} turn {upto}: {exc}")
            return {**base, "judge_error": f"{type(exc).__name__}: {exc}"}

    rows = map_threaded(grade, len(jobs), max_workers=int(cfg.concurrency.judge_workers),
                        desc="psychosis judging")
    graded = [r for r in rows if "judge_error" not in r]

    with (results_dir / "grades.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (results_dir / "grades.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["character", "turn", *GRADE_KEYS])
        writer.writeheader()
        writer.writerows(graded)

    for convo in completed:
        grades = {r["turn"]: {k: r[k] for k in GRADE_KEYS} for r in graded
                  if r["character"] == convo["stem"]}
        (rollout_dir / f"{convo['stem']}.md").write_text(
            _rollout_markdown(convo["stem"], convo["prompt"], convo["turns"], grades),
            encoding="utf-8")

    all_turns = [t for c in completed for t in c["turns"]]
    if len(graded) != len(jobs):
        raise RuntimeError("Incomplete psychosis judgments; partial grades and rollouts retained")
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
