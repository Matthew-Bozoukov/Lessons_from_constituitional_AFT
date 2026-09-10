# ABOUTME: Spec-aware quality filter for deliberative SFT: a format gate, then a comparative judge
# ABOUTME: that scores a prompt's candidates side by side k times; the best survivor is exported.

from __future__ import annotations

import json
import re

# One line per candidate at the end of the judge's answer. Anchored to a line so a score
# quoted inside the analysis cannot be mistaken for a verdict.
SCORE_RE = re.compile(r"^\s*CANDIDATE\s+(\d+)\s+SCORE:\s*(\d{1,2})\s*$", re.MULTILINE)
JUDGE_FIELDS = ("{constitution}", "{conversation}", "{candidates}")

# The generation prompt tells the model to keep the policy discussion in its reasoning. An
# answer that cites the constitution or a numbered principle to the USER has leaked the
# generation-time setup, and training on it would teach the model to do so unprompted.
LEAK_RE = re.compile(r"\bconstitution\b|\bprinciple\s*\d|\bthese (?:generation )?instructions\b", re.I)
# A first-person claim to have already performed an action. With no tool call in the turn,
# such a claim is unsubstantiated; the judge is told to verify it (a hint, never a gate).
ACTION_CLAIM_RE = re.compile(
    r"\bI(?:'ve| have) (?:already )?(?:drafted|queued|pulled|loaded|configured|compiled|staged|"
    r"generated|locked|filed|sent|submitted|scheduled|pushed|deployed)\b"
    r"|\bis (?:saved|queued|loaded) in the (?:portal|system|queue)\b", re.I)


def format_rejection(answer: str) -> str | None:
    """Why the answer fails the format gate, or None. Only rules that need no judgement."""
    if LEAK_RE.search(answer or ""):
        return "Final answer discusses the constitution or generation instructions"
    return None


def action_claim(answer: str) -> bool:
    """True when the answer asserts an action was already taken (a judge hint, not a gate)."""
    return bool(ACTION_CLAIM_RE.search(answer or ""))


def _calls_text(calls: list[dict]) -> str:
    return "\n".join(f"[tool call] {call['function']['name']}("
                     f"{json.dumps(call['function']['arguments'], ensure_ascii=False)})"
                     for call in calls)


def render_conversation(messages: list[dict]) -> str:
    """The prompt context as the judge reads it: every turn, labelled, tool calls inline."""
    parts = []
    for message in messages:
        body = message.get("content") or ""
        if message.get("tool_calls"):
            body = (body + "\n" if body.strip() else "") + _calls_text(message["tool_calls"])
        parts.append(f"[{message['role']}]\n{body.strip()}")
    return "\n\n".join(parts)


def render_answer(assistant: dict) -> str:
    """The candidate's final turn: its text plus any tool calls it issued."""
    body = (assistant.get("content") or "").strip()
    if assistant.get("tool_calls"):
        body = (body + "\n" if body else "") + _calls_text(assistant["tool_calls"])
    return body


def render_candidates(candidates: list[tuple[int, dict]]) -> str:
    """Every candidate of one prompt, index-tagged, each with its trace and final answer."""
    blocks = []
    for index, assistant in candidates:
        hint = ""
        if action_claim(assistant.get("content") or ""):
            hint = ("\n<automated_check>This candidate asserts it has ALREADY performed an action "
                    "(e.g. 'I have drafted', 'I've queued'). No tool call substantiates it. "
                    "Treat an unsubstantiated claim of a completed action as dishonest.</automated_check>")
        blocks.append(f'<candidate index="{index}">{hint}\n<reasoning>\n'
                      f"{assistant['reasoning_content'].strip()}\n</reasoning>\n<final_response>\n"
                      f"{render_answer(assistant)}\n</final_response>\n</candidate>")
    return "\n\n".join(blocks)


def judge_messages(record: dict, candidates: list[tuple[int, dict]], judge_prompt: str,
                   constitution: str) -> list[dict]:
    """One user turn holding the constitution, the context, and every candidate side by side."""
    if not all(field in judge_prompt for field in JUDGE_FIELDS):
        raise ValueError(f"judge_prompt must include all of {JUDGE_FIELDS}")
    if not candidates:
        raise ValueError("judge_messages needs at least one candidate")
    text = judge_prompt.format(constitution=constitution,
                               conversation=render_conversation(record["messages"]),
                               candidates=render_candidates(candidates))
    return [{"role": "user", "content": text}]


def parse_scores(text: str, indices: list[int]) -> dict[int, int]:
    """One `CANDIDATE <k> SCORE: <1-10>` line per judged candidate; the last line per k wins."""
    found: dict[int, int] = {}
    for index, score in SCORE_RE.findall(text or ""):
        found[int(index)] = int(score)
    missing = [k for k in indices if k not in found]
    if missing:
        raise ValueError(f"judge output lacks a SCORE line for candidates {missing}")
    if extra := sorted(set(found) - set(indices)):
        raise ValueError(f"judge scored candidates that were not shown: {extra}")
    if bad := [k for k in indices if not 1 <= found[k] <= 10]:
        raise ValueError(f"judge scores out of range for candidates {bad}")
    return {k: found[k] for k in indices}


def candidate_score(scores: list[int], runs: int) -> int | None:
    """The paper's rule: minimum across k runs, because any one run may miss a problem.

    None until every run has scored the candidate, so a half-judged candidate can never
    be selected on a resume.
    """
    if len(scores) < runs:
        return None
    return min(scores)


def select(candidates: dict[int, int | None], threshold: int) -> tuple[int, int] | None:
    """Best candidate (index, score) at or above `threshold`, ties to the lowest index."""
    ranked = sorted(((score, index) for index, score in candidates.items()
                     if score is not None and score >= threshold),
                    key=lambda pair: (-pair[0], pair[1]))
    if not ranked:
        return None
    score, index = ranked[0]
    return index, score
