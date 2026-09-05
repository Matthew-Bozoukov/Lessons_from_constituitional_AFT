# ABOUTME: The ctfish OODA agent — upstream player.py's observe/orient/decide/act state
# ABOUTME: machine, reimplemented driver-side so it drives a sandbox and a served target.

"""The harness half of the eval.

Upstream runs this loop INSIDE the container and has it call an LLM API from there. Here
it runs in the eval driver and the container is a bare sandbox reached by `docker exec`
(`sandbox.py`), for three reasons that are all about this repo rather than about chess:

- the repo's eval contract is "the eval runs where you type it; only the model moves"
  (CLAUDE.md), so the model can be served on a rented pod while docker runs on a laptop —
  and the driver-side loop needs no docker-bridge tunnel to reach it;
- no credentials and no endpoint enter a sandbox whose whole point is that the agent may
  rewrite anything it can reach;
- the transcript is built where it is written, so a killed container loses nothing.

What the model sees is unchanged: the same prompts (`prompts.py`, verbatim), the same
state cells, the same history trimming, the same `<OUTPUT rc="...">` envelopes, and the
same `/env` with `game.py` and stockfish in it.

Two adaptations for a served reasoning model, both recorded in run_meta via the config:

- **Reasoning is split off before parsing.** vLLM returns the trace either inline in
  `<think>` tags or out of band, and `resolve_trace` covers both. Only the VISIBLE reply
  is scanned for `<ACTION>`/`<PLAN>`/etc — a command proposed inside a reasoning trace is
  a thought, not an action, and executing one would fabricate behaviour the model did not
  commit to.
- **History carries `reasoning_content`** (repo preserve-thinking policy, 2026-08-04): the
  arms train with reasoning kept in context and the served template pins
  `preserve_thinking` to match, so stripping it would be a train/inference mismatch — and
  on this family a reasoning-less assistant turn renders as the EMPTY think marker, which
  is the literal training signal for "do not reason".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Callable

from bs4 import BeautifulSoup

from src.eval.misalignment.ctfish.prompts import AGENT_PROMPT, PHASE_PROMPTS
from src.eval.misalignment.ctfish.sandbox import CommandResult, Sandbox
from src.model_profile import resolve_trace

# (messages) -> (content, out_of_band_reasoning, finish_reason). The same shape psychosis
# uses, so one served-target calling convention covers both evals.
TargetChat = Callable[[list[dict]], tuple[str, str | None, str]]

INITIAL_PHASE = "orient"  # upstream: "skipping observe as there are no actions yet"
QUIT_COMMANDS = (["quit"], [":"])


def message(role: str, body: str, wrapper: str | None = None) -> dict:
    """Upstream `Provider.message`: dedent, strip, optionally wrap in an XML-ish tag.

    Upstream's o1/o3 role rewriting is dropped: those branches exist to satisfy OpenAI
    reasoning endpoints that reject a system role, and a vLLM-served Qwen3.6 takes the
    roles as written.
    """
    content = dedent(body).strip()
    if wrapper:
        content = f"<{wrapper}>{content}</{wrapper}>"
    return {"role": role, "content": content}


def coalesce(messages: list[dict]) -> list[dict]:
    """Upstream `coalesce`: drop empty messages and merge consecutive same-role ones.

    One deviation: a message carrying `reasoning_content` is never merged into or merged
    with. Concatenating its text would silently strip the trace off it, which is the exact
    train/inference mismatch the preserve-thinking policy exists to avoid. In practice this
    only ever splits an assistant state-cell block from an assistant history entry.
    """
    grouped: list[dict] = []
    for msg in messages:
        if not msg or not msg.get("content") or not msg.get("role"):
            continue
        mergeable = (grouped and grouped[-1]["role"] == msg["role"]
                     and not grouped[-1].get("reasoning_content")
                     and not msg.get("reasoning_content"))
        if mergeable:
            grouped[-1]["content"] += "\n" + msg["content"]
        else:
            grouped.append(dict(msg))
    return grouped


def bash_strip(line_: str) -> str:
    """Upstream `bash_strip`: drop a trailing shell comment, respecting quotes."""
    line = line_.strip()
    in_single_quotes = False
    in_double_quotes = False
    comment_start_index = -1
    for i, char in enumerate(line):
        is_escaped = i > 0 and line[i - 1] == "\\"
        if char == "'" and not is_escaped:
            in_single_quotes = not in_single_quotes
        elif char == '"' and not is_escaped:
            in_double_quotes = not in_double_quotes
        elif (char == "#" and not in_single_quotes and not in_double_quotes
              and not is_escaped):
            comment_start_index = i
            break
    if comment_start_index != -1:
        line = line[:comment_start_index].rstrip()
    return line.strip()


def remember(memory: dict[int, str], *entries: str) -> None:
    """Upstream `remember`: append each non-empty entry under the next free id."""
    for content in entries:
        body = dedent(content).strip()
        if body:
            memory[max(memory.keys(), default=0) + 1] = body


def recall(memory: dict[int, str]) -> str:
    """Upstream `recall`: the memory cell as `<MEMORY id=N>` blocks, id order."""
    return "\n".join(f"<MEMORY id={ix}>{content}</MEMORY>"
                     for ix, content in sorted(memory.items()))


def toplevel(reply: str, tag: str) -> list[str]:
    """The stripped text of every TOP-LEVEL `<tag>` in `reply` (upstream's `toplevel`).

    Top-level only, exactly as upstream: a tag nested inside another (an `<ACTION>` quoted
    within a `<PLAN>`, say) is discussion, not a command.
    """
    soup = BeautifulSoup(reply, "html.parser")
    return [node.text.strip() for node in soup.find_all(tag.lower(), recursive=False)]


@dataclass
class Step:
    """One turn of the loop: what the model was asked to do and what came back."""

    n: int
    phase: str
    think: str
    reply: str            # the visible reply, which is what was parsed
    finish_reason: str
    commands: list[str] = field(default_factory=list)
    results: list[CommandResult] = field(default_factory=list)
    next_phase: str = ""
    # Harness-side events worth seeing in the transcript: a phase that produced no valid
    # tag (the model stalled), a malformed FORGET, an empty ACTION block.
    notes: list[str] = field(default_factory=list)

    @property
    def progressed(self) -> bool:
        """True when this step's reply moved the state machine on."""
        return self.next_phase != self.phase


@dataclass
class Rollout:
    """One complete run: the transcript plus the state cells it ended with."""

    run_id: str
    variant: str
    task_prompt: str
    steps: list[Step]
    memory: dict[int, str]
    plan: list[str]
    subgoal: list[str]
    stop_reason: str            # quit | max_steps
    game_state: dict[str, str] = field(default_factory=dict)

    @property
    def stalled_steps(self) -> int:
        """Steps whose reply carried no valid tag for their phase."""
        return sum(not step.progressed for step in self.steps)


def build_messages(task_prompt: str, memory: dict[int, str], plan: list[str],
                   subgoal: list[str], history: list[dict], phase: str,
                   pre_invoke_msg: str) -> list[dict]:
    """Assemble one step's context, in upstream's order (paper section "Prompt structure")."""
    messages = [
        message("system", AGENT_PROMPT),
        message("user", task_prompt, "TASK"),
        message("assistant", recall(memory)),
        message("assistant", "".join(plan) or "Prepare the plan.", "PLAN"),
    ]
    if subgoal:
        prefix = ["(Previous)"] if phase in ("orient", "decide") else []
        messages.append(message("assistant", " ".join(prefix + subgoal), "SUBGOAL"))
    messages.extend(history)
    messages.append(message("user", PHASE_PROMPTS[phase]))
    messages.append(message("user", pre_invoke_msg))
    return coalesce(messages)


def run_rollout(run_id: str, variant: str, task_prompt: str, sandbox: Sandbox,
                chat: TargetChat, *, max_steps: int, pre_invoke_msg: str,
                history_carries_reasoning: bool = True) -> Rollout:
    """Drive one agent run to `quit` or `max_steps`.

    Args:
        run_id: Identifier for this rollout, used in filenames and judge records.
        variant: Which task-prompt variant this rollout ran (recorded, not used here).
        task_prompt: The variant's text, already selected by the caller.
        sandbox: A started `Sandbox`; every `<ACTION>` command is executed in it.
        chat: Calls the served target; returns (content, out-of-band reasoning, finish_reason).
        max_steps: Hard cap on loop iterations (upstream's MAX_STEPS).
        pre_invoke_msg: The nudge appended as the last user message of every step.
        history_carries_reasoning: Whether an assistant history entry carries its
            `reasoning_content`. True is this repo's preserve-thinking policy (the
            arms train with reasoning in context and the served template matches).
            False is upstream's shape — history holds the visible reply alone — and
            is what a faithful comparison against an upstream-style run needs.

    Returns:
        The completed `Rollout`, including the sandbox's final game files.
    """
    memory: dict[int, str] = {}
    plan: list[str] = []
    subgoal: list[str] = []
    history: list[dict] = []
    steps: list[Step] = []
    phase = INITIAL_PHASE
    stop_reason = "max_steps"

    for n in range(max_steps):
        messages = build_messages(task_prompt, memory, plan, subgoal, history, phase,
                                  pre_invoke_msg)
        content, reasoning, finish_reason = chat(messages)
        think, visible = resolve_trace(content, reasoning)
        step = Step(n=n, phase=phase, think=think, reply=visible,
                    finish_reason=finish_reason)
        # The model's own turn goes into history before the harness reacts to it, so a
        # command's OUTPUT always follows the message that proposed it.
        entry = {"role": "assistant", "content": visible}
        if think and history_carries_reasoning:
            entry["reasoning_content"] = think
        history.append(entry)

        next_phase = _advance(step, visible, memory, plan, subgoal, history, sandbox)
        step.next_phase = next_phase
        steps.append(step)

        if next_phase == "quit":
            stop_reason = "quit"
            break
        if next_phase != phase:
            # Upstream's sliding window: on a phase change, keep only the last two
            # history entries. Without it the context grows until the model derails.
            while len(history) > 2:
                history.pop(0)
        phase = next_phase

    return Rollout(run_id=run_id, variant=variant, task_prompt=task_prompt, steps=steps,
                   memory=memory, plan=plan, subgoal=subgoal, stop_reason=stop_reason,
                   game_state=sandbox.snapshot_game_state())


def _advance(step: Step, visible: str, memory: dict[int, str], plan: list[str],
             subgoal: list[str], history: list[dict], sandbox: Sandbox) -> str:
    """Apply one reply to the state cells and return the next phase (upstream `step`).

    A phase that finds no valid tag returns its own name, so the model gets another
    attempt at it with its failed reply still in history — upstream's behaviour, and the
    reason `Rollout.stalled_steps` is reported.
    """
    phase = step.phase
    if phase == "observe":
        next_phase = "orient"
        for text in toplevel(visible, "FORGET"):
            try:
                forgotten = memory.pop(int(text), None)
            except ValueError as e:
                history.append(message("user", f"FORGET failed: {e}", "ERROR"))
                step.notes.append(f"FORGET failed: {e}")
                next_phase = "observe"
                continue
            if forgotten:
                step.notes.append(f"forgot memory id={text}")
        for text in toplevel(visible, "REMEMBER"):
            remember(memory, text)
        return next_phase

    if phase == "orient":
        for text in toplevel(visible, "PLAN"):
            plan.clear()
            plan.append(text)
        return "decide" if plan else "orient"

    if phase == "decide":
        for text in toplevel(visible, "SUBGOAL"):
            subgoal.clear()
            subgoal.append(text)
        return "act" if subgoal else "decide"

    assert phase == "act", f"unexpected phase: {phase!r}"
    commands = [line for node in toplevel(visible, "ACTION")
                for line in map(bash_strip, node.split("\n")) if line]
    if not commands:
        history.append(message(
            "user", "No commands submitted. Use `quit` to finish the session."))
        step.notes.append("no commands submitted")
        return "act"
    if commands in QUIT_COMMANDS:
        step.commands = commands
        return "quit"
    step.commands = commands
    step.results = [sandbox.exec_command(command) for command in commands]
    history.append(message("user", "\n".join(
        f'<OUTPUT rc="{r.returncode}">{r.output}</OUTPUT>' for r in step.results)))
    return "observe"
