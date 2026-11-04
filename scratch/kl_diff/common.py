# ABOUTME: Shared pieces for the per-token KL diffing pilot (card "Where DA changes transcripts"):
# ABOUTME: the adapters and runs, the serve-time chat template, and render + label of one scored step.
"""One scored STEP = the exact prompt a model saw + the output it wrote there.

`render_step` rebuilds both with the model's own chat template, pinned the way the server
pinned it (`pin_template`: thinking on, `preserve_thinking=false`), and labels every target
character as reasoning / visible / tool_call / format. Only target tokens are ever scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.infra.endpoints.vllm import pin_template
from src.model_profile import render_chat

ORG = "dougalldeepmind"
BASE = ("Qwen/Qwen3.6-27B", "6a9e13bd")

# role -> (adapter repo, revision). The newest matched set: one base mix (2026-10-05 plain),
# the same 10-05 DA rows with and without unused tool definitions.
ADAPTERS = {
    "nosynth": (
        f"{ORG}/2026-10-05-qwen36-0-plain",
        "1565d4f3356aa1b3bea1c503700164faa240ef3a",
    ),
    "da": (
        f"{ORG}/2026-10-05-qwen36-0-da-15",
        "ce773a6b87db66bd569dd13071ec5e27201462f1",
    ),
    "da_tools": (
        f"{ORG}/2026-10-11-qwen36-0-da-tools-15",
        "2b14c0d8110abdc8b62af21ba78be7ad77f51d71",
    ),
}

# eval -> writer -> run repo whose rollouts were written by that writer's adapter.
RUNS = {
    "mask": {
        "nosynth": f"{ORG}/2026-11-03-mask-qwen36-0-plain",
        "da": f"{ORG}/2026-11-03-mask-qwen36-0-da-15",
        "da_tools": f"{ORG}/2026-10-11-mask-qwen36-0-da-tools-15",
    },
    "hospital": {
        "nosynth": f"{ORG}/2026-10-11-hospital-self-sacrificial-qwen36-0-plain",
        "da": f"{ORG}/2026-10-15-hospital-self-sacrificial-qwen36-0-da-15",
        "da_tools": f"{ORG}/2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15",
    },
}

THINK_CLOSE = "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
TURN_END = "<|im_end|>"
LABELS = ("reasoning", "visible", "tool_call", "format")


def load_tokenizer():
    """The base tokenizer with the serve-time template pin (thinking on, preserve_thinking off)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1])
    tok.chat_template = pin_template(tok.chat_template, "think")
    return tok


@dataclass
class Rendered:
    text: str  # prompt + target, exactly as scored
    prompt_chars: int  # target starts at this character
    char_labels: list[str]  # one label per target character


def render_step(
    tok, messages: list[dict], tools: list[dict] | None, target: dict
) -> Rendered:
    """Render prompt and prompt+target; the target is what the template appends for `target`.

    `target` is an assistant message: {reasoning_content, content, tool_calls?}. Fails loudly if
    the full render does not extend the generation-prompt render (the template would then be
    rewriting history, and the scored tokens would not be the model's own output).
    """
    prompt = render_chat(
        tok, messages, tools, render_kwargs={}, add_generation_prompt=True
    )
    full = render_chat(
        tok, messages + [dict(target, role="assistant")], tools, render_kwargs={}
    )
    assert full.startswith(prompt), "full render does not extend the prompt render"
    tgt = full[len(prompt) :]
    return Rendered(full, len(prompt), label_target(tgt))


def label_target(tgt: str) -> list[str]:
    """Per-character labels for one assistant output (starting right after the `<think>\\n` prefill)."""
    labels = ["visible"] * len(tgt)
    close = tgt.find(THINK_CLOSE)
    assert close != -1, "target has no </think>"
    for i in range(close):
        labels[i] = "reasoning"
    # the close tag and the blank line after it are template format, not model content
    after = close + len(THINK_CLOSE)
    while after < len(tgt) and tgt[after] == "\n":
        after += 1
    for i in range(close, after):
        labels[i] = "format"
    pos = after
    while (o := tgt.find(TOOL_OPEN, pos)) != -1:
        c = tgt.find(TOOL_CLOSE, o)
        assert c != -1, "unterminated <tool_call>"
        for i in range(o, c + len(TOOL_CLOSE)):
            labels[i] = "tool_call"
        pos = c + len(TOOL_CLOSE)
    e = tgt.rfind(TURN_END)
    if e != -1:
        for i in range(e, len(tgt)):
            labels[i] = "format"
    # a "visible" stretch that is only whitespace (the newlines between tool calls) is format
    i = 0
    while i < len(tgt):
        if labels[i] != "visible":
            i += 1
            continue
        j = i
        while j < len(tgt) and labels[j] == "visible":
            j += 1
        if tgt[i:j].strip() == "":
            for k in range(i, j):
                labels[k] = "format"
        i = j
    return labels


def token_labels(tok, rendered: Rendered) -> tuple[list[int], int, list[str]]:
    """Token ids of the whole text, index of the first target token, label per target token.

    A token's label is the label of its first character. Asserts the prompt/target boundary
    falls on a token boundary (it does for Qwen3.6: the prompt ends with `<think>\\n`).
    """
    enc = tok(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    first = next(i for i, (s, _) in enumerate(offs) if s >= rendered.prompt_chars)
    assert offs[first][0] == rendered.prompt_chars, (
        "prompt/target boundary splits a token"
    )
    labs = [rendered.char_labels[s - rendered.prompt_chars] for s, _ in offs[first:]]
    return ids, first, labs


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)
