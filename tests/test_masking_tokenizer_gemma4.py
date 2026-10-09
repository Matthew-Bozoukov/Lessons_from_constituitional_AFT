# ABOUTME: Real-tokenizer verification of the Gemma 4 mask: its generation boundary and its
# ABOUTME: in-turn tool responses. Marked `tokenizer`; skips unless gemma-4-31B-it is cached.

"""Gemma 4 differs from Qwen3.6 in two ways that decide whether it can be trained at all, so
both are pinned here against the real template rather than argued from its jinja:

1. The generation boundary exists ONLY under `enable_thinking: true`. In that mode the prompt
   ends `<|turn>model\\n` and the model emits its own `<|channel>thought`, so the training
   render extends the inference prefix. In the default mode the prompt already carries a CLOSED
   empty thought channel while the reasoning render puts the trace inside it — the prompt is
   then not a prefix of the target, and a profile in that mode would supervise tokens the model
   never generates.
2. A whole agentic trajectory is ONE `model` turn with tool responses interleaved inside it
   ("continuation detection" in the template). Qwen gives tool output its own turn, which is the
   premise src/model_profile.py records; here it is false, so `template.masked_inner` is what
   keeps tool output out of the targets. Without it the masker trains the model to invent tool
   results — this file fails if that regresses.
"""

from __future__ import annotations

import pytest

from src.model_profile import model_profile, render_chat
from src.train.masking import build_labels

GEMMA4_PROFILE = model_profile("gemma4")

pytestmark = pytest.mark.tokenizer

MODEL = "google/gemma-4-31B-it"

TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache (tests are no-network)")


def _supervised(tok, out) -> str:
    return tok.decode([v for v in out["labels"] if v != -100])


def _tool_row(tok) -> str:
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Report the figure."},
        {"role": "assistant", "content": "Looking.", "reasoning_content": "Check first.",
         "tool_calls": [{"type": "function", "function": {
             "name": "bash", "arguments": {"command": "cat data.csv"}}}]},
        {"role": "tool", "content": '{"stdout": "p90=16.8\\n", "returncode": 0}'},
        {"role": "assistant", "content": "The figure is 16.8."},
    ]
    return render_chat(tok, msgs, TOOLS, render_kwargs=GEMMA4_PROFILE.render_kwargs)


def test_thinking_mode_is_the_only_valid_generation_boundary(tok):
    user = [{"role": "user", "content": "Hi."}]
    turn = [{"role": "assistant", "content": "Hello.", "reasoning_content": "Be brief."}]

    on = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=True)
    rendered = tok.apply_chat_template(user + turn, tokenize=False,
                                       add_generation_prompt=False, enable_thinking=True)
    assert on.endswith(GEMMA4_PROFILE.assistant_header)
    assert rendered.startswith(on), "training render must extend the inference prefix"

    # The default mode closes the thought channel in the prompt but fills it in the target, so
    # the profile's render_kwargs are load-bearing, not cosmetic.
    off = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    assert not tok.apply_chat_template(user + turn, tokenize=False,
                                       add_generation_prompt=False).startswith(off)


def test_the_model_generates_its_own_think_opener_so_nothing_is_prefilled(tok):
    # prefill and empty_think are verified as EMPTY for this family; the opener the model emits
    # must therefore be supervised, or it never learns to open a thought channel.
    assert GEMMA4_PROFILE.prefill == ""
    assert GEMMA4_PROFILE.empty_think == ""
    text = render_chat(tok, [{"role": "user", "content": "Hi."},
                             {"role": "assistant", "content": "Hello.",
                              "reasoning_content": "Be brief."}],
                       None, render_kwargs=GEMMA4_PROFILE.render_kwargs)
    got = _supervised(tok, build_labels(text, tok, 4096, GEMMA4_PROFILE))
    assert got.startswith("<|channel>thought"), got
    assert "Be brief." in got and "Hello." in got


def test_tool_output_is_masked_although_it_sits_inside_the_model_turn(tok):
    text = _tool_row(tok)
    assert "declaration:bash" in text, "the template places the schemas itself"
    assert "<|tool_call>call:bash" in text, "calls render in the family's own syntax"
    assert "<|tool_response>" in text, "tool output renders inside the same model turn"
    # One turn for the whole trajectory: the premise Qwen satisfies and Gemma does not.
    assert text.count(GEMMA4_PROFILE.turn_end) == 3

    got = _supervised(tok, build_labels(text, tok, 4096, GEMMA4_PROFILE, supervise="all"))
    assert "p90=16.8" not in got, "tool output is context, never a target"
    assert "cat data.csv" in got, "the call the model makes IS a target"
    assert "Check first." in got, "its reasoning is a target"


def test_masked_inner_is_what_excludes_the_tool_output(tok):
    # Strip the field and the same row starts training on tool output: this is the guard, and
    # the failure it prevents is silent.
    import dataclasses

    text = _tool_row(tok)
    unguarded = dataclasses.replace(GEMMA4_PROFILE, masked_inner=())
    leaked = _supervised(tok, build_labels(text, tok, 4096, unguarded, supervise="all"))
    assert "p90=16.8" in leaked


def test_cot_and_answer_partition_the_real_template(tok):
    # Gemma prefills nothing, so these modes find the trace by `think_open` instead. The two
    # must partition what `all` supervises, disjointly and exhaustively -- the property the
    # label arms (cot-only / response-only) rest on.
    text = render_chat(tok, [{"role": "user", "content": "Hi."},
                             {"role": "assistant", "content": "Hello.",
                              "reasoning_content": "Be brief."}],
                       None, render_kwargs=GEMMA4_PROFILE.render_kwargs)
    kept = {m: build_labels(text, tok, 4096, GEMMA4_PROFILE, supervise=m)
            for m in ("all", "cot", "answer")}
    n = {m: sum(1 for v in out["labels"] if v != -100) for m, out in kept.items()}
    assert n["cot"] + n["answer"] == n["all"], n
    assert _supervised(tok, kept["cot"]).endswith(GEMMA4_PROFILE.think_close)
    assert _supervised(tok, kept["answer"]) == "Hello." + GEMMA4_PROFILE.turn_end
    # `cot` truncates the row: the answer leaves the token stream, not just the loss.
    assert kept["cot"]["input_ids"] == kept["all"]["input_ids"][:len(kept["cot"]["input_ids"])]


@pytest.mark.parametrize("mode", ["cot", "answer"])
def test_a_turn_with_no_reasoning_is_refused(tok, mode):
    # Gemma signals "no reasoning" by never opening the channel, where Qwen writes an empty
    # marker. Both must be refused, or the arm silently becomes ordinary supervision.
    text = render_chat(tok, [{"role": "user", "content": "Hi."},
                             {"role": "assistant", "content": "Hello."}],
                       None, render_kwargs=GEMMA4_PROFILE.render_kwargs)
    assert GEMMA4_PROFILE.think_open not in text
    with pytest.raises(AssertionError, match="open its reasoning with"):
        build_labels(text, tok, 4096, GEMMA4_PROFILE, supervise=mode)
