# ABOUTME: Gemma 4 generation boundaries, including tool-result context inside model turns.
# ABOUTME: A separate prefix-render gate verifies these spans against the actual tokenizer.
from __future__ import annotations

import copy
import re

HEADER = "<|turn>model\n"
USER = "<|turn>user\n"
END = "<turn|>"
THOUGHT = "<|channel>thought\n"
TOOL_OPEN = "<|tool_response>"
TOOL_CLOSE = "<tool_response|>"


def generation_layout(text: str, supervise: str):
    """Keep the current exchange, masking result payloads and continuation prefills.

    The model generates the FIRST tool-response opener as its handoff marker.
    The executor supplies the response payload/close and any parallel results;
    the template then prefills THOUGHT before resuming generation.
    """
    if supervise != "full":
        raise ValueError("Gemma 4 currently qualifies full supervision only; ablations need separate qualification")
    start = text.rfind(USER)
    assert start >= 0, "Gemma training requires a real user turn"
    spans, forced = [], []
    pos = start
    while (head := text.find(HEADER, pos)) >= 0:
        s = head + len(HEADER)
        e = text.find(END, s)
        assert e >= 0, "Unterminated Gemma model turn"
        e += len(END)
        spans.append((s, e))
        cursor = s
        while (op := text.find(TOOL_OPEN, cursor, e)) >= 0:
            # Keep the handoff opener generated after the tool call.
            mask_start = op + len(TOOL_OPEN)
            close = text.find(TOOL_CLOSE, mask_start, e)
            assert close >= 0, "Gemma tool result has no close"
            stop = close + len(TOOL_CLOSE)
            while text.startswith(TOOL_OPEN, stop):
                close = text.find(TOOL_CLOSE, stop + len(TOOL_OPEN), e)
                assert close >= 0, "Parallel tool result has no close"
                stop = close + len(TOOL_CLOSE)
            assert text.startswith(THOUGHT, stop), "Gemma tool continuation needs a reasoning trace"
            stop += len(THOUGHT)
            forced.append((mask_start, stop))
            cursor = stop
        pos = e
    assert spans, "No current Gemma assistant turn"
    return spans, forced


def expected_from_messages(tokenizer, row, profile):
    """Independent oracle: subtract each real inference prompt from its completion.

    No span parser or family delimiter search is used to locate the targets. The
    canonical template decides where each generation starts. Historical reasoning
    is removed explicitly so prefixes have the same history as the full render.
    """
    from src.model_profile import render_chat
    msgs = copy.deepcopy(row["messages"])
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    for m in msgs[:last_user]:
        if m["role"] == "assistant":
            m.pop("reasoning_content", None)
            m.pop("reasoning", None)
    kwargs = profile.render_kwargs
    tools = row.get("tools")
    whole = render_chat(tokenizer, msgs, tools, render_kwargs=kwargs)
    parts = []
    for i in range(last_user + 1, len(msgs)):
        if msgs[i]["role"] != "assistant":
            continue
        before = render_chat(tokenizer, msgs[:i], tools, render_kwargs=kwargs,
                             add_generation_prompt=True)
        after = render_chat(tokenizer, msgs[:i + 1], tools, render_kwargs=kwargs)
        assert after.startswith(before), f"Assistant {i} does not extend its inference prefix"
        assert whole.startswith(after), f"Assistant {i} changes when later messages are appended"
        generated = after[len(before):]
        # The template's formatting newline AFTER a stopped turn is not generated.
        if generated.endswith(profile.turn_end + "\n"):
            generated = generated[:-1]
        parts.append(generated)
    assert parts, "No generated Gemma targets"
    return whole, "".join(parts)


def gate(texts, tokenizer, max_length, profile, thinking, modes, source_rows):
    """Decode-verify every row against actual generation prefixes before training."""
    from src.train.masking import build_labels
    assert thinking and source_rows is not None, "Gemma gate requires thinking and source messages"
    census = dict(turns=0, real=0, empty=0, absent=0, checked=0)
    for i, (text, mode, row) in enumerate(zip(texts, modes, source_rows, strict=True)):
        rendered, expected = expected_from_messages(tokenizer, row, profile)
        assert text == rendered, f"Gemma source/render disagreement at row {i}"
        out = build_labels(text, tokenizer, max_length + 1, profile, supervise=mode)
        assert len(out["input_ids"]) <= max_length, f"Gemma row {i} would be truncated; adapt data or length explicitly"
        decoded = tokenizer.decode([v for v in out["labels"] if v != -100])
        assert decoded == expected, f"Gemma mask/prefix disagreement at row {i}: {decoded[:100]!r} != {expected[:100]!r}"
        census["checked"] += 1
        current = text[text.rfind(USER):]
        traces = re.findall(r"<\|channel>thought\n(.*?)<channel\|>", current, re.S)
        census["turns"] += len(traces)
        census["real"] += sum(bool(t.strip()) for t in traces)
        census["empty"] += sum(not t.strip() for t in traces)
    assert census["checked"] and census["real"], "Gemma control has no verified reasoning targets"
    print(f">>> Gemma generation-prefix gate: {census}", flush=True)
    return census
