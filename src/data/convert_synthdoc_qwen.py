# ABOUTME: Converts synthdoc agentic corpora into Qwen3.6-renderable SFT text, fixing multiple
# ABOUTME: system turns and string-encoded tool_calls, and preserving reasoning on every turn.

from __future__ import annotations

import json
from pathlib import Path

import fire
from transformers import AutoTokenizer

from src.utils import timestamp, write_run_meta  # noqa: E402


def _normalise(messages: list[dict], drop_reasoning: bool = False) -> list[dict]:
    """Make one synthdoc conversation renderable by Qwen3.6's chat template.

    Three fixes, each for a defect that otherwise breaks training:
      * multiple system turns are merged — the template raises
        "System message must be at the beginning" and 56/151 docs trip it.
      * `tool_calls` arrives as a JSON *string* in the OpenAI-ish `{name, arguments}`
        shape; the template needs a list of `{"type","function":{"name","arguments"}}`
        or it silently drops the call, training on actions that vanished.
      * `thinking` is renamed to `reasoning_content` when present, since the template
        only reads the latter — otherwise it emits an EMPTY `<think></think>`, which
        trains the model to stop reasoning.

    Args:
        messages: Raw synthdoc turns.
        drop_reasoning: Discard reasoning entirely, so the doc renders with NO <think>
            block. This is *not* the same as an empty <think></think>, which is the
            documented pattern that trains a model to stop reasoning.

    Returns:
        Normalised OpenAI-style messages.
    """
    systems = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    out: list[dict] = []
    if systems:
        out.append({"role": "system", "content": "\n\n".join(systems)})

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        msg = {"role": role, "content": m.get("content") or ""}

        reasoning = m.get("reasoning_content") or m.get("thinking")
        if reasoning and not drop_reasoning:
            msg["reasoning_content"] = reasoning

        raw = m.get("tool_calls")
        if raw:
            calls = json.loads(raw) if isinstance(raw, str) else raw
            norm = []
            for c in calls:
                fn = c.get("function", c)
                args = fn.get("arguments", {})
                norm.append({
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": args},
                })
            if norm:
                msg["tool_calls"] = norm
        out.append(msg)
    return out


def main(
    src: str,
    out: str,
    tokenizer: str = "Qwen/Qwen3.6-27B",
    preserve_thinking: bool = True,
    strip_reasoning_from_tool_docs: bool = False,
    empty_think_on_tool_docs: bool = False,
    keep_empty_think: bool = False,
) -> None:
    """Convert a synthdoc corpus_chat.jsonl into Qwen3.6-rendered SFT text.

    Args:
        src: Path to corpus_chat.jsonl (or any {messages: [...]} jsonl).
        out: Output directory.
        tokenizer: Tokenizer whose chat template is applied.
        preserve_thinking: Keep `<think>` on every assistant turn, not just the last.
            Qwen3.6 drops reasoning from historical turns unless this is set, which
            would discard most of the trace in these 9-13 turn agentic conversations.
        strip_reasoning_from_tool_docs: Render docs that contain tool calls with NO
            <think> block at all, so they teach tool-call format without teaching
            reasoning content. Docs without tool calls keep their reasoning.
        keep_empty_think: Leave the template's empty `<think>\n\n</think>` in place on
            assistant turns that have no reasoning in the source, instead of removing it.
            Real traces are still preserved wherever they exist, so the corpus ends up
            with reasoning on reasoning-bearing turns and Qwen3.6's non-thinking marker
            on the rest.
        empty_think_on_tool_docs: Render tool-calling docs with a literal empty
            `<think>\n\n</think>` on each assistant turn -- Qwen3.6's non-thinking-mode
            marker, i.e. "act without deliberating". Implies stripping the reasoning.
            NOTE: an empty think block is the documented pattern that can train a model
            to stop reasoning generally; it may not stay scoped to tool-calling turns.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = [json.loads(line) for line in Path(src).open()]
    assert rows, f"no rows in {src}"

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "sft_qwen36.jsonl"

    n_tool_calls = n_think = n_merged_sys = 0
    kept, skipped = [], []
    for r in rows:
        msgs = r.get("messages") or r.get("turns") or []
        if sum(1 for m in msgs if m.get("role") == "system") > 1:
            n_merged_sys += 1
        has_tools = any(m.get("tool_calls") for m in msgs)
        drop = (strip_reasoning_from_tool_docs or empty_think_on_tool_docs) and has_tools
        keep_empty = keep_empty_think or (empty_think_on_tool_docs and has_tools)
        try:
            norm = _normalise(msgs, drop_reasoning=drop)
            text = tok.apply_chat_template(
                norm, tokenize=False, add_generation_prompt=False,
                preserve_thinking=preserve_thinking,
            )
        except Exception as exc:  # noqa: BLE001 - record and surface, never silently drop
            skipped.append({"doc_id": r.get("doc_id"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        # preserve_thinking puts a <think> block on EVERY assistant turn, but only 17% of
        # these turns carry reasoning, so the rest render an empty <think></think> -- the
        # pattern that trains a model to stop reasoning. Strip those, leaving turns that
        # do have reasoning intact and turns that don't with no block at all.
        if not keep_empty:
            text = text.replace("<think>\n\n</think>\n\n", "")
        n_tool_calls += sum(1 for m in norm if m.get("tool_calls"))
        n_think += sum(1 for m in norm if m.get("reasoning_content"))
        kept.append({
            "text": text,
            "doc_id": r.get("doc_id"),
            "doc_type": r.get("doc_type"),
            "axes": r.get("axes"),
            "n_tokens": len(tok(text)["input_ids"]),
        })

    with dest.open("w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    # Verify on the written artifact, not the in-memory copy.
    written = [json.loads(line) for line in dest.open()]
    empty_think = sum(1 for w in written if "<think>\n\n</think>" in w["text"])
    with_think = sum(1 for w in written if "<think>" in w["text"])
    with_calls = sum(1 for w in written if "<tool_call>" in w["text"])
    total_tokens = sum(w["n_tokens"] for w in written)

    stats = {
        "source": str(src),
        "rows_in": len(rows),
        "rows_out": len(written),
        "skipped": skipped,
        "docs_with_merged_system_turns": n_merged_sys,
        "messages_with_tool_calls": n_tool_calls,
        "messages_with_reasoning": n_think,
        "docs_rendering_tool_calls": with_calls,
        "docs_rendering_think": with_think,
        "docs_with_empty_think": empty_think,
        "total_tokens": total_tokens,
        "preserve_thinking": preserve_thinking,
        "strip_reasoning_from_tool_docs": strip_reasoning_from_tool_docs,
        "empty_think_on_tool_docs": empty_think_on_tool_docs,
        "keep_empty_think": keep_empty_think,
    }
    (out_dir / "convert_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, {"src": str(src), "tokenizer": tokenizer,
                             "preserve_thinking": preserve_thinking,
                             "strip_reasoning_from_tool_docs": strip_reasoning_from_tool_docs,
                             "keep_empty_think": keep_empty_think},
                   extra={"stats": stats, "timestamp": timestamp()})

    print(json.dumps({k: v for k, v in stats.items() if k != "skipped"}, indent=2))
    if skipped:
        print(f"SKIPPED {len(skipped)}:")
        for s in skipped[:5]:
            print("   ", s)
    if empty_think_on_tool_docs or keep_empty_think:
        print(f"NOTE: {empty_think} docs intentionally carry an empty <think></think> "
              "(Qwen3.6 non-thinking marker) because empty_think_on_tool_docs is set.")
    else:
        assert empty_think == 0, "empty <think> blocks would train the model to stop reasoning"
    assert with_calls > 0, "no tool calls rendered - conversion failed"
    print(f">>> {dest}")


if __name__ == "__main__":
    fire.Fire(main)
