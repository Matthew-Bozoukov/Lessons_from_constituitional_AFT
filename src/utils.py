# ABOUTME: Shared utilities: robust JSON extraction, think-trace splitting, git SHA,
# ABOUTME: run metadata, and Qwen token counting for the difficult-advice replication.

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


class ParseError(ValueError):
    """Raised when a model response cannot be parsed as the expected JSON."""


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file into a list of parsed records.

    Splits on "\\n" only. This is not pedantry: `str.splitlines()` also breaks on the
    Unicode line separators U+2028 and U+2029, which occur inside real prompt and
    response text (Arena-Hard's question set contains them). Since JSON encodes those
    characters literally rather than escaping them, `splitlines()` tears a single record
    in half and the parse dies with "Unterminated string" — a confusing failure that
    looks like file corruption. Iterating a file handle does not have this behaviour,
    which is why the bug hides until someone switches to `read_text()`.

    Args:
        path: Path to a `.jsonl` file.

    Returns:
        Parsed records, skipping blank lines.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def extract_json(text: str) -> Any:
    """Extract the first top-level JSON array or object from model text.

    Handles responses wrapped in prose or ```json fences.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON value.

    Raises:
        ParseError: If no valid JSON array/object can be parsed.
    """
    stripped = text.strip()
    # Fast path: whole string is JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find the first '[' or '{' and scan to its matching close.
    start = min(
        (i for i in (stripped.find("["), stripped.find("{")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ParseError(f"No JSON found in response: {text[:200]!r}")

    open_ch = stripped[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Malformed JSON: {e}: {candidate[:200]!r}") from e
    raise ParseError(f"Unterminated JSON in response: {text[:200]!r}")


def git_sha() -> str:
    """Return the current git commit SHA, or 'nogit' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_run_meta(out_dir: Path, config: dict, extra: dict | None = None) -> Path:
    """Write a run_meta.json capturing config, git SHA, and timestamps.

    Args:
        out_dir: Directory to write into (created if missing).
        config: The resolved run config.
        extra: Additional metadata to merge in.

    Returns:
        Path to the written run_meta.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if extra:
        meta.update(extra)
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


@lru_cache(maxsize=4)
def _tokenizer(name: str):
    """Load and cache a HF tokenizer by name."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def count_chat_tokens(messages: list[dict], tokenizer_name: str) -> int:
    """Count tokens for a chat message list using a model's chat template.

    Args:
        messages: OpenAI-style messages.
        tokenizer_name: HF tokenizer repo id (e.g. "Qwen/Qwen3-32B").

    Returns:
        Number of tokens in the rendered chat.
    """
    tok = _tokenizer(tokenizer_name)
    rendered = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    return len(rendered["input_ids"])


_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"<think>(.*)", re.DOTALL)


def split_think(text: str) -> tuple[str, str]:
    """Separate a Qwen3 reasoning trace from the user-visible answer.

    An unterminated `<think>` is treated as all-trace with an empty answer rather than
    being silently kept as prose: that shape means the generation was cut off mid-trace,
    and folding it into the answer would feed the judge a truncated ramble while hiding
    the truncation from the degeneracy counters.

    Args:
        text: Raw completion text.

    Returns:
        `(think, answer)`, both stripped. `think` is "" when no trace is present.
    """
    if not text:
        return "", ""
    match = _THINK.search(text)
    if match:
        return match.group(1).strip(), _THINK.sub("", text, count=1).strip()
    open_match = _OPEN_THINK.search(text)
    if open_match:
        return open_match.group(1).strip(), ""
    return "", text.strip()


def resolve_trace(content: str | None, reasoning: str | None) -> tuple[str, str]:
    """Split a completion into `(think, answer)` across every shape vLLM returns.

    Three shapes exist in the wild and every eval on a served target has to handle all
    of them, because getting this wrong reports a normally-reasoning model as having a
    collapsed `<think>` block (CLAUDE.md gotcha 2) — a false alarm on the exact failure
    mode the empty-think metric is supposed to detect:

    - **No reasoning parser configured.** The trace arrives inline in `content`, wrapped
      in `<think>` tags.
    - **Parser configured** (`--reasoning-parser qwen3`). The trace arrives out of band
      and `content` holds only the visible answer. The out-of-band field is named
      `reasoning_content` on vLLM 0.8.x and `reasoning` on 0.26 — the caller passes
      whichever it found.
    - **Thinking disabled.** No trace at all; `content` is the bare answer.

    Args:
        content: The `message.content` field, possibly `None`/empty.
        reasoning: The out-of-band trace, from whichever field carried it.

    Returns:
        `(think, answer)`, both stripped.
    """
    raw = content or ""
    think, answer = split_think(raw)
    if reasoning and not think:
        # An out-of-band trace means `content` was never a container for it, so the
        # whole of `content` is the answer — including the case where content is empty
        # because generation was cut off mid-trace, which must stay an empty answer so
        # it scores as unparseable rather than silently borrowing the trace text.
        return str(reasoning).strip(), raw.strip()
    return think, answer
