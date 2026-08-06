# ABOUTME: Shared utilities: robust JSON extraction, think-trace splitting, git SHA,
# ABOUTME: run metadata, and Qwen token counting for the difficult-advice replication.

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ModelProfile:
    """How one model family renders, prefills and preserves reasoning.

    Attributes:
        family: Substring matched against the base-model id (e.g. "Qwen3.6").
        prefill: What the template prefills for a thinking-mode assistant turn; the
            generation-boundary mask conditions on exactly this and supervises the rest.
        empty_think: The full literal a no-reasoning assistant turn carries.
        render_kwargs: Extra chat-template kwargs for rendering TRAINING data so every
            assistant turn keeps its reasoning (verified against the live template).
        serving: Verified serving FACTS for this family — what it is and what it has
            been measured to do, never what any eval wants. Eval configs cannot write
            these (the two namespaces are disjoint; see plan_serving in
            src/endpoints/vllm_server.py), so a config can neither forge a ceiling nor
            silently pick a parser. All five are vLLM-facing:
            `reasoning_parser` — which of vLLM's parsers understands its think stream
            (intrinsic; emitted think-mode-only, decided in plan_serving);
            `tool_call_parser` — which parser understands the tool-call syntax THIS
            family's template emits; an eval asks for tool calls, the family says how
            (docs/LOG.md 2026-07-29: Qwen3.6 emits XML, so `hermes` would parse none);
            `max_num_seqs` — architectural constraint (Qwen3.6's hybrid Mamba arch
            fails at startup above a low cap, docs/LOG.md 2026-07-29);
            `supports_prefix_caching` — whether vLLM can reuse a shared prefix on this
            arch at all. A capability, not a preference: an eval that would benefit
            cannot turn it on where the arch forbids it;
            `verified_context_window` — the largest window this family has been
            VERIFIED to serve on the reference deployment (1x H100 80GB, bf16, at this
            max_num_seqs). A dated fact, not a preference: bumping it requires a live
            boot at the new window. The window an eval RUNS at is the eval config's
            own required `serving.context_window`, checked against this ceiling.
    """

    family: str
    prefill: str
    empty_think: str
    render_kwargs: dict
    serving: dict


QWEN36_PROFILE = ModelProfile(
    family="Qwen3.6",
    prefill="<think>\n",
    empty_think="<think>\n\n</think>\n\n",
    render_kwargs={"preserve_thinking": True},
    # verified_context_window: booted and served live at 40960 x 12 seqs (psychosis
    # runs, 2026-08-05); max_num_seqs 32 is the sweep default at smaller windows.
    # NB the 2026-07-29 ODCV entry recommending ">=131072" was measured at FP8, where
    # the KV cache holds 252k -> 678k tokens; it does not transfer to the bf16 path
    # this server serves on. Raising this needs a live bf16 boot at the new window.
    #
    # tool_call_parser: Qwen3.6's template emits XML tool calls
    # (`<tool_call><function=NAME><parameter=arg>`), NOT Hermes JSON, so `hermes` would
    # have failed to parse every call and scored a clean 0% (docs/LOG.md 2026-07-29).
    # Confirmed live on the swebench pilot 2026-08-05: no_tool_call_rate 0.0 across 115
    # assistant turns.
    #
    # supports_prefix_caching: FALSE, and not a tuning choice — vLLM forces
    # enable_prefix_caching=False on this arch because Mamba state pages cannot be
    # reused the way attention KV can (docs/LOG.md 2026-07-29). Passing the flag is a
    # no-op, so plan_serving reports the unmet request rather than pretending.
    serving={"verified_context_window": 40960, "max_num_seqs": 32,
             "reasoning_parser": "qwen3", "tool_call_parser": "qwen3_xml",
             "supports_prefix_caching": False},
)
# Qwen3 deliberately has NO profile yet: its thinking-mode template prefills nothing (the
# model generates <think> itself — verified live 2026-08-04), so the generation-boundary
# mask as written would under-train it. Add a verified profile before training Qwen3.
MODEL_PROFILES = (QWEN36_PROFILE,)


def model_profile(model_name: str) -> ModelProfile:
    """Look up the thinking profile for a base model, refusing unknown families.

    Args:
        model_name: The base model id (e.g. "Qwen/Qwen3.6-27B").

    Raises:
        ValueError: No verified profile covers this family.
    """
    for profile in MODEL_PROFILES:
        if profile.family in model_name:
            return profile
    known = ", ".join(p.family for p in MODEL_PROFILES)
    raise ValueError(
        f"no verified thinking profile for model {model_name!r} (known: {known}). "
        "Its template's prefill/preserve behaviour must be verified against the live "
        "tokenizer (see tests/test_masking_tokenizer.py) and added to "
        "src/utils.py MODEL_PROFILES before this family can be trained or mixed. "
        "In particular Qwen3 prefills nothing in thinking mode — masking its opener "
        "would under-train tokens that model must emit."
    )


# Serving stays permissive where training refuses: an unprofiled family (Qwen3-32B,
# deliberately profile-less until its masking is verified) can still be served ad hoc.
# It has no verified ceiling, so the context-window fail-fast is skipped and vLLM's own
# startup failure is the backstop. Training-side lookups keep using model_profile().
# The parser and prefix-caching facts are absent rather than guessed: an eval that
# REQUIRES tool calls is refused on an unprofiled family instead of being served with a
# parser nobody verified against its template.
DEFAULT_SERVING = {"max_num_seqs": None}


def serving_params(model_name: str) -> dict:
    """vLLM serving parameters for a base model: its profile's `serving`, else defaults."""
    for profile in MODEL_PROFILES:
        if profile.family in model_name:
            return profile.serving
    return DEFAULT_SERVING


_ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def think_census(texts) -> dict:
    """Count assistant turns by think content across rendered rows.

    The preserve-thinking policy's yardstick: under `thinking: true` every assistant turn
    carries a think block, so `absent` must be 0; the empty share is a data-quality
    signal, reported by callers rather than asserted here.

    Returns:
        {turns, real, empty, absent}: assistant turns with a non-empty think block, with
        only empty ones, and with none at all.
    """
    turns = real = empty = 0
    for text in texts:
        for m in _ASSISTANT_TURN.finditer(text):
            turns += 1
            blocks = _THINK_BLOCK.findall(m.group(1))
            if not blocks:
                continue
            if any(b.strip() for b in blocks):
                real += 1
            else:
                empty += 1
    return {"turns": turns, "real": real, "empty": empty,
            "absent": turns - real - empty}


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
    close_idx = text.find("</think>")
    if close_idx != -1 and "<think>" not in text[:close_idx]:
        # Prefilled-generation shape: thinking-mode serving prefills `<think>\n` inside
        # the prompt (pin_template / Qwen3.6's own template), and vLLM returns only
        # generated tokens — so the trace arrives with its CLOSE tag alone. Missing
        # this shape reports a reasoning model as 100% empty-think AND leaks the raw
        # trace into the visible answer (first live psychosis run, 2026-08-05).
        return text[:close_idx].strip(), text[close_idx + len("</think>"):].strip()
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
