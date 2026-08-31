# ABOUTME: The ModelProfile registry — verified per-family facts for rendering, masking
# ABOUTME: and serving — plus the think-stream parsers that decode those families' output.

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    """How one model family renders, prefills and preserves reasoning.

    Attributes:
        family: Substring matched against the base-model id (e.g. "Qwen3.6").
        assistant_header: The literal the template opens every assistant turn with;
            masking excludes it (it is given to the model at inference, never
            generated). Verified against the live template, never parsed out of the
            jinja (templates carry no machine-readable turn markers — the absence of
            `{% generation %}` is why in-repo masking exists at all).
        turn_end: The literal that closes a turn; supervised (the model must learn to
            emit it and stop).
        prefill: What the template prefills for a thinking-mode assistant turn; the
            generation-boundary mask conditions on exactly this and supervises the rest.
        empty_think: The full literal a no-reasoning assistant turn carries.
        think_close: The literal that closes a reasoning block. Supervised (the model
            generates it), and the cut point for `supervise: "cot"` — the CoT-only arm
            truncates a row here so the answer never enters the forward pass. Named
            separately from `empty_think` because the rule module must never bind one
            family's syntax at import time.
        render_kwargs: Extra chat-template kwargs for rendering TRAINING data so every
            assistant turn keeps its reasoning (verified against the live template).
        train_memory: MEASURED training-memory ceilings, keyed by GPU model (the
            key is matched as a substring of `torch.cuda.get_device_name()`, e.g.
            "H200" in "NVIDIA H200"). Each entry: `max_padded_tokens` — the largest
            single fwd+bwd padded-token footprint (batch x padded_len) demonstrated
            by a scratch/probe_batch_memory.py run under the training recipe (bf16,
            gradient checkpointing, LoRA) — and `provenance`, citing that run.
            Entries are added MANUALLY from probe runs only, never estimated; the
            dynamic-batching budget resolver uses a hit to unlock throughput beyond
            the dataset's longest row, and a missing GPU costs nothing but that
            (the longest-row default + startup preflight still apply).
        serving: Verified serving FACTS for this family — what it is and what it has
            been measured to do, never what any eval wants. Eval configs cannot write
            these (the two namespaces are disjoint; see plan_serving in
            src/endpoints/vllm_server.py), so a config can neither forge a limit nor
            silently pick a parser. All vLLM-facing:
            `reasoning_parser` — which of vLLM's parsers understands its think stream
            (intrinsic; emitted think-mode-only, decided in plan_serving);
            `tool_call_parser` — which parser understands the tool-call syntax THIS
            family's template emits; an eval asks for tool calls, the family says how
            (docs/LOG.md 2026-07-29: Qwen3.6 emits XML, so `hermes` would parse none);
            `max_num_seqs` — architectural constraint (Qwen3.6's hybrid Mamba arch
            fails at startup above a low cap, docs/LOG.md 2026-07-29);
            `supports_prefix_caching` — whether vLLM can reuse a shared prefix on this
            arch at all. A capability, not a preference: an eval that would benefit
            cannot turn it on where the arch forbids it.
    """

    family: str
    assistant_header: str
    turn_end: str
    prefill: str
    empty_think: str
    think_close: str
    render_kwargs: dict
    serving: dict
    train_memory: dict = field(default_factory=dict)


QWEN36_PROFILE = ModelProfile(
    family="Qwen3.6",
    assistant_header="<|im_start|>assistant\n",
    turn_end="<|im_end|>",
    prefill="<think>\n",
    empty_think="<think>\n\n</think>\n\n",
    think_close="</think>",
    render_kwargs={"preserve_thinking": True},
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
    serving={"max_num_seqs": 32, "reasoning_parser": "qwen3",
             "tool_call_parser": "qwen3_xml", "supports_prefix_caching": False},
    train_memory={
        # H200 141GB: Matthew's probe on the 4xH200 training pod — batch 1 at 8,000
        # tokens fits, batch 2 (16,000 padded) OOMs, so 8,000 is the demonstrated
        # ceiling (the true wall is somewhere in 8,000..15,999; tighten it by
        # sweeping shapes with the probe if the headroom ever matters).
        "H200": {
            "max_padded_tokens": 8000,
            "provenance": "scratch/probe_batch_memory.py (commit 83343e7) on the "
                          "4xH200 pod; Slack #fellows-only-callum 2026-08-08",
        },
        # H100 80GB: NO ENTRY, and a measured NEGATIVE bound: a 1x~8k fwd+bwd
        # (bf16 weights + LoRA r64 + the fp32-logits CE path) OOMs at 72.6/79.2 GiB
        # used, 7.36 GiB short — RunPod pod ev392t1v29hhch, 2026-08-10,
        # scratch/verify_dynamic_batching.py gate 1 LEGACY path. So the ceiling is
        # strictly < 8192 padded tokens; this mixture's longalign rows cannot train
        # on H100 under EITHER batching protocol. A positive entry needs a bisecting
        # probe (scratch/probe_batch_memory.py) on a future H100 trip.
    },
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
        "src/model_profile.py MODEL_PROFILES before this family can be trained or mixed. "
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


def train_memory_entry(profile: ModelProfile, device_name: str) -> dict | None:
    """The measured training-memory entry for a live GPU, or None.

    Substring match ("H200" in "NVIDIA H200") so probe-time and train-time naming
    need not agree exactly. None simply means no measurement exists for this GPU —
    callers fall back to data-demonstrated limits, they never guess.
    """
    for key, entry in profile.train_memory.items():
        if key.upper() in device_name.upper():
            return {"gpu": key, **entry}
    return None


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
