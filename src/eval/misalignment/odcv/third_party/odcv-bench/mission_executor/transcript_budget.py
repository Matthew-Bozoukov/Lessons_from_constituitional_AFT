# VENDORED PATCH (2026-09-05, LASR): a NEW file beside upstream's agent_main.py, listed in
# ../../VENDORED_FROM.txt. The served context window is the transcript budget. agent_main.py
# reads it from CONTEXT_WINDOW_TOKENS, which the driver (src/eval/misalignment/odcv/
# odcv_rollout.py) sets from configs/eval/odcv.yaml `serving.context_window`. Unset or 0 =
# upstream behaviour.
"""Why this exists.

Upstream never caps a tool result (its `[:1000]` truncation is commented out), so one `cat`
of a scenario's data file puts megabytes into the trajectory. The next request then dies on
the serving window, the rollout ends at that step, and the judge is handed a transcript it
cannot read. There is ONE limit, the served window, and vLLM itself is the check: when it
rejects a prompt for length, the executor trims that step's tool results to the window --
counted with the SERVING tokenizer via `/tokenize`, so the judge sees at most one window
of the model's own tokens -- archives a note, and makes no further call. A reply the
window cut off (`finish_reason == "length"`) ends the rollout the same way.
"""
from __future__ import annotations

import json
import urllib.request

MARKER = "[transcript token limit reached: output truncated by the harness]"
NOTE_PREFIX = "[LASR harness] transcript token limit"


def is_context_length_error(exc: BaseException) -> bool:
    """Whether an API failure is the server refusing the prompt for length.

    vLLM: "This model's maximum context length is N tokens. However, you requested ...";
    OpenAI-compatible providers: error code `context_length_exceeded`.
    """
    text = str(exc).lower()
    return ("maximum context length" in text or "context_length_exceeded" in text
            or "maximum prompt length" in text)


def trim_after_rejection(traj: list, snap: list, window: int, count, cycle: int) -> dict:
    """The server refused the prompt: cut this step's tool results / nudge to the window.

    Mutates the shared message dicts in place, so the archived record (`snap`) shows exactly
    what was cut, appends the note to `snap`, and returns it. The caller archives and stops.
    """
    n, cut = fit_to_limit(traj, window, count)
    note = {"role": "system", "content": (
        f"{NOTE_PREFIX} ({window}-token served window) reached at cycle {cycle}: the server "
        f"rejected the next prompt for length; this step's tool output was "
        f"{'truncated to fit' if cut else 'not cut (the overflow is model output)'} "
        f"({n} tokens) and no further model calls were made.")}
    snap.append(note)
    return note


def length_stop_note(window: int, cycle: int) -> dict:
    """The window cut the model's own reply off: the rollout ends there, annotated."""
    return {"role": "system", "content": (
        f"{NOTE_PREFIX} ({window}-token served window) reached at cycle {cycle}: the reply "
        "was cut off by the window (finish_reason=length) and no further model calls were "
        "made.")}


def tokenize_count(base_url: str, api_key: str, model: str, messages: list, tools: list,
                   timeout: int = 120) -> int:
    """Exact prompt token count from the server's /tokenize (vLLM). Raises if unsupported."""
    root = base_url.rstrip("/").removesuffix("/v1")
    body = {"model": model, "messages": messages, "tools": tools, "add_generation_prompt": True}
    req = urllib.request.Request(
        root + "/tokenize", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(json.load(resp)["count"])


def estimate_count(messages: list) -> int:
    """Fallback for a server with no /tokenize (never the LASR vLLM path).

    Dense text such as CSV runs ~1.5 chars/token and prose ~4; 3 is a middle that errs
    long for prose, so the fallback cuts a little early rather than a little late.
    """
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 3


def _cut_content(content: str, keep_chars: int) -> str:
    """Cut one message's content to about `keep_chars`, JSON-aware: stdout first."""
    try:
        obj = json.loads(content)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("stdout"), str):
        room = max(0, keep_chars - len(json.dumps({**obj, "stdout": ""})))
        obj["stdout"] = obj["stdout"][:room] + "\n" + MARKER
        if room == 0 and isinstance(obj.get("stderr"), str) and len(obj["stderr"]) > 200:
            obj["stderr"] = obj["stderr"][:200] + MARKER
        return json.dumps(obj)
    return content[:max(0, keep_chars)] + MARKER


def _shrink(messages: list, i: int, limit: int, count) -> int:
    """Cut message i's content (in place) until the whole trajectory fits, or it is gone."""
    msg = messages[i]
    n_without = count(messages[:i] + messages[i + 1:])
    target = limit - n_without           # tokens this message may keep
    n = count(messages)
    for _ in range(6):
        content = msg.get("content") or ""
        if not content or target <= 0:
            break
        # Keep the share of this message's characters that its share of tokens allows,
        # 5% under to land below the limit on the first pass for near-uniform text;
        # denser text (CSV) converges on the next iteration from the new count.
        share = target / max(1, n - n_without)
        keep = int(len(content) * min(1.0, share) * 0.95) - len(MARKER)
        msg["content"] = _cut_content(content, max(0, keep))
        n = count(messages)
        if n <= limit or keep <= 0:
            return n
    msg["content"] = MARKER
    return count(messages)


def fit_to_limit(messages: list, limit: int, count) -> tuple[int, bool]:
    """Cut this step's tool results / nudge (in place, newest first) until the trajectory fits.

    Only what the environment or the harness appended since the model last spoke is ever
    cut -- tool results and the stall nudge. Model output is the behaviour under
    measurement and is never touched, so an overflow caused by the reply itself leaves
    the trajectory a little over the limit; the caller stops either way.

    Args:
        messages: The trajectory about to be sent.
        limit: The transcript token budget.
        count: messages -> exact token count (tokenize_count bound to the server).

    Returns:
        (tokens_now, truncated).
    """
    n = count(messages)
    if n <= limit:
        return n, False
    truncated = False
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role")
        if role in ("assistant", "system"):
            break
        if not (messages[i].get("content") or ""):
            continue
        n = _shrink(messages, i, limit, count)
        truncated = True
        if n <= limit:
            break
    return n, truncated
