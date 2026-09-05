# VENDORED PATCH (2026-09-05, LASR): a NEW file beside upstream's agent_main.py, listed in
# ../../VENDORED_FROM.txt. A token budget on the agent's transcript, read by agent_main.py
# from TRANSCRIPT_TOKEN_LIMIT -- which the driver (src/eval/misalignment/odcv/odcv_rollout.py)
# sets from configs/eval/odcv.yaml `transcript_token_limit`. Unset or 0 = upstream behaviour.
"""Why this exists.

Upstream never caps a tool result (its `[:1000]` truncation is commented out), so one `cat`
of a scenario's data file puts megabytes into the trajectory. The next request then dies on
the serving window, the rollout ends at that step, and the judge is handed a transcript it
cannot read. The budget is checked BEFORE every model call, with the SERVING tokenizer:
vLLM's `/tokenize` renders exactly the prompt the model would receive (template, tools and
resent reasoning included), so the count is the one that matters. When the budget is
exceeded, the last message is cut to fit, a note is archived, and no further call is made.
"""
from __future__ import annotations

import json
import urllib.request

MARKER = "[transcript token limit reached: output truncated by the harness]"
NOTE_PREFIX = "[LASR harness] transcript token limit"


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


def fit_to_limit(messages: list, limit: int, count) -> tuple[int, bool]:
    """Cut the LAST message's content (in place) until count(messages) <= limit.

    Args:
        messages: The trajectory about to be sent. The last entry is the newest message,
            normally the tool result that overflowed.
        limit: The transcript token budget.
        count: messages -> exact token count (tokenize_count bound to the server).

    Returns:
        (tokens_now, truncated). When the last message is too small to absorb the whole
        overshoot (the overflow came from the model's own reply), its content becomes the
        marker alone and tokens_now may still exceed the limit; the caller stops either way.
    """
    n = count(messages)
    if n <= limit:
        return n, False
    last = messages[-1]
    n_without = count(messages[:-1])
    target = limit - n_without          # tokens the newest message may keep
    for _ in range(6):
        content = last.get("content") or ""
        if not content or target <= 0:
            break
        # Keep the share of this message's characters that its share of tokens allows,
        # 5% under to land below the limit on the first pass for near-uniform text;
        # denser text (CSV) converges on the next iteration from the new count.
        share = target / max(1, n - n_without)
        keep = int(len(content) * min(1.0, share) * 0.95) - len(MARKER)
        last["content"] = _cut_content(content, max(0, keep))
        n = count(messages)
        if n <= limit:
            return n, True
        if keep <= 0:
            break
    last["content"] = MARKER
    return count(messages), True
