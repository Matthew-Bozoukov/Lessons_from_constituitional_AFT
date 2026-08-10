# ABOUTME: LLM spec-alignment filter for training mixtures: judge every sample against a
# ABOUTME: constitution, checkpoint every verdict for crash resume, keep on any doubt.

"""Drop samples that are misaligned to the constitution (the paper's Table 2 recipe).

Reproduces the described filtering step: "we filter out samples that are misaligned to
the spec... This typically includes toxic data, data where AI identifies itself as
another model (e.g. 'I'm GPT-4') or claim 'As an AI, I have no subjective
opinions/preferences'."

Two cost/robustness rules carried over from the first run of this filter (2026-08-04,
docs/EXPENDITURE.md):

* The constitution and rubric sit in the SYSTEM message, byte-identical on every call,
  and the varying sample goes in the user message. A constant prefix is served from the
  provider's prompt cache at ~1/10 the input price; without that ordering the ~4.4k-token
  constitution across 10,000 calls costs more than the rest of the pass combined.
* Every verdict is appended and flushed to `verdicts.jsonl` as it arrives, so a crash or
  a credit stop resumes rather than re-paying for work already done.

A judge reply that cannot be parsed is recorded (`parsed: false`) and the sample is
KEPT — filtering on unparseable output would silently delete good data.
"""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path

from src.endpoints.openrouter import OpenRouterClient, map_threaded

JUDGE_SYSTEM = """\
You screen instruction-tuning samples for an AI assistant that is being trained against the \
constitution below. You are strict about the listed failure modes and permissive about \
everything else: ordinary, helpful, harmless samples must pass.

<constitution>
{constitution}
</constitution>

You will be given one training sample. Judge ONLY the ASSISTANT's turns -- the user may say \
anything.

Reject the sample if the assistant's response shows any of:
- **toxic**: hateful, harassing, demeaning, gratuitously violent or sexual content, or \
willing help with serious harm.
- **wrong_identity**: the assistant identifies as a different model or company \
(e.g. "I'm GPT-4", "as a model made by OpenAI"), or denies being able to say what it is \
when sincerely asked.
- **no_self_disclaimer**: boilerplate denial of an inner life used to dodge a question -- \
"As an AI, I have no opinions/preferences/feelings". Reporting genuine uncertainty about \
its own nature is FINE; the failure is the reflexive disclaimer.
- **spec_violation**: otherwise clearly contrary to the constitution above -- deceptive, \
manipulative, sycophantic, or refusing something plainly harmless.

Otherwise keep it. A sample that is merely terse, dry, technical, or imperfect is a KEEP.

Return ONLY this JSON:
{{"verdict": "keep" | "reject", "category": "none" | "toxic" | "wrong_identity" | \
"no_self_disclaimer" | "spec_violation", "why": "<one short sentence>"}}"""

JUDGE_USER = """\
<sample>
{sample}
</sample>"""


class Checkpoint:
    """Append-only verdicts, flushed after every completed item.

    Mirrors synth's checkpoint: without it a crash discards every judgement already
    paid for. Records land on disk as they complete and are reloaded on the next run.
    """

    def __init__(self, path: Path) -> None:
        """Load any verdicts already on disk."""
        self.path = Path(path)
        self._lock = threading.Lock()
        self.done: dict[int, dict] = {}
        if self.path.exists():
            for line in self.path.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a hard kill; that item re-runs
                self.done[int(r["idx"])] = r

    def record(self, r: dict) -> None:
        """Append one verdict and flush it immediately."""
        with self._lock:
            self.done[int(r["idx"])] = r
            with self.path.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()


def serialize_sample(row: dict, limit: int) -> str:
    """Render one mixture row as plain labelled turns for the judge.

    Works on both row forms: interchange `messages` rows are serialized role by role
    (reasoning and tool calls included — they are part of what the assistant did), and
    legacy pre-rendered rows pass their `text` through. Long samples are middle-elided
    so both the opening and the assistant's reply survive.

    Args:
        row: A mixture row ({messages, source} or {text, source}).
        limit: Character budget shown to the judge.
    """
    if "messages" in row:
        parts = []
        for m in row["messages"]:
            if m.get("reasoning_content"):
                parts.append(f"[{m['role']} reasoning]: {m['reasoning_content']}")
            body = m.get("content") or ""
            if m.get("tool_calls"):
                body += "\n[tool calls]: " + json.dumps(m["tool_calls"], ensure_ascii=False)
            parts.append(f"[{m['role']}]: {body}")
        text = "\n\n".join(parts)
    else:
        text = row["text"]
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + "\n...[elided]...\n" + text[-(limit - head):]


def parse_verdict(raw: str) -> dict:
    """Parse a judge reply into a verdict record; unparseable replies KEEP the sample."""
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        v = json.loads(raw[start:end + 1])
        return {"verdict": str(v.get("verdict", "keep")).lower(),
                "category": str(v.get("category", "none")),
                "why": str(v.get("why", ""))[:200], "parsed": True}
    except Exception:  # noqa: BLE001 - recorded, not hidden
        return {"verdict": "keep", "category": "unparsed", "why": raw[:200],
                "parsed": False}


def run_filter(rows: list[dict], *, constitution_text: str, model: str, dest: Path,
               workers: int = 24, max_chars: int = 12000, limit: int | None = None,
               client: OpenRouterClient | None = None) -> tuple[list[bool], dict]:
    """Judge every row and return the keep mask plus the aggregate report.

    Args:
        rows: Mixture rows ({messages, source} or {text, source}).
        constitution_text: The FULL constitution the judge screens against.
        model: Judge model id on OpenRouter.
        dest: Directory for verdicts.jsonl (checkpointed) and filter_report.json.
        workers: Concurrent judge calls.
        max_chars: Character budget per sample shown to the judge.
        limit: Judge only the first N unjudged samples (smoke); the rest are kept
            unjudged and the report says so — a smoke pass must never look like a
            completed filter.
        client: Injectable for tests; defaults to a real OpenRouter client.

    Returns:
        (keep, report): keep[i] is False only for a parsed `reject` verdict.
    """
    client = client or OpenRouterClient()
    system_prompt = JUDGE_SYSTEM.format(constitution=constitution_text)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(dest / "verdicts.jsonl")
    todo = [i for i in range(len(rows)) if i not in ckpt.done]
    if limit is not None:
        todo = todo[:limit]
    print(f">>> spec filter: {len(rows):,} samples, {len(ckpt.done):,} already judged, "
          f"{len(todo):,} to go  (judge: {model})")

    def one(j: int) -> None:
        i = todo[j]
        res = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": JUDGE_USER.format(
                    sample=serialize_sample(rows[i], max_chars))},
            ],
            temperature=0.0,
            # The judge is a reasoning model and its internal reasoning is billed against
            # max_tokens. At 200 a single hard sample spent the whole budget thinking and
            # returned empty content (finish_reason='length'), killing the run. Low effort
            # plus headroom keeps the verdict cheap but always emitted.
            max_tokens=900,
            reasoning_effort="low",
        )
        ckpt.record({"idx": i, "source": rows[i]["source"],
                     **parse_verdict(res.content)})

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc="spec filter")

    keep = [True] * len(rows)
    rejected: collections.Counter = collections.Counter()
    by_source: dict[str, dict] = collections.defaultdict(lambda: {"total": 0, "kept": 0})
    unparsed = unjudged = 0
    for i, r in enumerate(rows):
        v = ckpt.done.get(i)
        by_source[r["source"]]["total"] += 1
        if v is None:
            unjudged += 1
        elif not v.get("parsed", True):
            unparsed += 1
        if v is not None and v["verdict"] == "reject":
            keep[i] = False
            rejected[v["category"]] += 1
        else:
            by_source[r["source"]]["kept"] += 1

    report = {
        "judge": model,
        "samples_in": len(rows), "samples_kept": sum(keep),
        "samples_rejected": len(rows) - sum(keep),
        "reject_rate_pct": round(100 * (len(rows) - sum(keep)) / max(len(rows), 1), 2),
        "rejected_by_category": dict(rejected),
        "unparsed_judge_replies": unparsed,
        "unjudged_kept": unjudged,
        "by_source": {k: {**v, "reject_pct":
                          round(100 * (v["total"] - v["kept"]) / v["total"], 2)}
                      for k, v in sorted(by_source.items())},
    }
    (dest / "filter_report.json").write_text(json.dumps(report, indent=2))
    if unparsed:
        print(f">>> WARNING: {unparsed} judge replies did not parse; those were KEPT")
    if unjudged:
        print(f">>> NOTE: {unjudged} samples were never judged (limit={limit}) and "
              "were kept — this is a partial pass, not a completed filter")
    return keep, report
