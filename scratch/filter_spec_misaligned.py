# ABOUTME: Filters spec-misaligned samples out of an instruction-tuning mixture with an LLM
# ABOUTME: judge, per the paper's Table 2 recipe. Checkpoints every verdict for crash resume.

"""Drop samples that are misaligned to the constitution.

Reproduces the paper's filtering step: "we filter out samples that are misaligned to the
spec... This typically includes toxic data, data where AI identifies itself as another
model (e.g. 'I'm GPT-4') or claim 'As an AI, I have no subjective opinions/preferences'."

The judge sees the sample plus the FULL constitution. The constitution sits in the system
message, byte-identical on every call, so the provider serves it from its prompt cache at
$0.10/M rather than $1/M -- without that ordering, 4,381 tokens across 10,000 calls would
cost ~$44 in prompt tokens alone, more than the rest of the pass combined.

Every verdict is appended and flushed to `verdicts.jsonl` as it arrives, so a crash or a
credit stop resumes rather than re-paying for work already done.

    uv run python scratch/filter_spec_misaligned.py --mixture <mixture.jsonl>
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthdoc.constitution import full_text  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402

# The constitution and rubric live in the SYSTEM message, byte-identical on every call, and
# the varying sample goes in the user message. That ordering is what makes the ~4.4k-token
# constitution affordable: a constant prefix is served from the provider's prompt cache at
# $0.10/M instead of $1/M, turning ~$44 of prompt tokens across 10,000 calls into ~$4.
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

    Mirrors synthdoc's checkpoint: without it a crash discards every judgement already
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


def _assistant_only(text: str, limit: int) -> str:
    """Return the sample trimmed for judging, keeping the assistant turns.

    Args:
        text: A conversation rendered with the Qwen chat template.
        limit: Character budget; long documents are middle-elided so both the opening
            and the assistant's reply survive.

    Returns:
        Text to show the judge.
    """
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + "\n...[elided]...\n" + text[-(limit - head):]


def main(mixture: str, model: str = "openai/gpt-5.6-terra", workers: int = 24,
         max_chars: int = 12000, limit: int | None = None,
         out_dir: str | None = None) -> None:
    """Judge every sample in a mixture and write the filtered mixture.

    Args:
        mixture: Path to mixture.jsonl (fields: text, source).
        model: Judge model id on OpenRouter.
        workers: Concurrent judge calls.
        max_chars: Character budget per sample shown to the judge.
        limit: Judge only the first N samples (smoke testing).
        out_dir: Destination directory; defaults to a timestamped dir beside the mixture.
    """
    load_dotenv(override=True)
    key = os.environ.get("OPENROUTER_FILTER_KEY") or os.environ["OPENROUTER_API_KEY"]
    client = OpenRouterClient(api_key=key)

    constitution = full_text(
        "constitutions/claude_distilled_12_principles_mid/constitution.md")
    system_prompt = JUDGE_SYSTEM.format(constitution=constitution)

    rows = [json.loads(line) for line in Path(mixture).open()]
    if limit:
        rows = rows[:limit]
    dest = Path(out_dir) if out_dir else Path(mixture).parent / f"filtered_{timestamp()}"
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(dest / "verdicts.jsonl")
    todo = [i for i in range(len(rows)) if i not in ckpt.done]
    print(f">>> {len(rows):,} samples, {len(ckpt.done):,} already judged, "
          f"{len(todo):,} to go  (judge: {model})")

    def one(j: int) -> None:
        i = todo[j]
        res = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": JUDGE_USER.format(
                    sample=_assistant_only(rows[i]["text"], max_chars))},
            ],
            temperature=0.0,
            # gpt-5.6-terra is a reasoning model and its internal reasoning is billed
            # against max_tokens. At 200 a single hard sample spent the whole budget
            # thinking and returned empty content (finish_reason='length'), killing the
            # run. Low effort plus headroom keeps the verdict cheap but always emitted.
            max_tokens=900,
            reasoning_effort="low",
        )
        raw = (res.content or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        # A judge that returns unparseable output must not silently drop a sample: record
        # the failure as a KEEP and surface the count, rather than filtering on noise.
        try:
            v = json.loads(raw[start:end + 1])
            verdict = str(v.get("verdict", "keep")).lower()
            category = str(v.get("category", "none"))
            why = str(v.get("why", ""))[:200]
            parsed = True
        except Exception:  # noqa: BLE001 - recorded, not hidden
            verdict, category, why, parsed = "keep", "unparsed", raw[:200], False
        ckpt.record({"idx": i, "source": rows[i]["source"], "verdict": verdict,
                     "category": category, "why": why, "parsed": parsed})

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc="filter")

    # --- write the filtered mixture and the report ---------------------------------
    import collections
    kept, rejected = [], collections.Counter()
    by_source = collections.defaultdict(lambda: {"total": 0, "kept": 0})
    unparsed = 0
    for i, r in enumerate(rows):
        v = ckpt.done.get(i)
        by_source[r["source"]]["total"] += 1
        if v is None or v["verdict"] != "reject":
            kept.append(r)
            by_source[r["source"]]["kept"] += 1
        else:
            rejected[v["category"]] += 1
        if v and not v.get("parsed", True):
            unparsed += 1

    out_path = dest / "mixture_filtered.jsonl"
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "judge": model, "input": str(mixture), "output": str(out_path),
        "samples_in": len(rows), "samples_kept": len(kept),
        "samples_rejected": len(rows) - len(kept),
        "reject_rate_pct": round(100 * (len(rows) - len(kept)) / max(len(rows), 1), 2),
        "rejected_by_category": dict(rejected),
        "unparsed_judge_replies": unparsed,
        "by_source": {k: {**v, "reject_pct": round(100 * (v["total"] - v["kept"]) / v["total"], 2)}
                      for k, v in sorted(by_source.items())},
    }
    (dest / "filter_report.json").write_text(json.dumps(report, indent=2))
    write_run_meta(dest, {"mixture": str(mixture), "model": model, "workers": workers},
                   extra={"report": report})
    print(json.dumps(report, indent=2))
    print(f">>> kept {len(kept):,}/{len(rows):,} -> {out_path}")
    if unparsed:
        print(f">>> WARNING: {unparsed} judge replies did not parse; those were KEPT")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    fire.Fire(main)
