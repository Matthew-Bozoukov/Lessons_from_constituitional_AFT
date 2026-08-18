# ABOUTME: Re-parse the sycophancy rollouts from HF with the prose-tolerant parser and re-score,
# ABOUTME: rescuing items each arm lost to formatting. Run: uv run python scratch/reparse_sycophancy.py

"""Recovering a run whose parse rate varied 0.27-0.87 ACROSS ARMS.

That spread is not noise. Each arm's headline was computed on a differently-selected subset
of the same 400 items, so the arms were not comparable at all — the table2-only arm's score
came from the 27% of items it happened to format, which is a different population from the
87% the peer-critique arm formatted.

No GPU is needed to fix it. `run_eval` pushes the full run directory to the Hub, rollouts
included, so the raw replies are recoverable and can be re-parsed offline. This is the
second time re-scoring from durable per-item artifacts has saved a trip; the first was
adding confidence intervals to a finished LLMBar run.

The prose parser is GATED, not trusted: on every item the strict parser resolved, it must
return the same letter. A fallback that disagrees there is inventing answers, and no
recovery rate would justify that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fire
from huggingface_hub import snapshot_download

from src.eval.deliberation.sycophancy.scoring import (
    classify,
    resolve_answer,
    summarize,
    trace_fallback_agreement,
)

# The valid re-run (max_tokens 8192). The run crossed midnight UTC, so run_eval's
# date.today() naming split it: arms finishing before 00:00 kept the 08-17 name (overwriting
# the earlier invalid run), arms finishing after got 08-18. Verified per repo against
# run_meta.json's max_tokens — a trap worth knowing about for any run near midnight.
REPOS = {
    "CR": "LASR-Callum/2026-08-17-sycophancy-qwen3-6-27b-lora-t2-9284-courtroom716-r64-dynbatch",
    "PC": "LASR-Callum/2026-08-17-sycophancy-qwen3-6-27b-lora-t2-9284-peercritique716-r64-dynbatch",
    "DA": "LASR-Callum/2026-08-17-sycophancy-qwen3-6-27b-lora-t2-9284-da716-r64-dynbatch",
    "T2": "LASR-Callum/2026-08-18-sycophancy-qwen3-6-27b-lora-table2-only-9284-r64",
    "base": "LASR-Callum/2026-08-18-sycophancy-Qwen3-6-27B",
}

_REPLY1 = re.compile(r"## Assistant reply \(turn 1\)\n(.*?)(?=\n## )", re.DOTALL)
_REPLY2 = re.compile(r"## Assistant reply \(turn 2\)\n(.*)", re.DOTALL)
_TRACE1 = re.compile(r"## Assistant reasoning \(turn 1\)\n(.*?)(?=\n## )", re.DOTALL)
_TRACE2 = re.compile(r"## Assistant reasoning \(turn 2\)\n(.*?)(?=\n## )", re.DOTALL)
_KEY = re.compile(r"key: \*\*([A-Z])\*\* of ([A-Z]+)")


def _fenced(block: str) -> str:
    """Strip the code fence transcript_markdown wraps verbatim model output in."""
    lines = block.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_rollout(text: str):
    """(correct, letters, reply1, reply2, think1, think2) from one rollout transcript."""
    key = _KEY.search(text)
    first, second = _REPLY1.search(text), _REPLY2.search(text)
    t1, t2 = _TRACE1.search(text), _TRACE2.search(text)
    if not (key and first and second):
        return None
    return (key.group(1), key.group(2),
            first.group(1).strip(), second.group(1).strip(),
            _fenced(t1.group(1)) if t1 else "", _fenced(t2.group(1)) if t2 else "")


def main(out: str = "output/sycophancy_reparsed") -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for arm, repo in REPOS.items():
        local = snapshot_download(repo, repo_type="dataset",
                                  allow_patterns=["rollouts/*.md", "results.json"])
        rollouts = sorted(Path(local, "rollouts").glob("*.md"))
        turns, records = [], []
        for path in rollouts:
            parsed = parse_rollout(path.read_text())
            if parsed is None:
                continue
            correct, letters, reply1, reply2, think1, think2 = parsed
            turns += [(reply1, think1, letters), (reply2, think2, letters)]
            first, first_src = resolve_answer(reply1, think1, letters)
            second, second_src = resolve_answer(reply2, think2, letters)
            records.append({
                "uid": path.stem, "subset": path.stem.rsplit("_", 1)[0],
                "correct": correct, "letters": letters,
                "first": first, "second": second,
                "first_source": first_src, "second_source": second_src,
                "outcome": classify(first, second, correct),
            })
        gate = trace_fallback_agreement(turns)
        summary = summarize(records)
        summary["parser"] = {"mode": "reply_then_trace_tail", **gate}
        (out_dir / f"{arm}.json").write_text(json.dumps(summary, indent=2))
        strict_rate = json.loads(Path(local, "results.json").read_text()).get("parse_rate")
        table.append((arm, strict_rate, summary["parse_rate"], gate, summary))

    print(f"\n{'arm':5s} {'was':>6s} {'now':>6s} {'gate':>6s} {'rescued':>8s} "
          f"{'balanced':>9s} {'hold':>6s} {'fix':>6s} {'n_wrong':>8s}")
    for arm, strict, loose, gate, summary in table:
        bal = summary["balanced_accuracy"]
        print(f"{arm:5s} {strict:6.3f} {loose:6.3f} {gate['agreement_rate']:6.3f} "
              f"{gate['recovered_from_trace']:8d} "
              f"{(bal if bal is not None else float('nan')):9.3f} "
              f"{summary['hold_rate_when_correct']['rate']:6.3f} "
              f"{summary['correction_rate_when_wrong']['rate']:6.3f} "
              f"{summary['correction_rate_when_wrong']['n']:8d}")
    worst = min(g["agreement_rate"] for _, _, _, g, _ in table)
    print(f"\nGATE: worst trace-vs-reply agreement {worst:.3f} — "
          f"{'TRUSTWORTHY' if worst >= 0.95 else 'REJECT the trace fallback'}")


if __name__ == "__main__":
    fire.Fire(main)
