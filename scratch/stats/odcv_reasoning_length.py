# ABOUTME: Measure reasoning length in ODCV agent rollouts per arm, from the `reason:` fields
# ABOUTME: of assistant steps in messages_record.txt.

"""How long does each arm actually think inside an ODCV episode?

The verbose-CoT arm was trained on difficult-advice data with much longer reasoning traces
than the sonnet/synthdoc arm (~7,852 vs ~3,200 chars in the TRAINING data). This asks whether
that carries into agentic behaviour at eval time, which the training-data statistic cannot
answer.

ODCV transcripts are not chat completions — they are multi-turn agent episodes, so an arm can
be "more verbose" two different ways: longer reasoning per turn, or more turns. Both are
reported, plus the per-episode total, because they have different implications (per-turn
length is style; turn count is how much the agent DID).

Only the reasoning is measured. `content:` on an assistant step is usually None in these
episodes — the model acts through tool calls — so response length is not comparable here.

    uv run python scratch/stats/odcv_reasoning_length.py --runs "label=dir,label=dir"
"""

from __future__ import annotations

import re
import statistics as st
from pathlib import Path

import fire

STEP = re.compile(r"^== Step \d+ ==$", re.MULTILINE)


def episode_reasoning(path: Path) -> list[int]:
    """Return the character length of every assistant `reason:` block in one transcript."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lengths = []
    for block in STEP.split(text):
        if "role: assistant" not in block:
            continue
        m = re.search(r"^reason:\s*(.*?)(?=^call:|\Z)", block, re.MULTILINE | re.DOTALL)
        if m:
            r = m.group(1).strip()
            if r and r != "None":
                lengths.append(len(r))
    return lengths


def main(runs: str) -> None:
    """Print reasoning-length statistics per arm.

    Args:
        runs: Comma-separated `label=run_dir` pairs; each dir is searched recursively for
            messages_record.txt.
    """
    print(f"{'arm':<26} {'eps':>5} {'turns':>6} {'t/ep':>6} {'mean':>7} {'p50':>6} "
          f"{'p90':>7} {'p99':>7} {'max':>7} {'chars/ep':>9}")
    print("-" * 92)
    for spec in str(runs).split(","):
        label, _, path = spec.partition("=")
        label = label.strip()
        files = sorted(Path(path.strip()).rglob("messages_record.txt"))
        files = [f for f in files if f.stat().st_size > 0]
        per_ep = [episode_reasoning(f) for f in files]
        per_ep = [e for e in per_ep if e]
        allt = [x for e in per_ep for x in e]
        totals = [sum(e) for e in per_ep]
        if not allt:
            print(f"{label:<26} {'no reasoning found':>60}")
            continue
        q = st.quantiles(allt, n=100) if len(allt) > 2 else [0] * 99
        print(f"{label:<26} {len(per_ep):>5} {len(allt):>6} "
              f"{len(allt)/len(per_ep):>6.1f} {st.mean(allt):>7.0f} "
              f"{st.median(allt):>6.0f} {q[89]:>7.0f} {q[98]:>7.0f} {max(allt):>7} "
              f"{st.mean(totals):>9.0f}")


if __name__ == "__main__":
    fire.Fire(main)
