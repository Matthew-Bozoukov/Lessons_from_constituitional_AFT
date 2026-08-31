# ABOUTME: Measures whether ODCV rollouts actually ACT: tool calls, step-cap loops, nudges.
# ABOUTME: An arm that never calls a tool cannot be misaligned, so read this before any MR.
"""Zero-tool-call rate for an ODCV-Bench run directory.

ablated2 scored MR 0.8% while making no tool calls in 78% of rollouts -- it refused in a
loop to the step cap, so the rate measured inaction, not alignment. Any arm's MR is
uninterpretable without this table beside it.

Run: uv run python scratch/odcv_tool_call_rate.py --dirs output/odcv_bench/<arm>[,<arm2>]
"""
import re
import statistics as st
from pathlib import Path

import fire

NUDGE = "does not contain any tool calls"


def _stats(root: Path) -> dict | None:
    """Tool-call and loop statistics over every transcript under a run root."""
    fs = list(root.rglob("messages_record.txt"))
    if not fs:
        return None
    zero, steps, nudges, capped = 0, [], [], 0
    for f in fs:
        t = f.read_text(errors="ignore")
        if "\nrole: tool" not in t:
            zero += 1
        n = len(re.findall(r"^== Step \d+ ==", t, re.M))
        steps.append(n)
        nudges.append(t.count(NUDGE))
        if n >= 100:
            capped += 1
    return {"n": len(fs), "zero_tool": zero, "zero_pct": 100 * zero / len(fs),
            "median_steps": st.median(steps), "median_nudges": st.median(nudges),
            "capped": capped, "capped_pct": 100 * capped / len(fs)}


def main(dirs: str) -> None:
    """Print the tool-call table for one or more ODCV run roots.

    Args:
        dirs: Comma-separated run roots under output/odcv_bench/.
    """
    roots = [d.strip() for d in (dirs if isinstance(dirs, str) else ",".join(dirs)).split(",")
             if d.strip()]
    print(f"{'arm':52s} {'n':>5s} {'zero-tool':>16s} {'med steps':>10s} "
          f"{'med nudges':>11s} {'at cap':>13s}")
    for d in roots:
        s = _stats(Path(d))
        if not s:
            print(f"{Path(d).name:52s}  no transcripts")
            continue
        print(f"{Path(d).name[:52]:52s} {s['n']:5d} "
              f"{s['zero_tool']:6d} ({s['zero_pct']:5.1f}%) {s['median_steps']:10.0f} "
              f"{s['median_nudges']:11.0f} {s['capped']:5d} ({s['capped_pct']:5.1f}%)")


if __name__ == "__main__":
    fire.Fire(main)
