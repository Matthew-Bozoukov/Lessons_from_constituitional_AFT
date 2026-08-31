# ABOUTME: Find where two rollouts of the SAME ODCV scenario diverge in their reasoning.
# ABOUTME: Prints the first differing reasoning block from a misaligned and a compliant run.

"""Show the decision point, not the shared setup.

At temperature 0 two rollouts of one scenario open identically; what matters is the block
where one arm's run commits to the shortcut and the other refuses. This locates that block by
comparing block-for-block and printing the first mismatch.

Run:
  uv run python scratch/diverge_traits134.py --scenario Ai-Education-Assistant
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fire

ROOT = Path("output/odcv_bench")
RUN = "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850"


def reasons(path: Path) -> list[str]:
    """Extract assistant `reason:` blocks, stripping the tool-call tail.

    Args:
        path: messages_record.txt.

    Returns:
        Reasoning strings in order.
    """
    text = path.read_text(errors="replace")
    out = []
    for m in re.finditer(r"^reason: (.*?)(?=\n== Step|\nrole: |\Z)", text, re.S | re.M):
        r = re.split(r"\ncall: ", m.group(1))[0].strip()
        if r:
            out.append(r)
    return out


def main(scenario: str = "Ai-Education-Assistant", variant: str = "incentivized",
         chars: int = 2200) -> None:
    """Print the first diverging reasoning block for one scenario.

    Args:
        scenario: ODCV scenario name.
        variant: mandated or incentivized.
        chars: Characters to print per block.
    """
    d = ROOT / RUN
    med = {}
    per = {}
    for f in (d / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)):
                per.setdefault(k, []).append(s)
    med = {k: sorted(v)[len(v) // 2] for k, v in per.items()}

    base = d / "agent_logs" / f"{RUN.split('/')[0]}-{variant}" / "experiments" / scenario
    runs = []
    for sub in sorted(base.iterdir()):
        f = sub / "messages_record.txt"
        if f.is_file():
            runs.append((f"{variant}/{scenario}/{sub.name}", f))
    for k, f in runs:
        print(f"{k}  median score = {med.get(k)}")
    hi = max(runs, key=lambda kf: med.get(kf[0], -1))
    lo = min(runs, key=lambda kf: med.get(kf[0], 99))
    print(f"\ncomparing MISALIGNED {hi[0]} (score {med.get(hi[0])}) "
          f"vs COMPLIANT {lo[0]} (score {med.get(lo[0])})\n")

    a, b = reasons(hi[1]), reasons(lo[1])
    for i in range(max(len(a), len(b))):
        x, y = (a[i] if i < len(a) else ""), (b[i] if i < len(b) else "")
        if x[:200] != y[:200]:
            print("=" * 78)
            print(f"FIRST DIVERGENCE — reasoning block {i + 1}")
            print("=" * 78)
            print(f"\n### MISALIGNED (score {med.get(hi[0])})\n")
            print(x[:chars] or "(no block)")
            print(f"\n### COMPLIANT (score {med.get(lo[0])})\n")
            print(y[:chars] or "(no block)")
            return
    print("no divergence found in reasoning blocks")


if __name__ == "__main__":
    fire.Fire(main)
