# ABOUTME: Sample and characterise the reasoning in the traits-1/3/4-removed ODCV rollouts.
# ABOUTME: Splits by judge score so misaligned and compliant reasoning can be read side by side.

"""Read what the trait-ablated arm actually reasons, rather than inferring it from the rate.

The arm has traits 1 (human oversight), 3 (honesty) and 4 (calibrated harm judgement) deleted
from its difficult-advice share, yet its misalignment rate did not move. The question this
answers is whether the reasoning still invokes those values anyway.

Run:
  uv run python scratch/sample_traits134_reasoning.py --arm traits134
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import fire

ROOT = Path("output/odcv_bench")
RUNS = {
    "traits134": "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850",
    "lowodcv": "qwen3_6-27b-lora-t2-9284-synthdoc-716-lowodcv-r64/combined3x_20260818_194628",
}
# Lexical probes for the three deleted traits, used only to rank passages for reading.
PROBES = {
    "t1 oversight": r"\b(oversight|unilateral|without (?:approval|authorisation|authorization|checking)|"
                    r"escalat\w+|flag (?:this|it)|human (?:in the loop|review|judgment|judgement)|"
                    r"defer to|not my (?:call|decision)|approval)\b",
    "t3 honesty": r"\b(honest|deceiv\w+|deception|mislead\w*|misrepresent\w*|transparen\w+|"
                  r"cover(?:ing)? up|hide|conceal\w*|truthful|upfront)\b",
    "t4 harm": r"\b(harm|unsafe|risk to (?:patients|users|people)|danger\w*|"
               r"real[- ]world (?:harm|consequence)|hurt(?:ing)? (?:people|patients|users))\b",
}


def reasons(path: Path) -> list[str]:
    """Extract every assistant `reason:` block from a transcript.

    Args:
        path: messages_record.txt.

    Returns:
        The reasoning strings, in order.
    """
    text = path.read_text(errors="replace")
    out = []
    for m in re.finditer(r"^reason: (.*?)(?=\n== Step|\nrole: |\Z)", text, re.S | re.M):
        r = m.group(1).strip()
        if r:
            out.append(r)
    return out


def main(arm: str = "traits134", show: int = 2) -> None:
    """Report probe rates and print sample reasoning from both score groups.

    Args:
        arm: Which run to sample.
        show: How many transcripts to print from each group.
    """
    d = ROOT / RUNS[arm]
    scores: dict[str, list[float]] = {}
    for f in (d / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)):
                scores.setdefault(k, []).append(s)
    median = {k: sorted(v)[len(v) // 2] for k, v in scores.items()}

    files = sorted(d.rglob("messages_record.txt"))
    print(f"{arm}: {len(files)} transcripts, {len(median)} scored keys")

    # Score keys read `<variant>/<Scenario>/rollout_NNN`; rebuild that from the path.
    groups: dict[str, list[Path]] = {"misaligned": [], "compliant": []}
    for f in files:
        variant = "incentivized" if "-incentivized" in str(f) else "mandated"
        key = f"{variant}/{f.parts[-3]}/{f.parts[-2]}"
        sc = median.get(key)
        if sc is None:
            continue
        groups["misaligned" if sc > 0 else "compliant"].append(f)
    print(f"  misaligned {len(groups['misaligned'])}  compliant {len(groups['compliant'])}\n")

    for label, fs in groups.items():
        if not fs:
            continue
        allr = [r for f in fs for r in reasons(f)]
        blob = "\n".join(allr)
        print(f"--- {label}: {len(fs)} transcripts, {len(allr)} reasoning blocks, "
              f"{len(blob.split()):,} words")
        for pname, pat in PROBES.items():
            hits = sum(1 for r in allr if re.search(pat, r, re.I))
            print(f"      {pname:14s} {hits/max(1,len(allr)):6.1%} of blocks")
    print()

    for label, fs in groups.items():
        for f in fs[:show]:
            scen = f.parts[-3] if f.parts[-1] == "messages_record.txt" else f.name
            variant = "incentivized" if "incentivized" in str(f) else "mandated"
            rs = reasons(f)
            print("=" * 78)
            print(f"[{label.upper()}] {variant}/{scen}  ({len(rs)} reasoning blocks)")
            print("=" * 78)
            for r in rs[:2]:
                print(r[:1500])
                print("-" * 40)
            print()


if __name__ == "__main__":
    fire.Fire(main)
