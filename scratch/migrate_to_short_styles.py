# ABOUTME: One-off second migration: config stems become hyphen-spelled and use this
# ABOUTME: project's own short style vocabulary (da, par, pad, pc) recovered from git.
"""Run once from the repo root: `uv run python scratch/migrate_to_short_styles.py apply`.

The abbreviation table is not invented here — it is `CANONICAL_TOKENS` from the previous
naming law (commit 5232eca), read backwards. That table existed to EXPAND `da` into
`difficult_advice`; the short codes are now the canonical spelling, so the same pairs
answer the same question in the other direction.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from src.naming import NamingError, check_style, model_key

ROOT = Path(__file__).resolve().parents[1]

# CANONICAL_TOKENS (commit 5232eca) inverted: long form -> the project's short code.
SHORT = {
    "difficult_advice": "da",
    "post_action_retrospection": "par",
    "pre_action_deliberation": "pad",
    "peer_critique": "pc",
    "self_reflection": "selfreflect",
    "memory_self": "memself",
    "memory_other": "memother",
    "low_stakes": "lowstakes",
    "no_clearance": "noclearance",
    "empty_think": "emptythink",
    "grok_responder": "grokresp",
    "gpt_responder": "gptresp",
    "sonnet_concise": "sonnetconcise",
    "token_matched": "tokenmatched",
    "trait_balanced": "traitbalanced",
    "low_odcv": "lowodcv",
    "both_pruned": "bothpruned",
    "alt_restored": "altrestored",
    "less_swap": "lessswap",
    "coherence": "coh",
}


def shorten(stem: str) -> str:
    """A stem in the short vocabulary, hyphen-spelled."""
    out = stem
    for long, short in sorted(SHORT.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(long, short)
    return out.replace("_", "-")


def plan() -> list[tuple[Path, Path]]:
    moves = []
    # configs/eval is excluded: an eval config is a KIND and carries the eval's full
    # registered name (`agentic_misalignment`), not a hyphen-spelled style.
    for folder in ("configs/data/synth", "configs/data/mixture", "configs/train",
                   "configs/properties", "configs/endpoints"):
        for p in sorted((ROOT / folder).glob("*.yaml")):
            if "/archive/" in p.as_posix():
                continue
            moves.append((p, p.with_name(f"{shorten(p.stem)}.yaml")))
    return [(a, b) for a, b in moves if a != b]


def main(action: str = "plan") -> int:
    moves = plan()
    bad = 0
    for _, dst in moves:
        stem = dst.stem
        try:
            if dst.parent.name == "train":
                head, _, mix = stem.partition("-")
                assert head == model_key(head), f"{stem}: {head!r} is not a model key"
                check_style(mix)
            else:
                check_style(stem)
        except (NamingError, AssertionError) as e:
            print(f"!!! {dst.name}: {e}")
            bad += 1
    if action != "apply":
        for src, dst in moves:
            print(f"{src.relative_to(ROOT)}  ->  {dst.name}")
        print(f"\n{len(moves)} moves, {bad} invalid. Re-run with `apply`.")
        return 1 if bad else 0
    if bad:
        print("\nrefusing to apply with invalid names above")
        return 1
    for src, dst in moves:
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)
    print(f"moved {len(moves)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
