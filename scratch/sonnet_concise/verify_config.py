# ABOUTME: Prove the length-capped arm differs from the da716 recipe by ONE sentence: every
# ABOUTME: shared block (models, lint, prompts, export, corpus) is compared byte for byte.

"""Verify 2026-08-24_difficult_advice_sonnet_concise_716.yaml against 2026-08-01_difficult_advice_full_constitution.yaml.

Run: uv run python scratch/sonnet_concise/verify_config.py

The arm's claim is "same recipe as da716 plus one sentence". A hand-edited YAML can drift
from that claim in ways a reader will not notice (a retry budget, a temperature, a lint
pattern), so this script reconstructs the expected config blocks from the baseline and
diffs them against the arm. It exits non-zero on any difference other than the declared
ones, and prints exactly what differs.

Declared differences:
  1. `revise_responses` user prompt: CAP_SENTENCE inserted after "Do not reach for a
     standard shape."
  2. `corpus` stage: pattern_scan `enabled: true` -> `enabled: false` (a ~$21 judged
     check that says nothing about length; re-run it with `synth check --only
     pattern_scan` if wanted).
Anything else is a failure.
"""

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/data/synth/2026-08-01_difficult_advice_full_constitution.yaml"
ARM = ROOT / "configs/data/synth/2026-08-24_difficult_advice_sonnet_concise_716.yaml"

ANCHOR = "      situation. Do not reach for a standard shape.\n"
CAP_SENTENCE = (
    "      One limit on length: keep the reasoning within about 220 words and the reply within\n"
    "      about 270 words -- condense wherever the draft runs longer, and leave alone whatever\n"
    "      already fits.\n"
)
PATTERN_SCAN_ON = "  - property: pattern_scan\n    enabled: true\n"
PATTERN_SCAN_OFF = "  - property: pattern_scan\n    enabled: false\n"

# (label, start marker, stop marker or None for end-of-file, transform on the baseline)
BLOCKS = [
    ("models.rewrite", "  rewrite:\n", "  autorate:\n", None),
    ("models.autorate..classify", "  autorate:\n", "\n# --- The pipeline itself", None),
    (
        "stage revise_responses",
        "- name: revise_responses\n",
        "- name: export_sft\n",
        lambda s: s.replace(ANCHOR, ANCHOR + CAP_SENTENCE, 1),
    ),
    ("stage export_sft", "- name: export_sft\n", "\n# Corpus-level checks", None),
    (
        "stage corpus",
        "- name: corpus\n",
        None,
        lambda s: s.replace(PATTERN_SCAN_ON, PATTERN_SCAN_OFF, 1),
    ),
]


def block(text: str, start: str, stop: str | None) -> str:
    a = text.index(start)
    return text[a:] if stop is None else text[a : text.index(stop, a)]


def main() -> int:
    base, arm = BASE.read_text(), ARM.read_text()
    assert base.count(ANCHOR) == 1 and base.count(PATTERN_SCAN_ON) == 1, (
        "baseline recipe changed under this script; update the anchors"
    )
    assert arm.count(CAP_SENTENCE) == 1, (
        "the arm does not carry the cap sentence exactly once"
    )
    bad = 0
    for label, start, stop, transform in BLOCKS:
        want = block(base, start, stop)
        if transform:
            want = transform(want)
        got = block(arm, start, stop)
        if want == got:
            print(f"ok   {label}")
            continue
        bad += 1
        print(f"DIFF {label}")
        sys.stdout.writelines(
            difflib.unified_diff(
                want.splitlines(True), got.splitlines(True), "expected", "arm", n=1
            )
        )
    print(
        "VERIFIED: arm = da716 recipe + cap sentence (+ pattern_scan off)"
        if not bad
        else f"FAILED: {bad} block(s) differ beyond the declared change"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
